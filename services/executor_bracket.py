# services/executor_bracket.py
from __future__ import annotations
import os
import time
import logging
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.common.exceptions import APIError

from tools.util import pg_connect, market_is_open, retry
from tools.quotes import get_bid_ask_mid
from tools.atr import get_atr

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------- ENV ----------
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))
TP_MULT = float(os.getenv("TP_MULT", "1.5"))   # ×ATR take-profit
SL_MULT = float(os.getenv("SL_MULT", "1.0"))   # ×ATR stop-loss

# ---------- Alpaca ----------
def _trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )

def _get_buying_power(cli: TradingClient | None = None) -> float:
    cli = cli or _trading_client()
    try:
        acc = cli.get_account()
        bp = float(getattr(acc, "buying_power", 0) or 0)
        if bp == 0 and getattr(acc, "cash", None):
            log.warning("buying_power reported 0; falling back to cash=%s", acc.cash)
            bp = float(acc.cash)
        return bp
    except Exception as e:
        log.warning("bp fetch failed: %s", e)
        return 0.0

# ---------- market gate ----------
def _market_open() -> bool:
    try:
        return market_is_open()
    except Exception:
        return False

# ---------- DB ----------
def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    conn = pg_connect()
    sql = """
        SELECT s.created_at, s.symbol, s.side,
               COALESCE(s.scaled_strength, s.strength) AS strength,
               s.px
        FROM signals s
        WHERE s.created_at >= NOW() - INTERVAL %s
          AND COALESCE(s.scaled_strength, s.strength) >= %s
        ORDER BY s.created_at DESC
    """
    df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    conn.close()
    return df

# ---------- orders ----------
@retry(tries=3, delay=2)
def _submit_entry_and_attach_oco(cli: TradingClient, symbol: str, side: str, limit_px: float, atr_val: float) -> None:
    """
    Place the entry limit order. If it fills within our small polling window,
    attach an OCO (TP/SL) using ATR-based targets.
    """
    qty = 1
    try:
        entry_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_px,
            extended_hours=False,
        )
        entry = cli.submit_order(entry_req)
        log.info("Entry submitted %s %s @ %.2f", side, symbol, limit_px)

        # poll for fill a few times
        for _ in range(10):
            cur = cli.get_order_by_id(entry.id)
            if str(cur.status).lower() == "filled":
                _place_oco_exit(cli, symbol, side, limit_px, atr_val, qty)
                return
            time.sleep(5)

        log.info("%s not filled yet -> skip OCO attach", symbol)

    except APIError as e:
        log.warning("submit failed %s %s: %s", symbol, side, e)
    except Exception as e:
        log.warning("order submit error %s %s: %s", symbol, side, e)

def _place_oco_exit(cli: TradingClient, symbol: str, entry_side: str, entry_px: float, atr_val: float, qty: int) -> None:
    """
    Place ATR-based OCO exit (TP/SL). NOTE: OCO is an OrderClass, not OrderType.
    """
    try:
        # distance in dollars = multiplier * ATR
        tp_off = TP_MULT * atr_val
        sl_off = SL_MULT * atr_val

        if entry_side == "buy":
            tp_price = round(entry_px + tp_off, 2)
            sl_price = round(entry_px - sl_off, 2)
            oco_side = OrderSide.SELL
        else:
            tp_price = round(entry_px - tp_off, 2)
            sl_price = round(entry_px + sl_off, 2)
            oco_side = OrderSide.BUY

        tp_req = TakeProfitRequest(limit_price=tp_price)
        sl_req = StopLossRequest(stop_price=sl_price)

        # OCO is set via order_class
        oco_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=oco_side,
            type=OrderType.LIMIT,         # TP leg is a limit; SL leg is in stop_loss
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OCO,
            take_profit=tp_req,
            stop_loss=sl_req,
            extended_hours=False,
        )
        cli.submit_order(oco_req)
        log.info("OCO placed %s TP=%.2f SL=%.2f (ATR=%.3f)", symbol, tp_price, sl_price, atr_val)
    except Exception as e:
        log.warning("OCO place failed %s: %s", symbol, e)

# ---------- main ----------
def main(since_min: int = 180, min_strength: float = 0.45) -> None:
    cli = _trading_client()
    bp = _get_buying_power(cli)
    log.info(
        "executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f",
        since_min, min_strength, bp
    )

    if not ALLOW_AFTER_HOURS and not _market_open():
        log.info("market is closed and ALLOW_AFTER_HOURS=0 -> skip this pass")
        return

    if bp < MIN_ACCOUNT_BP_USD:
        log.info("buying_power %.2f < MIN_ACCOUNT_BP_USD %.2f -> skip", bp, MIN_ACCOUNT_BP_USD)
        return

    df = _fetch_signals(since_min, min_strength)
    if df.empty:
        log.info("no qualifying signals in last %d min (>= %.2f)", since_min, min_strength)
        return

    for _, row in df.iterrows():
        symbol = row.symbol
        side = row.side.lower()
        px = float(row.px)

        q = get_bid_ask_mid(symbol)
        if not q:
            log.warning("%s: no quote -> skip", symbol)
            continue

        bid, ask, mid = q
        spread_abs = abs(ask - bid)
        spread_pct = spread_abs / mid * 100.0
        if spread_pct > 0.25:
            log.info("%s: skip wide spread abs=%.4f pct=%.3f%%", symbol, spread_abs, spread_pct)
            continue

        atr_val, _ = get_atr(symbol)
        _submit_entry_and_attach_oco(cli, symbol, side, px, atr_val)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--since-min", type=int, default=180)
    p.add_argument("--min-strength", type=float, default=0.45)
    a = p.parse_args()
    main(a.since_min, a.min_strength)
