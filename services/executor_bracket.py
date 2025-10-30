# services/executor_bracket.py
import os, sys, logging, time, math
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from services.notify import notify_order_submitted

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor_bracket")

# --- ENV CONFIG ---
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "paper")

# --- ARGUMENTS (robust parsing) ---
def get_arg(flag: str, default: str) -> str:
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return default

SINCE_MIN = int(get_arg("--since-min", "20"))
MIN_STRENGTH = float(get_arg("--min-strength", "0.60"))

# --- ATR SETTINGS ---
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET", "")
APCA_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")


# ---------- HELPERS ----------
def _fetch_signals() -> List[Dict]:
    sql = """
    SELECT created_at, symbol, side,
           COALESCE(scaled_strength, strength) AS strength,
           COALESCE(px, NULL) px
    FROM public.signals
    WHERE created_at >= NOW() - INTERVAL %s
      AND COALESCE(scaled_strength, strength) >= %s
      AND side IN ('buy','sell')
      AND (portfolio_id = %s OR %s = '')
    ORDER BY created_at DESC
    """
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(sql, (f"{SINCE_MIN} minutes", MIN_STRENGTH, PORTFOLIO_ID, PORTFOLIO_ID))
        rows = cur.fetchall()
    return [
        {
            "created_at": r[0],
            "symbol": r[1],
            "side": r[2],
            "strength": float(r[3]),
            "px": (float(r[4]) if r[4] is not None else None),
        }
        for r in rows
    ]


def _latest_price(symbol: str) -> Optional[float]:
    try:
        return float(yf.Ticker(symbol).fast_info.last_price)
    except Exception:
        return None


def _download_5m(symbol: str, days: int) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=f"{days}d", interval="5m", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            return None
        return df.rename(columns=str.lower)
    except Exception as e:
        log.warning("bars fail %s: %s", symbol, e)
        return None


def _atr(h, l, c, n=14):
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def _compute_atr(symbol: str) -> Optional[float]:
    bars = _download_5m(symbol, ATR_LOOKBACK_DAYS)
    if bars is None or bars.empty:
        return None
    return float(_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD).iloc[-1])


def _dedupe_latest_by_symbol(signals: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for s in signals:
        key = (s["symbol"], s["side"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _qty_from_risk(symbol: str, price: float, atr: float) -> int:
    risk_per_trade = float(os.getenv("RISK_PER_TRADE_USD", "50"))
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty = max(1, int(risk_per_trade / sl_dist / max(0.01, price)))
    return qty


# ---------- ALPACA ----------
def _trading_client() -> TradingClient:
    return TradingClient(ALPACA_KEY, ALPACA_SEC, paper="paper" in APCA_BASE)


def _place_bracket(symbol: str, side: str, limit_price: float, tp: float, sl: float, qty: int) -> Optional[str]:
    client = _trading_client()
    req = LimitOrderRequest(
        symbol=symbol,
        qty=str(qty),
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        limit_price=round(limit_price, 2),
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit={"limit_price": round(tp, 2)},
        stop_loss={"stop_price": round(sl, 2)},
        extended_hours=False,
    )
    try:
        order = client.submit_order(req)
        return order.id
    except Exception as e:
        log.warning("submit failed %s %s: %s", symbol, side, e)
        return None


# ---------- MAIN ----------
def main():
    log.info("executor_bracket | since-min=%s min_strength=%.2f", SINCE_MIN, MIN_STRENGTH)
    sigs = _dedupe_latest_by_symbol(_fetch_signals())
    if not sigs:
        log.info("no qualifying signals found in last %s min (≥ %.2f)", SINCE_MIN, MIN_STRENGTH)
        return

    for s in sigs:
        sym = s["symbol"]
        side = s["side"]
        px = s["px"] or _latest_price(sym)
        if px is None:
            log.info("%s: no price; skip", sym)
            continue

        atr = _compute_atr(sym)
        if atr is None:
            log.info("%s: no ATR; skip", sym)
            continue

        # Bracket math
        if side == "buy":
            tp = px + TP_ATR_MULT * atr
            sl = px - SL_ATR_MULT * atr
        else:
            tp = px - TP_ATR_MULT * atr
            sl = px + SL_ATR_MULT * atr

        qty = _qty_from_risk(sym, px, atr)
        order_id = _place_bracket(sym, side, px, tp, sl, qty)
        if order_id:
            notify_order_submitted({
                "symbol": sym, "side": side, "qty": qty,
                "limit": px, "tp": tp, "sl": sl, "id": order_id
            })
            log.info("submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                     sym, side, qty, px, tp, sl, order_id)

if __name__ == "__main__":
    main()
