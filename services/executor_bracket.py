# services/executor_bracket.py
import os
import math
import logging
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order

import yfinance as yf
import pandas as pd
import psycopg2

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor_bracket")

# --- Environment ---
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
SINCE_MIN = int(os.getenv("EXEC_SINCE_MIN", "20"))

# Risk management constants
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))

# --- Alpaca helpers ---
def _trading_client() -> TradingClient:
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
        paper=True
    )

# helper: get account buying power
def _get_buying_power() -> float:
    try:
        cli = _trading_client()
        acct = cli.get_account()
        return float(getattr(acct, "buying_power", 0.0))
    except Exception as e:
        log.warning("buying power read failed: %s", e)
        return 0.0

# --- ATR calculation ---
def _atr(h, l, c, n=14):
    hl = (h - l).abs()
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _get_atr(symbol: str, period="5d", interval="5m") -> float:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        atr = _atr(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        return float(atr)
    except Exception as e:
        log.warning("%s ATR failed: %s", symbol, e)
        return 0.0

# --- Risk-based quantity with buying-power cap ---
def _qty_from_risk(price: float, atr: float) -> int:
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = max(1, int(RISK_PER_TRADE_USD / max(sl_dist, 1e-6)))

    # cap by buying power
    bp = _get_buying_power()
    max_bp_shares = int(max(0, bp) // max(price, 0.01))
    if max_bp_shares <= 0:
        return 0

    qty = min(qty_risk, max_bp_shares)
    if qty < qty_risk:
        log.info("qty capped by BP: %s -> %s (bp=%.2f, px=%.2f)", qty_risk, qty, bp, price)
    return qty

# --- Get recent signals from DB ---
def _get_recent_signals():
    sql = f"""
        SELECT symbol, side, strength, px
        FROM signals
        WHERE created_at >= NOW() - INTERVAL '{SINCE_MIN} minutes'
        AND scaled_strength >= {MIN_STRENGTH}
        ORDER BY created_at DESC;
    """
    with psycopg2.connect(DB_URL) as conn:
        return pd.read_sql(sql, conn)

# --- Submit order ---
def _submit_order(cli, sym, side, qty, limit_px, tp_px, sl_px):
    try:
        order = LimitOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type="limit",
            limit_price=limit_px,
            time_in_force=TimeInForce.DAY,
        )
        o = cli.submit_order(order)
        log.info("submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                 sym, side, qty, limit_px, tp_px, sl_px, o.id)
        return o
    except Exception as e:
        log.warning("submit failed %s %s: %s", sym, side, e)
        return None

# --- Main ---
def main():
    log.info("executor_bracket | since-min=%s min_strength=%.2f", SINCE_MIN, MIN_STRENGTH)
    cli = _trading_client()
    df = _get_recent_signals()
    if df.empty:
        log.info("no qualifying signals in last %s min (>= %.2f)", SINCE_MIN, MIN_STRENGTH)
        return

    for _, row in df.iterrows():
        sym = row["symbol"]
        side = row["side"]
        px = float(row["px"]) if not math.isnan(row["px"]) else None
        if not px:
            try:
                px = float(yf.Ticker(sym).fast_info.last_price)
            except Exception:
                continue
        atr = _get_atr(sym)
        qty = _qty_from_risk(px, atr)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        if side == "buy":
            tp_px = px + TP_ATR_MULT * atr
            sl_px = px - SL_ATR_MULT * atr
        else:
            tp_px = px - TP_ATR_MULT * atr
            sl_px = px + SL_ATR_MULT * atr

        _submit_order(cli, sym, side, qty, px, tp_px, sl_px)

if __name__ == "__main__":
    main()
