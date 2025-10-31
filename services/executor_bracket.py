# services/executor_bracket.py
import os, math, logging
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor_bracket")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
MIN_STRENGTH = float(os.getenv("EXEC_MIN_STRENGTH", os.getenv("MIN_STRENGTH", "0.60")))
SINCE_MIN = int(os.getenv("EXEC_SINCE_MIN", os.getenv("SINCE_MIN", "20")))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))  # ⬅ guard

def _trading_client() -> TradingClient:
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def _get_buying_power() -> float:
    try:
        bp = float(getattr(_trading_client().get_account(), "buying_power", 0.0))
        return bp
    except Exception as e:
        log.warning("buying power read failed: %s", e)
        return 0.0

def _atr(h, l, c, n=14):
    hl = (h - l).abs()
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _get_atr(symbol: str, period="5d", interval="5m") -> float:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        return float(_atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
    except Exception as e:
        log.warning("%s ATR failed: %s", symbol, e)
        return 0.0

def _qty_from_risk(price: float, atr: float) -> int:
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = max(1, int(RISK_PER_TRADE_USD / max(sl_dist, 1e-6)))
    bp = _get_buying_power()
    max_bp_shares = int(max(0, bp) // max(price, 0.01))
    if max_bp_shares <= 0:
        return 0
    qty = min(qty_risk, max_bp_shares)
    if qty < qty_risk:
        log.info("qty capped by BP: %s -> %s (bp=%.2f, px=%.2f)", qty_risk, qty, bp, price)
    return qty

def _get_recent_signals():
    eng = create_engine(DB_URL)
    sql = text(f"""
        SELECT DISTINCT ON (symbol)
               created_at, symbol, side,
               COALESCE(scaled_strength, strength) AS strength,
               px
        FROM signals
        WHERE created_at >= NOW() - INTERVAL '{SINCE_MIN} minutes'
          AND COALESCE(scaled_strength, strength) >= :thr
          AND side IN ('buy','sell')
        ORDER BY symbol, created_at DESC
    """)
    with eng.connect() as con:
        return pd.read_sql_query(sql, con, params={"thr": MIN_STRENGTH})

def main():
    bp = _get_buying_power()
    log.info("executor_bracket | since-min=%s min_strength=%.2f | buying_power=%.2f",
             SINCE_MIN, MIN_STRENGTH, bp)
    if bp < MIN_ACCOUNT_BP_USD:
        log.info("account BP < MIN_ACCOUNT_BP_USD (%.2f < %.2f) — skipping this cycle",
                 bp, MIN_ACCOUNT_BP_USD)
        return

    cli = _trading_client()
    df = _get_recent_signals()
    if df.empty:
        log.info("no qualifying signals in last %s min (>= %.2f)", SINCE_MIN, MIN_STRENGTH)
        return

    for _, row in df.iterrows():
        sym = row["symbol"]; side = row["side"]
        px = float(row["px"]) if pd.notna(row["px"]) else None
        if px is None:
            try:
                px = float(yf.Ticker(sym).fast_info.last_price)
            except Exception:
                log.info("%s: no price; skip", sym); continue

        atr = _get_atr(sym)
        qty = _qty_from_risk(px, atr)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        tp_px = px + TP_ATR_MULT * atr if side == "buy" else px - TP_ATR_MULT * atr
        sl_px = px - SL_ATR_MULT * atr if side == "buy" else px + SL_ATR_MULT * atr
        try:
            req = LimitOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=round(px, 2),
                time_in_force=TimeInForce.DAY
            )
            o = _trading_client().submit_order(req)
            log.info("submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                     sym, side, qty, px, tp_px, sl_px, o.id)
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)

if __name__ == "__main__":
    main()
