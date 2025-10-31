# services/executor_bracket.py
import os, logging, argparse, time
from decimal import Decimal, ROUND_DOWN

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor_bracket")

# ---------- ENV / defaults ----------
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
DEFAULT_MIN_STRENGTH = float(os.getenv("EXEC_MIN_STRENGTH", os.getenv("MIN_STRENGTH", "0.60")))
DEFAULT_SINCE_MIN   = int(os.getenv("EXEC_SINCE_MIN", os.getenv("SINCE_MIN", "20")))

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))

# fractional / policy
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
MIN_QTY = float(os.getenv("MIN_QTY", "0.01"))
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))
ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"  # Alpaca disallows; keep 0
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"  # set 1 to disable shorts entirely

# position limits / dedupe
MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD", "50000"))         # cap per symbol by notional
MAX_SHARES_PER_SYMBOL = float(os.getenv("MAX_SHARES_PER_SYMBOL", "1000"))
OPEN_ORDER_COOLDOWN_SEC = int(os.getenv("OPEN_ORDER_COOLDOWN_SEC", "600"))  # skip if open order exists recently

# ---------- Helpers ----------
def _trading_client() -> TradingClient:
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def _get_account():
    return _trading_client().get_account()

def _get_buying_power() -> float:
    try:
        return float(getattr(_get_account(), "buying_power", 0.0))
    except Exception as e:
        log.warning("buying power read failed: %s", e)
        return 0.0

def _get_position_qty(symbol: str) -> float:
    """Returns signed qty: positive for long, negative for short. 0 if none."""
    try:
        pos = _trading_client().get_open_position(symbol)
        qty = float(getattr(pos, "qty", 0.0))
        # Alpaca returns positive for both long and short with a 'side'; adjust:
        side = str(getattr(pos, "side", "long")).lower()
        return qty if side == "long" else -qty
    except Exception:
        return 0.0

def _has_recent_open_order(symbol: str) -> bool:
    """Best-effort dedupe: if there are open orders for symbol, skip a new one."""
    try:
        # new alpaca-py uses get_orders without 'status' kw; use list + filter
        orders = _trading_client().get_orders()
        now = time.time()
        for o in orders:
            if getattr(o, "symbol", "") == symbol and str(getattr(o, "status", "")).lower() in ("new","accepted","open","pending_new","partially_filled"):
                # crude time guard: if submitted in last cooldown sec, skip
                # many fields are strings; be tolerant
                return True
        return False
    except Exception as e:
        log.warning("get_orders check failed: %s", e)
        return False

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

def _round_qty(q: float) -> float:
    if not FRACTIONAL:
        return int(q)
    return float(Decimal(q).quantize(Decimal("0.01"), rounding=ROUND_DOWN))

def _qty_from_risk(price: float, atr: float) -> float:
    """Return qty (float when FRACTIONAL=1, otherwise int). 0 means 'don’t trade'."""
    if price <= 0:
        return 0
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = max(MIN_QTY if FRACTIONAL else 1, RISK_PER_TRADE_USD / max(sl_dist, 1e-6))
    bp = _get_buying_power()
    max_shares_by_bp = bp / price
    qty = min(qty_risk, max_shares_by_bp)
    qty = _round_qty(qty)
    if qty < (MIN_QTY if FRACTIONAL else 1):
        return 0
    return qty

def _cap_by_position_limits(symbol: str, side: str, qty: float, px: float) -> float:
    """Enforce: (1) notional cap, (2) max shares, (3) available shares for sells."""
    if qty <= 0:
        return 0
    # 1) notional / shares caps for both directions
    if MAX_POSITION_USD > 0:
        max_shares_by_notional = MAX_POSITION_USD / max(px, 0.01)
        qty = min(qty, max_shares_by_notional)
    if MAX_SHARES_PER_SYMBOL > 0:
        qty = min(qty, MAX_SHARES_PER_SYMBOL)

    # 2) For SELL: cannot exceed available long qty (or short add if allowed)
    pos_qty = _get_position_qty(symbol)  # +long / -short
    if side == "sell":
        if LONG_ONLY and pos_qty <= 0:
            return 0
        # If we have a long, we can sell up to that amount
        available = pos_qty if pos_qty > 0 else 0.0
        if FRACTIONAL:
            # fractional: cap with float
            qty = min(qty, available)
        else:
            qty = int(min(qty, available))
        # Also handle Alpaca's "fractional shorts not allowed"
        if FRACTIONAL and qty < 1.0 and not ALLOW_FRACTIONAL_SHORTS:
            # If qty < 1 but we actually have >= 1 to sell, round down handled above
            # If still <1, skip
            return 0
    # Ensure not negative
    if qty <= 0:
        return 0
    return _round_qty(qty)

def _get_recent_signals(since_min: int, min_strength: float):
    eng = create_engine(DB_URL)
    sql = text(f"""
        SELECT DISTINCT ON (symbol)
               created_at, symbol, side,
               COALESCE(scaled_strength, strength) AS strength,
               px
        FROM signals
        WHERE created_at >= NOW() - INTERVAL '{since_min} minutes'
          AND COALESCE(scaled_strength, strength) >= :thr
          AND side IN ('buy','sell')
        ORDER BY symbol, created_at DESC
    """)
    with eng.connect() as con:
        return pd.read_sql_query(sql, con, params={"thr": min_strength})

# ---------- Main ----------
def main():
    # CLI args override envs
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-min", type=int, default=DEFAULT_SINCE_MIN)
    ap.add_argument("--min-strength", type=float, default=DEFAULT_MIN_STRENGTH)
    args = ap.parse_args()

    since_min = args.since_min
    min_strength = args.min_strength

    bp = _get_buying_power()
    log.info("executor_bracket | since-min=%s min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
             since_min, min_strength, bp, FRACTIONAL, LONG_ONLY)
    if bp < MIN_ACCOUNT_BP_USD:
        log.info("account BP < MIN_ACCOUNT_BP_USD (%.2f < %.2f) — skipping this cycle",
                 bp, MIN_ACCOUNT_BP_USD)
        return

    df = _get_recent_signals(since_min, min_strength)
    if df.empty:
        log.info("no qualifying signals in last %s min (>= %.2f)", since_min, min_strength)
        return

    cli = _trading_client()

    for _, row in df.iterrows():
        sym = row["symbol"]; side = row["side"]

        if LONG_ONLY and side == "sell":
            log.info("%s: LONG_ONLY=1 — skip short/exit entry", sym)
            continue

        if _has_recent_open_order(sym):
            log.info("%s: open order exists (cooldown) — skip new entry", sym)
            continue

        # price
        px = float(row["px"]) if pd.notna(row["px"]) else None
        if px is None:
            try:
                px = float(yf.Ticker(sym).fast_info.last_price)
            except Exception:
                log.info("%s: no price; skip", sym); continue

        atr = _get_atr(sym)
        base_qty = _qty_from_risk(px, atr)
        qty = _cap_by_position_limits(sym, side, base_qty, px)
        if qty <= 0:
            log.info("%s: qty after caps is 0 (side=%s, base=%.4f); skip", sym, side, base_qty)
            continue

        # Simple limit entry; exits handled by manage_exits OCO job
        tp_px = px + TP_ATR_MULT * atr if side == "buy" else px - TP_ATR_MULT * atr
        sl_px = px - SL_ATR_MULT * atr if side == "buy" else px + SL_ATR_MULT * atr

        try:
            req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=round(px, 2),
                time_in_force=TimeInForce.DAY
            )
            o = cli.submit_order(req)
            log.info("submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                     sym, side, qty, px, tp_px, sl_px, o.id)
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)

if __name__ == "__main__":
    main()
