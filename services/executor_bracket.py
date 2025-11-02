# services/executor_bracket.py
import os, sys, logging, math, time
from decimal import Decimal, ROUND_DOWN
from typing import List, Dict, Any

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 👉 NEW: quote helper (uses Alpaca market data with yfinance fallback)
from tools.quotes import get_bid_ask_mid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("executor_bracket")

# --- Env / defaults ---
TP_ATR_MULT         = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT         = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_PERIOD          = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS   = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
RISK_PER_TRADE_USD  = float(os.getenv("RISK_PER_TRADE_USD", "50"))

MAX_SPREAD_PCT      = float(os.getenv("MAX_SPREAD_PCT", "0.15"))   # in %
MAX_SPREAD_ABS      = float(os.getenv("MAX_SPREAD_ABS", "0.06"))   # in $
QUOTE_SLIP          = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@postgres:5432/trader"

# Flags
FRACTIONAL   = os.getenv("FRACTIONAL", "0") == "1"
LONG_ONLY    = os.getenv("LONG_ONLY", "0") == "1"

def _trading_client() -> TradingClient:
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

# --- replace your _get_buying_power with this robust version ---
def _get_buying_power() -> float:
    """
    Returns account buying power with retries.
    - If API hiccups -> retry and log.
    - If buying_power is 0 but cash exists -> fall back to cash.
    - If FORCE_BP is set (debug), use that value.
    """
    # Debug override for local testing
    if os.getenv("FORCE_BP"):
        try:
            return float(os.getenv("FORCE_BP"))
        except Exception:
            pass

    cli = _trading_client()
    last_err = None
    for attempt in range(3):
        try:
            acct = cli.get_account()
            # Alpaca fields are usually strings
            bp = float(getattr(acct, "buying_power", 0.0) or 0.0)
            cash = float(getattr(acct, "cash", 0.0) or 0.0)
            if bp <= 0.0 and cash > 0.0:
                logging.getLogger("executor_bracket").warning(
                    "buying_power reported 0; falling back to cash=%.2f", cash
                )
                return cash
            return bp
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))

    logging.getLogger("executor_bracket").warning(
        "get_account() failed; assuming BP=0.0 | err=%s", last_err
    )
    return 0.0

def _round_cents(x: float) -> float:
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_DOWN))

def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    sql = """
        SELECT created_at, symbol, side,
               COALESCE(scaled_strength, strength) AS strength,
               COALESCE(px, NULL) AS px
        FROM public.signals
        WHERE created_at >= NOW() - INTERVAL %s
          AND COALESCE(scaled_strength, strength) >= %s
        ORDER BY created_at DESC;
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    return df

def _dedupe_latest_by_symbol(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    # Keep the most recent per-symbol
    df = df.sort_values("created_at", ascending=False).drop_duplicates(subset=["symbol"], keep="first")
    recs = df.to_dict("records")
    # normalize side to 'buy'/'sell'
    for r in recs:
        s = str(r.get("side", "")).lower()
        r["side"] = "buy" if s.startswith("b") else "sell"
    return recs

def _calc_atr(symbol: str) -> float:
    """
    Simple ATR using yfinance 5m bars over lookback days.
    Fallback to a small positive to avoid zero division.
    """
    try:
        import yfinance as yf
        period_days = max(2, ATR_LOOKBACK_DAYS)
        df = yf.download(symbol, interval="5m", period=f"{period_days}d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return 0.25
        df = df.rename(columns=str.lower)
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD, min_periods=max(2, ATR_PERIOD // 2)).mean().iloc[-1]
        return float(max(atr, 0.01))
    except Exception:
        return 0.25

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

def _parse_arg(flag: str, default: str) -> str:
    """
    Robust argv parsing: looks for '--flag value' or '--flag=value'
    """
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default

def main():
    since_min = int(_parse_arg("--since-min", "20"))
    min_strength = float(_parse_arg("--min-strength", os.getenv("MIN_STRENGTH", "0.50")))

    # Account context
    bp = _get_buying_power()
    log.info("executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
             since_min, min_strength, bp, str(FRACTIONAL), str(LONG_ONLY))

    # Fetch signals
    df = _fetch_signals(since_min, min_strength)
    rows = _dedupe_latest_by_symbol(df)
    if not rows:
        log.info("no qualifying signals in last %d min (>= %.2f)", since_min, min_strength)
        return

    cli = _trading_client()

    for r in rows:
        sym = r["symbol"]
        side = r["side"]
        if LONG_ONLY and side == "sell":
            # Skip shorts in long-only mode
            log.info("%s: skip short in LONG_ONLY mode", sym)
            continue

        # Reference price for risk sizing
        px = float(r["px"] or 0.0)
        if px <= 0:
            # fallback to mid
            q = get_bid_ask_mid(sym)
            if q:
                _, _, mid = q
                px = float(mid)

        if px <= 0:
            log.info("%s: no usable reference price; skip", sym)
            continue

        atr = _calc_atr(sym)
        qty = _qty_from_risk(px, atr)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        # --- price / quotes (NEW) ---
        quote = get_bid_ask_mid(sym)
        if quote:
            bid, ask, mid = quote
            spread_abs = max(0.0, ask - bid)
            spread_pct = (spread_abs / mid) * 100.0 if mid > 0 else 999.0

            if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
                log.info("%s: skip due to wide spread (abs=%.4f pct=%.3f%% > limits)", sym, spread_abs, spread_pct)
                continue

            # Quote-aware limit target
            slip = QUOTE_SLIP
            if side == "buy":
                limit_px = min(ask + slip, max(ask, px))
            else:
                limit_px = max(bid - slip, min(bid, px if px else bid))
        else:
            # Fallback to prior px (less ideal)
            limit_px = px

        limit_px = _round_cents(limit_px)

        # Compute TP/SL off *entry ref* px (keep consistent with ATR risk model)
        tp_px = px + TP_ATR_MULT * atr if side == "buy" else px - TP_ATR_MULT * atr
        sl_px = px - SL_ATR_MULT * atr if side == "buy" else px + SL_ATR_MULT * atr
        tp_px, sl_px = _round_cents(tp_px), _round_cents(sl_px)

        # Build & submit primary limit order (TP/SL handled by jobs.manage_exits in your stack)
        try:
            req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=limit_px,                      # 👉 NEW: quote-aware limit
                time_in_force=TimeInForce.DAY
            )
            o = cli.submit_order(req)
            log.info("submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                     sym, side, qty, limit_px, tp_px, sl_px, getattr(o, "id", ""))
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)
            continue

        # You already have jobs.manage_exits doing OCO logic using ATR; keep that path.
        # (If you ever want native OCO attached here, we can wire TakeProfitRequest/StopLossRequest.)

if __name__ == "__main__":
    main()
