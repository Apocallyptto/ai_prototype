# services/executor_bracket.py
import os
import sys
import time
import math
import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest

# === quotes helper (we added this module earlier) ===
from tools.quotes import get_bid_ask_mid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("executor_bracket")

# === ENV / defaults ===
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@postgres:5432/trader"
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER = (os.getenv("ALPACA_PAPER", "1") == "1")

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))

RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"

# quote-aware limit + spread guard
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))   # percent
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.06"))   # dollars
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")  # optional column filter in signals

# === CLI flags ===
def _arg_or_default(flag: str, default: str) -> str:
    """Return the value after a flag like '--since-min 20', or default string if not provided."""
    for i, tok in enumerate(sys.argv):
        if tok == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default

SINCE_MIN = int(_arg_or_default("--since-min", "20"))
MIN_STRENGTH = float(_arg_or_default("--min-strength", os.getenv("MIN_STRENGTH", "0.50")))

# === DB / Alpaca helpers ===
def _pg_conn():
    return psycopg2.connect(DB_URL)

def _trading_client() -> TradingClient:
    return TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)

def _get_buying_power() -> float:
    """
    Returns account buying power with retries.
    - If API hiccups -> retry and log.
    - If buying_power is 0 but cash exists -> fall back to cash.
    - If FORCE_BP is set (debug), use that value for this run.
    """
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
            bp = float(getattr(acct, "buying_power", 0.0) or 0.0)
            cash = float(getattr(acct, "cash", 0.0) or 0.0)
            if bp <= 0.0 and cash > 0.0:
                log.warning("buying_power reported 0; falling back to cash=%.2f", cash)
                return cash
            return bp
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))

    log.warning("get_account() failed; assuming BP=0.0 | err=%s", last_err)
    return 0.0

def _get_position_qty(symbol: str) -> float:
    try:
        cli = _trading_client()
        for p in cli.get_all_positions():
            if p.symbol.upper() == symbol.upper():
                try:
                    # p.qty is a string per alpaca-py
                    return float(getattr(p, "qty", "0") or 0.0)
                except Exception:
                    return 0.0
    except Exception:
        pass
    return 0.0

def _qty_from_risk(price: float, atr: float) -> int:
    """Size by risk, then cap by buying power."""
    risk_per_trade = RISK_PER_TRADE_USD
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = max(1, int(risk_per_trade / max(sl_dist, 1e-6)))

    # cap by buying power
    bp = _get_buying_power()
    max_bp_shares = int(max(0, bp) // max(price, 0.01))
    if max_bp_shares <= 0:
        return 0
    qty = min(qty_risk, max_bp_shares)
    if qty < qty_risk:
        log.info("qty capped by BP: %s -> %s (bp=%.2f, px=%.2f)", qty_risk, qty, bp, price)
    return qty

# === Signals / ATR ===
def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    sql = """
    SELECT created_at, symbol, side,
           COALESCE(scaled_strength, strength) AS strength,
           COALESCE(px, NULL) AS px
    FROM public.signals
    WHERE created_at >= NOW() - INTERVAL %s
      AND COALESCE(scaled_strength, strength) >= %s
    """
    params = [f"{since_min} minutes", min_strength]
    if PORTFOLIO_ID:
        sql += " AND (portfolio_id = %s OR portfolio_id IS NULL)"
        params.append(PORTFOLIO_ID)
    sql += " ORDER BY created_at DESC"

    with _pg_conn() as conn:
        # pandas warns about raw DBAPI conn; acceptable here
        df = pd.read_sql(sql, conn, params=params)
    return df

def _dedupe_latest_by_symbol(df: pd.DataFrame) -> list[dict]:
    out = {}
    for _, r in df.iterrows():
        sym = r["symbol"]
        if sym not in out:
            out[sym] = {
                "symbol": sym,
                "side": r["side"],
                "strength": float(r["strength"]),
                "px": float(r["px"]) if r["px"] is not None else None,
                "created_at": r["created_at"],
            }
    return list(out.values())

def _compute_atr(symbol: str) -> float:
    """Light ATR calc from recent 5m bars via yfinance; fallback 1.0 on error."""
    try:
        import yfinance as yf
        bars = yf.download(symbol, interval="5m", period="5d", progress=False, auto_adjust=False)
        if bars is None or bars.empty:
            return 1.0
        df = bars.rename(columns=str.lower)[["high", "low", "close"]].copy()
        df["prev_close"] = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=max(2, ATR_PERIOD)).mean().iloc[-1]
        return float(atr) if math.isfinite(atr) else 1.0
    except Exception:
        return 1.0

# === Order submit ===
def _submit_bracket(symbol: str, side: str, qty: int, limit_px: float, tp_px: float, sl_px: float):
    cli = _trading_client()
    side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side_enum,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_px, 2),
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp_px, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl_px, 2)),
    )
    return cli.submit_order(req)

# === Main ===
def main():
    bp = _get_buying_power()
    log.info(
        "executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
        SINCE_MIN, MIN_STRENGTH, bp, str(FRACTIONAL), str(LONG_ONLY)
    )

    df = _fetch_signals(SINCE_MIN, MIN_STRENGTH)
    if df.empty:
        log.info("no qualifying signals in last %d min (>= %.2f)", SINCE_MIN, MIN_STRENGTH)
        return

    sigs = _dedupe_latest_by_symbol(df)
    for s in sigs:
        sym = s["symbol"].upper()
        side = s["side"].lower().strip()
        strength = float(s["strength"])
        px = float(s["px"]) if s["px"] else None

        # position context
        pos_qty = _get_position_qty(sym)
        opening_short = (side == "sell" and pos_qty <= 0)

        # LONG_ONLY or zero-margin protections
        if LONG_ONLY and opening_short:
            log.info("%s: LONG_ONLY=1 -> skip opening short", sym)
            continue
        if opening_short and _get_buying_power() <= 0.0:
            log.info("%s: margin/BP=0 -> skip opening short", sym)
            continue

        # ATR + entry ref price
        atr = _compute_atr(sym)
        if px is None:
            # fallback entry reference: mid or last traded close-ish
            q = get_bid_ask_mid(sym)
            if q:
                _, _, mid = q
                px = float(mid)
        if px is None:
            log.info("%s: cannot determine entry price; skip", sym)
            continue

        # --- price / quotes ---
        quote = get_bid_ask_mid(sym)
        if quote:
            bid, ask, mid = quote
            spread_abs = max(0.0, ask - bid)
            spread_pct = (spread_abs / mid) * 100.0 if mid > 0 else 999.0

            if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
                log.info(
                    "%s: skip due to wide spread (bid=%.4f ask=%.4f mid=%.4f abs=%.4f pct=%.3f%% > limits)",
                    sym, bid, ask, mid, spread_abs, spread_pct
                )
                continue

            # Quote-aware limit target with small slippage guard
            slip = QUOTE_PRICE_SLIPPAGE
            if side == "buy":
                limit_px = min(ask + slip, max(ask, px))
            else:
                limit_px = max(bid - slip, min(bid, px))
        else:
            # Fallback to prior px (less ideal)
            limit_px = round(px, 2)

        # compute tp/sl off entry ref px
        tp_px = px + TP_ATR_MULT * atr if side == "buy" else px - TP_ATR_MULT * atr
        sl_px = px - SL_ATR_MULT * atr if side == "buy" else px + SL_ATR_MULT * atr

        # sizing
        qty = _qty_from_risk(price=px, atr=atr)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        try:
            resp = _submit_bracket(sym, side, qty, limit_px, tp_px, sl_px)
            log.info(
                "submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                sym, side, qty, limit_px, tp_px, sl_px, getattr(resp, "id", "<NA>")
            )
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, str(e))

if __name__ == "__main__":
    main()
