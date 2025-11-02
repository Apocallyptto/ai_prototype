# services/executor_bracket.py
import os
import sys
import math
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import psycopg2

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest

from tools.quotes import get_bid_ask_mid  # spread/quotes helper

log = logging.getLogger("executor_bracket")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# =========================
# Env / runtime parameters
# =========================

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/trader")

# executor CLI defaults (used if flags not passed)
EXEC_SINCE_MIN = int(os.getenv("EXEC_SINCE_MIN", "20"))
EXEC_MIN_STRENGTH = float(os.getenv("EXEC_MIN_STRENGTH", "0.50"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

# risk / exits
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.00"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.50"))

# share policy
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"
ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"  # normally False
MIN_QTY = float(os.getenv("MIN_QTY", "0.01")) if FRACTIONAL else 1.0

# execution precision / liquidity guards
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))  # %
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.06"))  # $
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))  # $ add/subtract from bid/ask

# new: market-hours & min BP guards
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))

# misc
ATR_INTERVAL = os.getenv("ATR_INTERVAL", "5m")
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK = os.getenv("ATR_LOOKBACK", "5d")  # yfinance period for ATR calc

# =========================
# Alpaca client helpers
# =========================

def _trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )

def _market_open() -> bool:
    """RTH gate: only trade when the market is open (unless ALLOW_AFTER_HOURS=1)."""
    try:
        cli = _trading_client()
        clk = cli.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        # Be conservative if we cannot check.
        return False

def _get_buying_power() -> float:
    """Read account buying power; if Alpaca returns '0', fall back to cash."""
    try:
        cli = _trading_client()
        acct = cli.get_account()
        bp = float(getattr(acct, "buying_power", 0.0) or 0.0)
        if bp <= 0:
            cash = float(getattr(acct, "cash", 0.0) or 0.0)
            log.warning("buying_power reported 0; falling back to cash=%.2f", cash)
            return cash
        return bp
    except Exception:
        return 0.0

# =========================
# DB / signals
# =========================

def _db_conn():
    return psycopg2.connect(DB_URL)

def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    """
    Pull recent signals; prefer normalized score if present.
    Expected columns: created_at, symbol, side, px (nullable), strength (scaled or raw)
    """
    sql = """
        SELECT created_at,
               symbol,
               side,
               COALESCE(scaled_strength, strength) AS strength,
               COALESCE(px, NULL) AS px
        FROM public.signals
        WHERE created_at >= NOW() - INTERVAL %s
          AND COALESCE(scaled_strength, strength) >= %s
        ORDER BY created_at DESC
    """
    with _db_conn() as conn:
        # Pandas warns on raw DBAPI; that's fine for us.
        df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    return df

def _dedupe_latest_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # keep the most recent per symbol
    return df.sort_values("created_at").groupby("symbol", as_index=False).tail(1)

# =========================
# Indicators / sizing
# =========================

def _get_atr(symbol: str, period: int = ATR_PERIOD, interval: str = ATR_INTERVAL, lookback: str = ATR_LOOKBACK) -> float:
    """Lightweight ATR using yfinance; returns a reasonable fallback if unavailable."""
    try:
        import yfinance as yf
        df = yf.download(symbol, interval=interval, period=lookback, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return 0.5  # fall back
        df = df.rename(columns=str.lower)
        h, l, c = df["high"], df["low"], df["close"]
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        atr = tr.rolling(window=period).mean().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return max(0.10, 0.0025 * float(c.iloc[-1]))
        return float(atr)
    except Exception:
        # Simple px-based fallback (~0.25% of price, min $0.10)
        try:
            px_guess = float(get_bid_ask_mid(symbol)[2])
        except Exception:
            px_guess = 100.0
        return max(0.10, 0.0025 * px_guess)

def _qty_from_risk(price: float, atr: float, bp: float) -> float:
    """Position sizing from risk per trade, capped by buying power. Returns qty (can be fractional)."""
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = RISK_PER_TRADE_USD / max(sl_dist, 1e-6)
    if not FRACTIONAL:
        qty_risk = math.floor(qty_risk)

    # Cap by buying power (price * qty <= bp)
    if price <= 0:
        return 0.0
    max_by_bp = bp / price
    qty = min(qty_risk, max_by_bp)
    # enforce min qty
    if qty < (MIN_QTY if FRACTIONAL else 1.0):
        return 0.0
    if not FRACTIONAL:
        qty = math.floor(qty)
    return float(qty)

# =========================
# Order submission
# =========================

def _submit_bracket(sym: str, side: str, limit_px: float, atr: float):
    """Submit a limit + bracket (TP/SL as separate legs)."""
    cli = _trading_client()
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Calculate TP/SL off the *reference* px (limit is near quote; TP/SL based on ATR)
    if side == "buy":
        tp_px = limit_px + TP_ATR_MULT * atr
        sl_px = limit_px - SL_ATR_MULT * atr
    else:
        tp_px = limit_px - TP_ATR_MULT * atr
        sl_px = limit_px + SL_ATR_MULT * atr

    take_profit = TakeProfitRequest(limit_price=round(tp_px, 2))
    stop_loss = StopLossRequest(stop_price=round(sl_px, 2))

    req = LimitOrderRequest(
        symbol=sym,
        qty=None,  # we set notional/qty below (we’ll call place_order with qty)
        side=order_side,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_px, 2),
        order_class=OrderClass.BRACKET,
        take_profit=take_profit,
        stop_loss=stop_loss,
        extended_hours=False,  # <—— RTH only
    )
    return cli, req

# =========================
# Main
# =========================

def _parse_flag(name: str, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
        if a.startswith(name + "="):
            return type(default)(a.split("=", 1)[1])
    return default

def main():
    since_min = _parse_flag("--since-min", EXEC_SINCE_MIN)
    min_strength = _parse_flag("--min-strength", EXEC_MIN_STRENGTH)

    bp = _get_buying_power()
    log.info(
        "executor_bracket | since-min=%s min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
        since_min, min_strength, bp, FRACTIONAL, LONG_ONLY,
    )

    # ===== RTH and BP guards =====
    if not ALLOW_AFTER_HOURS and not _market_open():
        log.info("market is closed and ALLOW_AFTER_HOURS=0 -> skip this pass")
        return

    if bp < MIN_ACCOUNT_BP_USD:
        log.info("buying_power %.2f < MIN_ACCOUNT_BP_USD %.2f -> skip", bp, MIN_ACCOUNT_BP_USD)
        return

    # ===== Get signals =====
    df = _fetch_signals(since_min, min_strength)
    if df.empty:
        log.info("no qualifying signals in last %s min (>= %.2f)", since_min, min_strength)
        return
    sigs = _dedupe_latest_by_symbol(df)

    # Iterate signals
    for _, row in sigs.iterrows():
        sym = str(row["symbol"]).upper()
        side = str(row["side"]).lower()
        strength = float(row["strength"])
        px_hint = float(row["px"]) if row["px"] is not None else None

        # LONG_ONLY gate
        if LONG_ONLY and side == "sell":
            log.info("%s: LONG_ONLY=1 -> skip opening short", sym)
            continue
        if side not in ("buy", "sell"):
            continue

        # === Quotes + spread guard ===
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

            # Quote-aware limit target
            slip = QUOTE_PRICE_SLIPPAGE
            if side == "buy":
                limit_px = min(ask + slip, max(ask, (px_hint if px_hint else ask)))
            else:
                limit_px = max(bid - slip, min(bid, (px_hint if px_hint else bid)))
        else:
            # Fallback to prior px (less ideal)
            ref_px = px_hint if px_hint else 0.0
            limit_px = round(ref_px, 2)

        # === ATR / sizing ===
        atr = _get_atr(sym)
        qty = _qty_from_risk(limit_px, atr, bp)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        if not FRACTIONAL:
            qty_disp = int(qty)
        else:
            qty_disp = round(qty, 4)  # Alpaca fractional precision

        # === Build and submit ===
        try:
            cli, req = _submit_bracket(sym, side, float(limit_px), float(atr))
            # Fill qty on the request object (SDK expects number type for fractional too)
            req.qty = qty_disp

            o = cli.submit_order(req)
            log.info(
                "submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                sym, side, qty_disp, float(limit_px),
                float(req.take_profit.limit_price),
                float(req.stop_loss.stop_price),
                getattr(o, "id", "n/a"),
            )
            # Reduce local buying power approximation
            bp -= float(limit_px) * float(qty)
        except Exception as e:
            # Bubble up common Alpaca errors in logs
            try:
                msg = str(getattr(e, "response", {}).get("text", "")) or str(e)
            except Exception:
                msg = str(e)
            log.warning("submit failed %s %s: %s", sym, side, msg)
            continue

if __name__ == "__main__":
    main()
