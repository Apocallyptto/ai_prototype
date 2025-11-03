import os
import sys
import logging
import math
from datetime import datetime, timezone

import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

from tools.quotes import get_bid_ask_mid  # quote-aware limits / spread filter

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("executor_bracket")

# -----------------------
# ENV / Config
# -----------------------
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/trader")
SYMBOLS_CSV = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
DEFAULT_SINCE_MIN = int(os.getenv("EXEC_SINCE_MIN", "20"))
DEFAULT_MIN_STRENGTH = float(os.getenv("EXEC_MIN_STRENGTH", "0.50"))

# Risk / ATR
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))

# Quotes / spreads
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.06"))
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

# Market hours gate
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))
ACCOUNT_FALLBACK_TO_CASH = os.getenv("ACCOUNT_FALLBACK_TO_CASH", "1") == "1"

# Position policy
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"
MIN_QTY = float(os.getenv("MIN_QTY", "0.01"))

# Alpaca creds are read from env by TradingClient
def _trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )

# -----------------------
# Helpers
# -----------------------
def _get_buying_power() -> float:
    """Read buying_power; optionally fall back to cash if BP is 0 on paper accounts."""
    try:
        cli = _trading_client()
        acct = cli.get_account()
        bp = float(getattr(acct, "buying_power", 0.0) or 0.0)
        if bp <= 0 and ACCOUNT_FALLBACK_TO_CASH:
            cash = float(getattr(acct, "cash", 0.0) or 0.0)
            log.warning("buying_power reported 0; falling back to cash=%.2f", cash)
            return cash
        return bp
    except Exception as e:
        log.warning("cannot read account/buying_power: %s", e)
        return 0.0

def _market_open() -> bool:
    try:
        cli = _trading_client()
        clk = cli.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False  # conservative

def _calc_atr_from_df(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Compute ATR from OHLC df that includes 'high','low','close'."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    val = float(atr.iloc[-1])
    return max(1e-4, val)

def _fetch_bars(symbol: str, interval: str = "5m", period: str = "30d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"no bars for {symbol}")
    df = df.rename(columns=str.lower)
    # yfinance sometimes returns single-level index; ensure columns present
    req = {"open","high","low","close"}
    if not req.issubset(set(df.columns)):
        raise RuntimeError(f"bars missing columns for {symbol}: have {df.columns}")
    return df

def _latest_px(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])

def _get_atr(symbol: str) -> tuple[float, float]:
    """Return (atr, last_close_px)."""
    df = _fetch_bars(symbol, interval="5m", period=f"{ATR_LOOKBACK_DAYS}d")
    atr = _calc_atr_from_df(df, ATR_PERIOD)
    px = _latest_px(df)
    return atr, px

def _qty_from_risk(price: float, atr: float) -> float:
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    if FRACTIONAL:
        # fractional sizing in notional terms
        # risk_per_share ~= sl_dist -> target notional = RISK / (sl_dist/price) = RISK * price / sl_dist
        target_notional = max(0.0, RISK_PER_TRADE_USD * price / max(sl_dist, 1e-6))
        bp = _get_buying_power()
        max_notional = max(0.0, min(bp, target_notional))
        if max_notional <= 0:
            return 0.0
        qty = max_notional / max(price, 0.01)
        return max(0.0, qty)
    else:
        # whole-share
        qty_risk = max(1, int(RISK_PER_TRADE_USD / max(sl_dist, 1e-6)))
        bp = _get_buying_power()
        max_bp_shares = int(max(0, bp) // max(price, 0.01))
        if max_bp_shares <= 0:
            return 0
        qty = min(qty_risk, max_bp_shares)
        if qty < qty_risk:
            log.info("qty capped by BP: %s -> %s (bp=%.2f, px=%.2f)", qty_risk, qty, bp, price)
        return max(0, qty)

def _submit_bracket(symbol: str, side: str, qty: float, limit_px: float, tp_px: float, sl_px: float):
    cli = _trading_client()

    # Alpaca side enum
    side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Fractional constraints
    notional = None
    qty_val = None
    if FRACTIONAL:
        # Alpaca fractional short selling is not allowed unless explicitly enabled by your plan.
        if side == "sell" and not ALLOW_FRACTIONAL_SHORTS:
            raise RuntimeError("fractional orders cannot be sold short")
        # Use notional when fractional
        notional = round(float(qty) * float(limit_px), 2)
        if notional <= 0:
            raise RuntimeError("invalid notional computed")
    else:
        if qty <= 0:
            raise RuntimeError("At least one of qty or notional must be provided")
        qty_val = int(qty)

    # Build bracket limit order with explicit RTH only
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty_val,
        notional=notional,
        side=side_enum,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        limit_price=round(float(limit_px), 2),
        take_profit=TakeProfitRequest(limit_price=round(float(tp_px), 2)),
        stop_loss=StopLossRequest(stop_price=round(float(sl_px), 2)),
        extended_hours=False,
    )
    o = cli.submit_order(req)
    return o

def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    """Fetch recent qualifying signals; no s.atr dependency."""
    sql = """
        SELECT s.created_at, s.symbol, s.side, COALESCE(s.scaled_strength, s.strength) AS strength,
               s.px
        FROM signals s
        WHERE s.created_at >= NOW() - INTERVAL %s
          AND COALESCE(s.scaled_strength, s.strength) >= %s
        ORDER BY s.created_at DESC
    """
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    return df

def _dedupe_latest_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # keep the latest created_at per symbol
    df = df.sort_values(["symbol", "created_at"], ascending=[True, False])
    df = df.groupby("symbol", as_index=False).head(1)
    # keep only symbols we trade
    allowed = set([s.strip().upper() for s in SYMBOLS_CSV.split(",") if s.strip()])
    df = df[df["symbol"].str.upper().isin(allowed)]
    return df.sort_values("symbol")

def _parse_args(argv) -> tuple[int, float]:
    # flags like: --since-min 90 --min-strength 0.55
    def _read_flag(name: str, default_val: str) -> str:
        for i, tok in enumerate(argv):
            if tok == name and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(name):
                parts = tok.split("=", 1)
                if len(parts) == 2:
                    return parts[1]
        return default_val

    since_min = int(_read_flag("--since-min", str(DEFAULT_SINCE_MIN)))
    min_strength = float(_read_flag("--min-strength", str(DEFAULT_MIN_STRENGTH)))
    return since_min, min_strength

# -----------------------
# Main
# -----------------------
def main():
    since_min, min_strength = _parse_args(sys.argv[1:])

    bp = _get_buying_power()
    log.info(
        "executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
        since_min, min_strength, bp, str(FRACTIONAL), str(LONG_ONLY)
    )

    # RTH & min BP gates
    if not ALLOW_AFTER_HOURS and not _market_open():
        log.info("market is closed and ALLOW_AFTER_HOURS=0 -> skip this pass")
        return
    if bp < MIN_ACCOUNT_BP_USD:
        log.info("buying_power %.2f < MIN_ACCOUNT_BP_USD %.2f -> skip", bp, MIN_ACCOUNT_BP_USD)
        return

    df = _dedupe_latest_by_symbol(_fetch_signals(since_min, min_strength))
    if df.empty:
        log.info("no qualifying signals in last %d min (>= %.2f)", since_min, min_strength)
        return

    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        side = str(row["side"]).lower().strip()  # 'buy' or 'sell'
        strength = float(row["strength"])
        px_from_signal = float(row["px"]) if not pd.isna(row.get("px", np.nan)) else None

        # LONG_ONLY
        if LONG_ONLY and side == "sell":
            log.info("%s: LONG_ONLY=1 -> skip opening short", sym)
            continue

        # --- ATR + reference price ---
        try:
            atr, px = _get_atr(sym)
        except Exception as e:
            log.warning("%s: ATR fetch failed: %s -> skip", sym, e)
            continue

        # prefer DB px if present and sensible
        if px_from_signal and px_from_signal > 0:
            px = float(px_from_signal)

        # --- quotes / spread guard + quote-aware limit ---
        quote = get_bid_ask_mid(sym)
        if quote:
            bid, ask, mid = quote
            spread_abs = max(0.0, ask - bid)
            spread_pct = (spread_abs / mid) * 100.0 if mid > 0 else 999.0

            if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
                log.info("%s: skip due to wide spread (bid=%.4f ask=%.4f mid=%.4f abs=%.4f pct=%.3f%% > limits)",
                         sym, bid, ask, mid, spread_abs, spread_pct)
                continue

            if side == "buy":
                limit_px = min(ask + QUOTE_PRICE_SLIPPAGE, max(ask, px))
            else:
                limit_px = max(bid - QUOTE_PRICE_SLIPPAGE, min(bid, px if px else bid))
        else:
            limit_px = round(px, 2)

        # --- bracket targets from ATR ---
        if side == "buy":
            tp_px = px + TP_ATR_MULT * atr
            sl_px = px - SL_ATR_MULT * atr
        else:
            tp_px = px - TP_ATR_MULT * atr
            sl_px = px + SL_ATR_MULT * atr

        # --- sizing ---
        qty = _qty_from_risk(px, atr)
        if FRACTIONAL:
            if qty < MIN_QTY:
                log.info("%s: fractional qty %.4f < MIN_QTY %.4f; skip", sym, qty, MIN_QTY)
                continue
        else:
            if qty <= 0:
                log.info("%s: no buying power for qty; skip", sym)
                continue

        # --- submit ---
        try:
            o = _submit_bracket(sym, side, qty, limit_px, tp_px, sl_px)
            log.info("submitted %s %s %s px=%.2f tp=%.2f sl=%.2f id=%s",
                     sym, side, (f"qty={qty}" if not FRACTIONAL else f"notional~{qty*limit_px:.2f}"),
                     float(limit_px), float(tp_px), float(sl_px), getattr(o, "id", "n/a"))
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)

if __name__ == "__main__":
    main()
