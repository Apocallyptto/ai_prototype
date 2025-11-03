import os
import sys
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

from tools.quotes import get_bid_ask_mid

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("executor_bracket")

# ============================================================
# ENV / CONFIG
# ============================================================
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
ENGINE = create_engine(DB_URL, pool_pre_ping=True)

SYMBOLS_CSV = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
DEFAULT_SINCE_MIN = int(os.getenv("EXEC_SINCE_MIN", "20"))
DEFAULT_MIN_STRENGTH = float(os.getenv("EXEC_MIN_STRENGTH", "0.50"))

# --- risk / ATR ---
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))

# --- quote / spreads ---
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.06"))
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

# --- market hours & account ---
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))
ACCOUNT_FALLBACK_TO_CASH = os.getenv("ACCOUNT_FALLBACK_TO_CASH", "1") == "1"

# --- position policy ---
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"
MIN_QTY = float(os.getenv("MIN_QTY", "0.01"))
TICK_SIZE = float(os.getenv("TICK_SIZE", "0.01"))


# ============================================================
# ALPACA CLIENT
# ============================================================
def _trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )


# ============================================================
# HELPERS
# ============================================================
def _get_buying_power() -> float:
    try:
        cli = _trading_client()
        acct = cli.get_account()
        bp = float(acct.buying_power or 0)
        if bp <= 0 and ACCOUNT_FALLBACK_TO_CASH:
            cash = float(acct.cash or 0)
            log.warning("buying_power reported 0; falling back to cash=%.2f", cash)
            return cash
        return bp
    except Exception as e:
        log.warning("cannot read account: %s", e)
        return 0.0


def _shorting_enabled() -> bool:
    try:
        acct = _trading_client().get_account()
        return bool(getattr(acct, "shorting_enabled", False))
    except Exception:
        return False


def _market_open() -> bool:
    try:
        clk = _trading_client().get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False


def _flatten_yf_multiindex(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    sym = symbol.lower()
    try:
        if sym in [s.lower() for s in df.columns.get_level_values(-1)]:
            return df.xs(sym, level=-1, axis=1)
    except Exception:
        pass
    try:
        return df.droplevel(1, axis=1)
    except Exception:
        return df


def _fetch_bars(symbol: str, interval="5m", period="30d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"no bars for {symbol}")
    df = _flatten_yf_multiindex(df, symbol).rename(columns=str.lower)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise RuntimeError(f"bars missing {col} for {symbol}")
    return df


def _calc_atr_from_df(df: pd.DataFrame, period=ATR_PERIOD) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return float(max(atr.iloc[-1], 1e-4))


def _get_atr(symbol: str) -> tuple[float, float]:
    df = _fetch_bars(symbol, "5m", f"{ATR_LOOKBACK_DAYS}d")
    atr = _calc_atr_from_df(df)
    return atr, float(df["close"].iloc[-1])


def _qty_from_risk(price: float, atr: float) -> float:
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    bp = _get_buying_power()
    if FRACTIONAL:
        notional = min(bp, RISK_PER_TRADE_USD * price / max(sl_dist, 1e-6))
        return 0 if notional <= 0 else notional / price
    qty = max(1, int(RISK_PER_TRADE_USD / max(sl_dist, 1e-6)))
    return max(0, min(qty, int(bp // price)))


def _submit_bracket(symbol, side, qty, limit_px, tp_px, sl_px):
    cli = _trading_client()
    side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
    if FRACTIONAL:
        notional = round(qty * limit_px, 2)
        req = LimitOrderRequest(
            symbol=symbol,
            notional=notional,
            side=side_enum,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(limit_px, 2),
            take_profit=TakeProfitRequest(limit_price=round(tp_px, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl_px, 2)),
            extended_hours=False,
        )
    else:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=side_enum,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(limit_px, 2),
            take_profit=TakeProfitRequest(limit_price=round(tp_px, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl_px, 2)),
            extended_hours=False,
        )
    return cli.submit_order(req)


def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    sql = """
        SELECT s.created_at,
               s.symbol,
               s.side,
               COALESCE(s.scaled_strength, s.strength) AS strength,
               s.px
        FROM signals s
        WHERE s.created_at >= NOW() - INTERVAL %s
          AND COALESCE(s.scaled_strength, s.strength) >= %s
        ORDER BY s.created_at DESC
    """
    return pd.read_sql(sql, ENGINE, params=(f"{since_min} minutes", min_strength))


def _dedupe_latest_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["symbol", "created_at"], ascending=[True, False])
    df = df.groupby("symbol", as_index=False).head(1)
    allowed = {s.strip().upper() for s in SYMBOLS_CSV.split(",") if s.strip()}
    return df[df["symbol"].str.upper().isin(allowed)].sort_values("symbol")


def _parse_args(argv):
    def _read_flag(name, default):
        for i, t in enumerate(argv):
            if t == name and i + 1 < len(argv):
                return argv[i + 1]
            if t.startswith(name + "="):
                return t.split("=", 1)[1]
        return default
    return int(_read_flag("--since-min", str(DEFAULT_SINCE_MIN))), float(
        _read_flag("--min-strength", str(DEFAULT_MIN_STRENGTH))
    )


# ============================================================
# MAIN
# ============================================================
def main():
    since_min, min_strength = _parse_args(sys.argv[1:])
    bp = _get_buying_power()
    log.info(
        "executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
        since_min, min_strength, bp, FRACTIONAL, LONG_ONLY,
    )

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

    short_ok = _shorting_enabled()

    for _, row in df.iterrows():
        sym = str(row.symbol).upper()
        side = str(row.side).lower()
        px_signal = float(row.px or 0)

        if side == "sell":
            if LONG_ONLY:
                log.info("%s: LONG_ONLY=1 -> skip short", sym)
                continue
            if not short_ok:
                log.info("%s: account not margin/short-enabled -> skip short", sym)
                continue

        try:
            atr, px = _get_atr(sym)
        except Exception as e:
            log.warning("%s: ATR fetch failed: %s -> skip", sym, e)
            continue
        if px_signal > 0:
            px = px_signal

        quote = get_bid_ask_mid(sym)
        if quote:
            bid, ask, mid = quote
            spread_abs = max(0.0, ask - bid)
            spread_pct = (spread_abs / mid) * 100 if mid > 0 else 999
            if spread_abs > MAX_SPREAD_ABS or spread_pct > MAX_SPREAD_PCT:
                log.info("%s: skip wide spread abs=%.4f pct=%.3f%%", sym, spread_abs, spread_pct)
                continue
            limit_px = min(ask + QUOTE_PRICE_SLIPPAGE, max(ask, px)) if side == "buy" else max(
                bid - QUOTE_PRICE_SLIPPAGE, min(bid, px or bid)
            )
        else:
            limit_px = px

        entry = float(limit_px)
        tick = max(TICK_SIZE, 0.01)
        if side == "buy":
            tp_px = max(entry + TP_ATR_MULT * atr, entry + tick)
            sl_px = min(entry - SL_ATR_MULT * atr, entry - tick)
        else:
            tp_px = min(entry - TP_ATR_MULT * atr, entry - tick)
            sl_px = max(entry + SL_ATR_MULT * atr, entry + tick)

        qty = _qty_from_risk(entry, atr)
        if FRACTIONAL:
            if qty < MIN_QTY:
                log.info("%s: fractional qty %.4f < MIN_QTY %.4f", sym, qty, MIN_QTY)
                continue
        elif qty <= 0:
            log.info("%s: no BP for qty", sym)
            continue

        try:
            o = _submit_bracket(sym, side, qty, entry, tp_px, sl_px)
            sized = f"notional~{qty*entry:.2f}" if FRACTIONAL else f"qty={int(qty)}"
            log.info("%s %s %s px=%.2f tp=%.2f sl=%.2f id=%s", sym, side, sized, entry, tp_px, sl_px, getattr(o, "id", "n/a"))
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)


if __name__ == "__main__":
    main()
