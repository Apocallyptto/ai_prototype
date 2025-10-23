# lib/atr_utils.py
from __future__ import annotations
import os
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Try yfinance first (easy), then Alpaca as fallback if available
def _fetch_yf(symbol: str, days: int, interval: str = "5m") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        period = f"{days}d" if days <= 59 else "60d"  # yfinance cap for 5m
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        df = df.rename(
            columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        )[["open","high","low","close","volume"]].astype("float64")
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception:
        return None

def _fetch_alpaca(symbol: str, days: int, timeframe: str = "5Min") -> Optional[pd.DataFrame]:
    try:
        from lib.broker_alpaca import get_bars
    except Exception:
        return None
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        rows = get_bars(symbol, timeframe, start, end, limit=20000)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")[["open","high","low","close","volume"]].astype("float64")
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception:
        return None

def fetch_bars(symbol: str, lookback_days: int = 30, interval_5m: bool = True) -> pd.DataFrame:
    df = _fetch_yf(symbol, lookback_days, "5m" if interval_5m else "15m")
    if df is None or df.empty:
        df = _fetch_alpaca(symbol, lookback_days, "5Min" if interval_5m else "15Min")
    if df is None or df.empty:
        raise RuntimeError(f"ATR fetch: no data for {symbol}")
    return df

def true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    a = (high - low).abs()
    b = (high - close_prev).abs()
    c = (low  - close_prev).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR (RMA). Requires at least period+1 rows for a stable seed.
    """
    h, l, c = df["high"], df["low"], df["close"]
    c_prev = c.shift(1)
    tr = true_range(h, l, c_prev)
    # First ATR seed = SMA of first 'period' TRs
    atr = tr.rolling(window=period, min_periods=period).mean()
    # Wilder smoothing from next point onwards
    for i in range(period+1, len(tr)):
        atr.iat[i] = (atr.iat[i-1] * (period - 1) + tr.iat[i]) / period
    return atr

def last_atr(symbol: str,
             period: int = 14,
             lookback_days: int = 30) -> float:
    """
    Get last ATR value (Wilder) from recent 5m bars. Raises if unavailable.
    """
    df = fetch_bars(symbol, lookback_days, interval_5m=True)
    if len(df) < period + 5:
        raise RuntimeError(f"ATR: insufficient bars for {symbol} (got {len(df)})")
    atr = atr_wilder(df, period).dropna()
    if atr.empty:
        raise RuntimeError(f"ATR: nan for {symbol}")
    return float(atr.iloc[-1])
