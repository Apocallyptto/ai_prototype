"""
Robust ATR helper with dual source:
1) Alpaca market data (preferred if API keys present)
2) yfinance fallback

Returns (atr, last_close).
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import os
import pandas as pd

# Alpaca (v2 data client)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# yfinance fallback
import yfinance as yf


def _to_ohlc_df(df_like: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize any bar DataFrame into columns: Open, High, Low, Close.
    Handles lower-case, multi-index, and Alpaca formats.
    """
    if df_like is None or len(df_like) == 0:
        raise ValueError(f"no bars for {symbol}")

    df = df_like.copy()

    # Alpaca's .df may come with columns: open high low close volume; ensure title-case
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl == "close":
            rename[c] = "Close"
    if rename:
        df = df.rename(columns=rename)

    # yfinance can return multiindex ('Open','AAPL'), etc.
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer a slice for our symbol (case-insensitive)
        lower = { (a.lower(), b.lower() if isinstance(b, str) else b): (a,b) for (a,b) in df.columns }
        def grab(key: str):
            tup = lower.get((key, symbol.lower()))
            if tup is None:
                # fall back: any column with first level == key
                matches = [col for col in df.columns if isinstance(col, tuple) and str(col[0]).lower() == key]
                if not matches:
                    raise KeyError(key)
                return df[matches[0]]
            return df[tup]

        out = pd.DataFrame({
            "Open":  grab("open"),
            "High":  grab("high"),
            "Low":   grab("low"),
            "Close": grab("close"),
        })
        out.index = df.index
        return out

    # Ensure required columns exist
    for need in ("Open", "High", "Low", "Close"):
        if need not in df.columns:
            # some yfinance builds use title-case already but may miss one — try lowercase
            lc = need.lower()
            if lc in df.columns:
                df[need] = df[lc]
            else:
                raise ValueError(f"bars missing '{need}' for {symbol}")

    return df[["Open", "High", "Low", "Close"]]


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    h_l = (df["High"] - df["Low"]).abs()
    h_pc = (df["High"] - prev_close).abs()
    l_pc = (df["Low"] - prev_close).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)


def _atr_from_df(df: pd.DataFrame, period: int) -> Tuple[float, float]:
    tr = _true_range(df)
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    last_close = float(df["Close"].iloc[-1])
    if not pd.notna(atr):
        raise ValueError("ATR nan")
    return float(atr), last_close


def _fetch_alpaca(symbol: str, days: int) -> Optional[pd.DataFrame]:
    key = os.getenv("ALPACA_API_KEY") or ""
    sec = os.getenv("ALPACA_API_SECRET") or ""
    if not key or not sec:
        return None
    try:
        cli = StockHistoricalDataClient(key, sec)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=max(60, days + 10),
            adjustment="raw",
        )
        bars = cli.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            # select this symbol only
            df = df.xs(symbol, level="symbol")
        return _to_ohlc_df(df, symbol)
    except Exception:
        return None


def _fetch_yf(symbol: str, days: int) -> Optional[pd.DataFrame]:
    try:
        # Pull a bit more than needed to survive throttling gaps
        hist = yf.Ticker(symbol).history(period=f"{max(days, 60)}d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return _to_ohlc_df(hist, symbol)
    except Exception:
        return None


def get_atr(symbol: str, lookback_days: int = 30, period: int = 14) -> Tuple[float, float]:
    """
    Compute ATR(period) over daily bars. Tries Alpaca first, then yfinance.
    Returns (atr, last_close).
    """
    # fetch (Alpaca -> yfinance)
    df = _fetch_alpaca(symbol, lookback_days)
    if df is None or df.empty:
        df = _fetch_yf(symbol, lookback_days)
    if df is None or df.empty:
        raise ValueError(f"no history available for {symbol} (Alpaca+YF)")

    # Keep only the last `lookback_days` rows to match caller expectation
    df = df.tail(max(lookback_days, period + 2))
    return _atr_from_df(df, period)
