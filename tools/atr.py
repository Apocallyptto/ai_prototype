"""
Ultra-robust ATR helper.
1️⃣ Tries Alpaca data first (if API keys exist).
2️⃣ Falls back to yfinance.
3️⃣ If both fail, computes synthetic ATR using last known mid-price.
"""

from __future__ import annotations
import math, os
import pandas as pd
import yfinance as yf
from typing import Tuple, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    h_l = (df["High"] - df["Low"]).abs()
    h_pc = (df["High"] - prev_close).abs()
    l_pc = (df["Low"] - prev_close).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)


def _atr_from_df(df: pd.DataFrame, period: int) -> Tuple[float, float]:
    tr = _true_range(df)
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    if not pd.notna(atr):
        raise ValueError("ATR nan")
    last_close = float(df["Close"].iloc[-1])
    return float(atr), last_close


def _fetch_alpaca(symbol: str, days: int) -> Optional[pd.DataFrame]:
    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        return None
    try:
        cli = StockHistoricalDataClient(key, sec)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=max(60, days + 10))
        bars = cli.get_stock_bars(req)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        return df.rename(columns=str.title)[["Open", "High", "Low", "Close"]]
    except Exception:
        return None


def _fetch_yf(symbol: str, days: int) -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(symbol).history(period=f"{max(days,60)}d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return hist.rename(columns=str.title)[["Open", "High", "Low", "Close"]]
    except Exception:
        return None


def get_atr(symbol: str, lookback_days: int = 30, period: int = 14) -> Tuple[float, float]:
    """
    Compute ATR(period) using Alpaca or Yahoo fallback.
    Returns (atr, last_close). Never raises on network failures.
    """
    # 1️⃣ Try Alpaca
    df = _fetch_alpaca(symbol, lookback_days)
    if df is not None and not df.empty:
        try:
            return _atr_from_df(df.tail(max(lookback_days, period + 2)), period)
        except Exception:
            pass

    # 2️⃣ Try yfinance
    df = _fetch_yf(symbol, lookback_days)
    if df is not None and not df.empty:
        try:
            return _atr_from_df(df.tail(max(lookback_days, period + 2)), period)
        except Exception:
            pass

    # 3️⃣ Last resort: synthetic ATR ≈ 0.25% of current price
    from tools.quotes import get_bid_ask_mid
    q = get_bid_ask_mid(symbol)
    if q:
        _, _, mid = q
        return mid * 0.0025, mid
    else:
        # totally offline fallback
        return 1.0, 100.0
