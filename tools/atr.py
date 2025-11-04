"""
Lightweight ATR helper using yfinance so we don't depend on Alpaca bars here.
Returns (atr, last_close).
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import pandas as pd
import yfinance as yf


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    h_l = (df["High"] - df["Low"]).abs()
    h_pc = (df["High"] - prev_close).abs()
    l_pc = (df["Low"] - prev_close).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr


def get_atr(symbol: str, lookback_days: int = 30, period: int = 14) -> Tuple[float, float]:
    """
    Compute ATR(period) over the last `lookback_days` of daily bars.

    Returns
    -------
    (atr, last_close)
    """
    # Pull enough days so ATR(period) is meaningful.
    extra = max(0, period * 2)
    days = max(lookback_days, period + extra)

    hist = yf.Ticker(symbol).history(period=f"{days}d", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(f"no history from yfinance for {symbol}")

    # Standardize expected columns.
    for col in ("Open", "High", "Low", "Close"):
        if col not in hist.columns:
            # Some yfinance versions return lowercase or multiindex; normalize if needed.
            lc = col.lower()
            if lc in hist.columns:
                hist[col] = hist[lc]
            else:
                # Try to handle multiindex ('Price','Ticker') shape
                try:
                    mi_cols = [c for c in hist.columns if isinstance(c, tuple)]
                    if mi_cols:
                        # pick first level matching our label for this symbol
                        # e.g., ('high','aapl') -> 'High'
                        candidate = [c for c in mi_cols if c[0].lower() == lc]
                        if candidate:
                            hist[col] = hist[candidate[0]]
                        else:
                            raise KeyError
                    else:
                        raise KeyError
                except Exception:
                    raise ValueError(f"bars missing '{col}' for {symbol}")

    tr = _true_range(hist)
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    if not isinstance(atr, (int, float)) or (isinstance(atr, float) and (math.isnan(atr) or math.isinf(atr))):
        raise ValueError(f"ATR could not be computed for {symbol}")

    last_close = float(hist["Close"].iloc[-1])
    return float(atr), last_close
