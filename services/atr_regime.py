# services/atr_regime.py
"""
ATR Regime module
-----------------
Detects volatility regimes (low / normal / high) and adapts
take-profit (TP) and stop-loss (SL) multipliers dynamically.

Use:
    from services.atr_regime import get_dynamic_tp_sl

Returns (tp_mult, sl_mult) given current ATR and historical ATR mean.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta


def _envint(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _get_atr_series(symbol: str, lookback_days: int, period: int) -> pd.Series:
    """
    Downloads daily bars from Yahoo and computes ATR series.
    """
    t = yf.Ticker(symbol)
    bars = t.history(period=f"{lookback_days}d", interval="1d")
    if bars is None or bars.empty:
        raise RuntimeError(f"no data for {symbol}")
    high = bars["High"]
    low = bars["Low"]
    close = bars["Close"].shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean().dropna()
    return atr


def get_dynamic_tp_sl(symbol: str) -> tuple[float, float]:
    """
    Compute adaptive (tp_mult, sl_mult) based on volatility regime.
    Default base: TP=1.5x ATR, SL=1.0x ATR.
    """
    base_tp = float(os.getenv("ATR_MULT_TP", "1.5"))
    base_sl = float(os.getenv("ATR_MULT_SL", "1.0"))
    lookback_days = _envint("ATR_LOOKBACK_DAYS", 30)
    period = _envint("ATR_PERIOD", 14)

    atr = _get_atr_series(symbol, lookback_days, period)
    if atr.empty:
        return base_tp, base_sl

    current_atr = atr.iloc[-1]
    avg_atr = atr.mean()
    ratio = float(current_atr / avg_atr) if avg_atr > 0 else 1.0

    # classify regime
    if ratio < 0.9:      # low volatility
        tp_mult = base_tp * 1.2    # target further
        sl_mult = base_sl * 0.9    # tighter stop
        regime = "low"
    elif ratio > 1.1:    # high volatility
        tp_mult = base_tp * 0.8    # take profits quicker
        sl_mult = base_sl * 1.1    # looser stop
        regime = "high"
    else:                # normal
        tp_mult = base_tp
        sl_mult = base_sl
        regime = "normal"

    print(f"[ATR_REGIME] {symbol}: regime={regime} current/avg={ratio:.2f} TP×{tp_mult:.2f} SL×{sl_mult:.2f}")
    return tp_mult, sl_mult
