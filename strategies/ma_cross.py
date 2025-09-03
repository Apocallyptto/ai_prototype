# strategies/ma_cross.py
from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf

def ma_cross_signal(ticker: str, period_days=200, fast=20, slow=50) -> dict | None:
    """Return {'side': 'buy'|'sell', 'strength': float, 'meta': {...}} or None."""
    # pull daily candles
    df = yf.download(ticker, period=f"{period_days}d", interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < slow + 2:
        return None

    df["fast"] = df["Close"].rolling(fast).mean()
    df["slow"] = df["Close"].rolling(slow).mean()
    df.dropna(inplace=True)

    # last two rows for cross detection
    a, b = df.iloc[-2], df.iloc[-1]
    cross_up   = a["fast"] <= a["slow"] and b["fast"] >  b["slow"]
    cross_down = a["fast"] >= a["slow"] and b["fast"] <  b["slow"]

    if cross_up:
        strength = float((b["fast"] - b["slow"]) / b["Close"])
        return {"side": "buy", "strength": strength, "meta": {
            "ts": datetime.now(timezone.utc).isoformat(),
            "fast": fast, "slow": slow, "price": float(b["Close"])
        }}
    if cross_down:
        strength = float((b["slow"] - b["fast"]) / b["Close"])
        return {"side": "sell", "strength": strength, "meta": {
            "ts": datetime.now(timezone.utc).isoformat(),
            "fast": fast, "slow": slow, "price": float(b["Close"])
        }}
    return None
