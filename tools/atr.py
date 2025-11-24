# tools/atr.py
"""
ATR helper utilities.

compute_atr(symbol: str, period: int = 14, lookback_days: int = 30)
  -> (atr_value, last_close_price)

Používa Yahoo Finance (yfinance) na stiahnutie OHLC dát.
"""

import datetime as dt
import logging
from typing import Tuple, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _download_ohlc(symbol: str,
                   lookback_days: int = 30,
                   interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Stiahne OHLC dáta z Yahoo pre daný symbol.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days + 5)  # +5 buffer

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        logger.error(f"Failed to download data for {symbol}: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"No data returned for {symbol}")
        return None

    # normalize columns
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    )

    return df


def compute_atr(
    symbol: str,
    period: int = 14,
    lookback_days: int = 30,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Average True Range (ATR) a poslednú close cenu.

    Returns:
        (atr_value, last_close_price) – oboje môžu byť None, ak nie sú dáta.
    """
    df = _download_ohlc(symbol, lookback_days=lookback_days, interval="1d")
    if df is None or len(df) < period + 1:
        logger.warning(
            f"Not enough data to compute ATR for {symbol} "
            f"(len={0 if df is None else len(df)})"
        )
        return None, None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = SMA of TR
    atr = tr.rolling(window=period).mean()

    atr_val = float(atr.iloc[-1])
    last_close = float(close.iloc[-1])

    return atr_val, last_close


if __name__ == "__main__":
    # simple manual test
    logging.basicConfig(level=logging.INFO)
    for sym in ["AAPL", "MSFT", "SPY"]:
        atr_val, last = compute_atr(sym)
        print(f"{sym}: close={last}, ATR={atr_val}")
