import yfinance as yf
import pandas as pd
import streamlit as st
from sqlalchemy import text
from lib.db import get_engine

@st.cache_data(ttl=600)
def load_candles(ticker: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch last `limit` candles from yfinance and return standardized columns."""
    # map timeframe -> yfinance interval
    interval = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "60m", "4h": "240m",
        "1d": "1d", "1wk": "1wk", "1mo": "1mo",
    }.get(timeframe, "1d")

    # pick a period big enough to guarantee `limit` bars
    if interval.endswith("m"):
        # intraday needs days; be generous
        period = f"{max(1, limit // 300 + 1)}d"
    elif interval in ("1d", "1wk", "1mo"):
        period = f"{max(30, int(limit * 2))}d"
    else:
        period = "60d"

    df = yf.download(
        ticker, interval=interval, period=period,
        auto_adjust=False, progress=False, prepost=False
    )
    if df.empty:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])

    df = df.tail(limit).reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        ts_col: "ts", "Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"
    })
    return df[["ts", "o", "h", "l", "c", "v"]]

@st.cache_data(ttl=300)
def load_signals(ticker: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch signals for overlay from Postgres."""
    q = text("""
        SELECT s.ts,
               (s.signal->>'side')          AS side,
               ((s.signal->>'strength')::float) AS strength
        FROM signals s
        JOIN symbols sym ON sym.id = s.symbol_id
        WHERE sym.ticker = :t
          AND s.timeframe = :tf
        ORDER BY s.ts DESC
        LIMIT :lim
    """)
    with get_engine().connect() as c:
        return pd.read_sql(q, c, params={"t": ticker, "tf": timeframe, "lim": limit})
