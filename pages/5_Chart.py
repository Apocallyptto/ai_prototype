# pages/5_Chart.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sqlalchemy import text
from lib.db import get_engine

st.title("📊 Chart + Signals")

# --- Controls (always render) ---
with st.sidebar:
    sym = st.text_input("Ticker", "AAPL").strip().upper()
    tf  = st.selectbox("Timeframe", ["1d", "1h", "4h", "15m", "5m", "1m"], index=0)
    limit = st.slider("Bars", min_value=50, max_value=1000, value=200, step=10)

@st.cache_data(ttl=600, show_spinner=False)
def load_candles_yf(ticker: str, timeframe: str, limit: int) -> pd.DataFrame:
    interval = {
        "1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "240m", "1d": "1d"
    }.get(timeframe, "1d")

    # choose a long enough period for the # of bars requested
    if interval in {"1m", "5m", "15m", "60m", "240m"}:
        # intraday bars: use days. yfinance caps per interval; be generous
        period = "5d" if limit <= 1500 else "60d"
    else:
        # daily bars
        period = "2y" if limit <= 500 else "5y"

    df = yf.download(
        ticker, interval=interval, period=period,
        auto_adjust=False, prepost=False, progress=False
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.tail(limit).reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        ts_col: "ts", "Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"
    })
    return df[["ts", "o", "h", "l", "c", "v"]]

@st.cache_data(ttl=300, show_spinner=False)
def load_signals_db(ticker: str, timeframe: str, lim: int = 5000) -> pd.DataFrame:
    q = text("""
        SELECT s.ts,
               s.signal->>'side'       AS side,
               (s.signal->>'strength')::float AS strength
        FROM signals s
        JOIN symbols sym ON sym.id = s.symbol_id
        WHERE sym.ticker = :t AND s.timeframe = :tf
        ORDER BY s.ts
        LIMIT :lim
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"t": ticker, "tf": timeframe, "lim": lim})

# --- Load data ---
with st.spinner("Loading data…"):
    candles = load_candles_yf(sym, tf, limit)
    try:
        sig = load_signals_db(sym, tf)
    except Exception:
        sig = pd.DataFrame()  # keep chart working even if DB is unavailable

if candles.empty:
    st.warning("No candles returned. Try AAPL + 1d, or another timeframe/ticker.")
else:
    # --- Build chart ---
    fig = go.Figure()
    fig.add_candlestick(
        x=candles.ts, open=candles.o, high=candles.h, low=candles.l, close=candles.c,
        name=sym
    )

    # Overlay signals if available
    if not sig.empty:
        px = candles.set_index("ts")["c"]
        buys  = sig[sig.side == "buy"]
        sells = sig[sig.side == "sell"]

        if not buys.empty:
            fig.add_scatter(
                x=buys.ts,
                y=px.reindex(buys.ts).values,
                mode="markers",
                name="buy",
                marker_symbol="triangle-up",
                marker_size=10
            )
        if not sells.empty:
            fig.add_scatter(
                x=sells.ts,
                y=px.reindex(sells.ts).values,
                mode="markers",
                name="sell",
                marker_symbol="triangle-down",
                marker_size=10
            )

    fig.update_layout(height=520, xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Candles (last rows)"):
        st.dataframe(candles.tail(20), use_container_width=True)
