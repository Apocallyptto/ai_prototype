import streamlit as st, pandas as pd, plotly.graph_objects as go
from sqlalchemy import text
from lib.db import get_engine

st.title("📊 Chart + Signals")

sym = st.text_input("Ticker", "AAPL")
tf = st.selectbox("Timeframe", ["1d"])
limit = st.slider("Bars", 50, 500, 200)

@st.cache_data(ttl=120)
def load_candles(ticker, timeframe, limit):
    q = text("""
      SELECT c.ts, c.o, c.h, c.l, c.c
      FROM candles c
      JOIN symbols s ON s.id=c.symbol_id
      WHERE s.ticker=:t AND c.timeframe=:tf
      ORDER BY c.ts DESC LIMIT :lim
    """)
    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"t": ticker, "tf": timeframe, "lim": limit})
    return df.sort_values("ts")

@st.cache_data(ttl=120)
def load_signals(ticker, limit):
    q = text("""
      SELECT s.ts, s.signal->>'side' AS side
      FROM signals s JOIN symbols sym ON sym.id=s.symbol_id
      WHERE sym.ticker=:t
      ORDER BY s.ts DESC LIMIT :lim
    """)
    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"t": ticker, "lim": limit})
    return df

cdl = load_candles(sym, tf, limit)
sigs = load_signals(sym, 500)

if cdl.empty:
    st.info("No candles yet. Load candles via ETL.")
else:
    fig = go.Figure(data=[go.Candlestick(x=cdl["ts"], open=cdl["o"], high=cdl["h"], low=cdl["l"], close=cdl["c"])])
    if not sigs.empty:
        buys = sigs[sigs.side=="buy"]
        sells = sigs[sigs.side=="sell"]
        fig.add_scatter(x=buys["ts"], y=cdl.set_index("ts").reindex(buys["ts"])["c"], mode="markers", name="buy", marker_symbol="triangle-up")
        fig.add_scatter(x=sells["ts"], y=cdl.set_index("ts").reindex(sells["ts"])["c"], mode="markers", name="sell", marker_symbol="triangle-down")
    st.plotly_chart(fig, use_container_width=True)
