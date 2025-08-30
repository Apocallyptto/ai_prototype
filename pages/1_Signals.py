import streamlit as st, pandas as pd
from sqlalchemy import text
from lib.db import get_engine

st.title("🔔 Signals")

@st.cache_data(ttl=120)
def load_signals(limit=200):
    q = text("""
        SELECT s.ts, sym.ticker, s.timeframe, s.model,
               s.signal->>'side' AS side,
               (s.signal->>'strength')::float AS strength
        FROM signals s JOIN symbols sym ON sym.id = s.symbol_id
        ORDER BY s.ts DESC
        LIMIT :lim
    """)
    return pd.read_sql(q, get_engine(), params={"lim": limit})

st.dataframe(load_signals(), use_container_width=True)
