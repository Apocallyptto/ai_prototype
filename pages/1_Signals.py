import streamlit as st, pandas as pd
from sqlalchemy import text
from lib.db import get_engine

st.title("🔔 Signals")

with st.sidebar:
    sym_like = st.text_input("Ticker contains", "")
    lim = st.slider("Rows", 50, 1000, 200, step=50)

@st.cache_data(ttl=120)
def load_signals(sym_like, limit):
    q = text("""
      SELECT s.ts, sym.ticker, s.timeframe, s.model,
             s.signal->>'side' AS side,
             (s.signal->>'strength')::float AS strength
      FROM signals s
      JOIN symbols sym ON sym.id = s.symbol_id
      WHERE (:sym = '' OR sym.ticker ILIKE '%'||:sym||'%')
      ORDER BY s.ts DESC
      LIMIT :lim
    """)
    with get_engine().connect() as c:
        return pd.read_sql(q, c, params={"sym": sym_like, "lim": limit})

st.dataframe(load_signals(sym_like, lim), use_container_width=True)
