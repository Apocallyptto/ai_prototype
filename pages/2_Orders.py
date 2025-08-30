import streamlit as st, pandas as pd
from lib.db import get_engine
st.title("📑 Orders & Fills")

@st.cache_data(ttl=120)
def load_orders(limit=200):
    q = """
    SELECT o.ts, sym.ticker, o.side, o.qty, o.type, o.limit_price, o.status
    FROM orders o JOIN symbols sym ON sym.id = o.symbol_id
    ORDER BY o.ts DESC LIMIT :lim;
    """
    return pd.read_sql(q, get_engine(), params={"lim": limit})

st.dataframe(load_orders(), use_container_width=True)
