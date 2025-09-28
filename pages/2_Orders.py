import pandas as pd
import sqlalchemy as sa
import streamlit as st
from lib.db import make_engine

@st.cache_data(ttl=30)
def load_orders(limit: int) -> pd.DataFrame:
    eng = make_engine()
    with eng.connect() as c:
        q = sa.text("""
            SELECT o.ts,
                   sym.ticker,
                   o.side,
                   o.qty,
                   o.order_type,
                   o.limit_price,
                   o.status
            FROM orders o
            JOIN symbols sym ON sym.id = o.symbol_id
            ORDER BY o.ts DESC
            LIMIT :lim
        """)
        return pd.read_sql(q, c, params={"lim": limit})

st.title("📄 Orders & Fills")
lim = st.slider("Max rows", 50, 1000, 200, 50)
st.dataframe(load_orders(lim), width="stretch")
