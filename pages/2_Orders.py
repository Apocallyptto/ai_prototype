import streamlit as st, pandas as pd
from sqlalchemy import text
from lib.db import get_engine

st.title("📑 Orders & Fills")

@st.cache_data(ttl=120)
def load_orders(limit=200):
    q = text("""
        SELECT o.ts,
               sym.ticker,
               o.side,
               o.qty,
               o."type" AS order_type,     -- quote to avoid keyword issues
               o.limit_price,
               o.status
        FROM orders o
        JOIN symbols sym ON sym.id = o.symbol_id
        ORDER BY o.ts DESC
        LIMIT :lim
    """)
    # use a connection so pandas doesn't switch to exec_driver_sql silently
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"lim": limit})

st.dataframe(load_orders(), use_container_width=True)
