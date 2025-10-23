# pages/4_PnL.py
from __future__ import annotations
import os
import psycopg2
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PnL Dashboard", layout="wide")

dsn = os.environ.get("DATABASE_URL")
if not dsn:
    st.error("❌ DATABASE_URL not set in environment.")
    st.stop()

@st.cache_data(ttl=60)
def load_pnl():
    with psycopg2.connect(dsn=dsn) as conn:
        df = pd.read_sql_query("""
            SELECT as_of_date,
                   equity::float8 AS equity,
                   COALESCE(profit, 0)::float8 AS profit,
                   COALESCE(profit_pct, 0)::float8 AS profit_pct,
                   created_at
            FROM public.daily_pnl
            ORDER BY as_of_date ASC;
        """, conn)
    return df

st.title("💰 Daily PnL and Equity Overview")

try:
    df = load_pnl()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.info("No PnL data yet. Run `python -m tools.pnl_snapshot` at least once.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("📊 Current Equity", f"{df['equity'].iloc[-1]:,.2f}")
col2.metric("📈 Today's PnL", f"{df['profit'].iloc[-1]:,.2f}")
col3.metric("📉 PnL %", f"{df['profit_pct'].iloc[-1]*100:.3f}%")

st.line_chart(df.set_index("as_of_date")[["equity"]], height=260)
st.bar_chart(df.set_index("as_of_date")[["profit"]], height=200)

st.subheader("📋 Raw PnL Data")
st.dataframe(df, use_container_width=True)
