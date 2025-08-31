import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import text
from lib.db import get_engine

st.title("📈 Portfolio PnL")

# ---- Sidebar filters
with st.sidebar:
    st.header("Filters")
    pid = st.number_input("Portfolio ID", 1, step=1, value=1)

    # Default 30-day range to avoid ValueError
    default_start = date.today() - timedelta(days=30)
    default_end = date.today()
    d1, d2 = st.date_input("Date range", (default_start, default_end))

@st.cache_data(ttl=300)
def load_pnl(pid, d1, d2):
    q = text("""
        SELECT "date" AS date,
               (realized + unrealized - fees) AS equity
        FROM daily_pnl
        WHERE portfolio_id = :pid
          AND (:d1 IS NULL OR "date" >= :d1)
          AND (:d2 IS NULL OR "date" <= :d2)
        ORDER BY "date"
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"pid": pid, "d1": d1, "d2": d2})

df = load_pnl(pid, d1, d2)

if df.empty:
    st.info("No PnL rows yet.")
    st.stop()

# ---- KPIs
df["ret"] = df["equity"].pct_change()
df["dd"] = df["equity"] / df["equity"].cummax() - 1
rets = df["ret"].dropna()

if len(rets) > 1 and rets.std(ddof=1) > 0:
    sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252)
else:
    sharpe = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Win rate", f"{(rets > 0).mean() * 100:.1f}%")
c2.metric("Sharpe (ann.)", f"{sharpe:.2f}")
c3.metric("Total return", f"{(df.equity.iloc[-1] / df.equity.iloc[0] - 1) * 100:+.1f}%")
c4.metric("Max drawdown", f"{df['dd'].min() * 100:.1f}%")

# ---- Charts
st.subheader("Equity")
st.line_chart(df.set_index("date")[["equity"]])

st.subheader("Drawdown")
st.line_chart(df.set_index("date")[["dd"]])

with st.expander("Data"):
    st.dataframe(df, use_container_width=True)
