import streamlit as st, pandas as pd, numpy as np
from sqlalchemy import text
from lib.db import get_engine

st.title("📈 Portfolio PnL")

with st.sidebar:
    st.header("Filters")
    pid = st.number_input("Portfolio ID", 1, step=1, value=1)
    d1, d2 = st.date_input("Date range", [])

@st.cache_data(ttl=300)
def load_pnl(pid, d1, d2):
    q = text("""
        SELECT date::date AS date,
               (realized + unrealized - fees) AS equity
        FROM daily_pnl
        WHERE portfolio_id = :pid
          AND (:d1::date IS NULL OR date >= :d1)
          AND (:d2::date IS NULL OR date <= :d2)
        ORDER BY date
    """)
    params = {"pid": pid, "d1": d1 if d1 else None, "d2": d2 if d2 else None}
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params=params)

df = load_pnl(pid, d1 if d1 else None, d2 if d2 else None)

if df.empty:
    st.info("No PnL rows yet.")
    st.stop()

df["ret"] = df["equity"].pct_change()
df["dd"] = df["equity"] / df["equity"].cummax() - 1
rets = df["ret"].dropna()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Win rate", f"{(rets>0).mean()*100:.1f}%")
c2.metric("Sharpe (ann.)", f"{(rets.mean()/rets.std()*np.sqrt(252)) if rets.std() else 0:.2f}")
c3.metric("Total return", f"{(df.equity.iloc[-1]/df.equity.iloc[0]-1)*100:+.1f}%")
c4.metric("Max drawdown", f"{df['dd'].min()*100:.1f}%")

st.subheader("Equity")
st.line_chart(df.set_index("date")[["equity"]])

st.subheader("Drawdown")
st.line_chart(df.set_index("date")[["dd"]])

with st.expander("Data"):
    st.dataframe(df, use_container_width=True)
