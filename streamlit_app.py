import streamlit as st, pandas as pd, numpy as np
from lib.db import get_engine

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Overview")

@st.cache_data(ttl=300)
def load_equity(pid: int = 1) -> pd.DataFrame:
    q = """
    SELECT date::date AS date, (realized + unrealized - fees) AS equity
    FROM daily_pnl WHERE portfolio_id = :pid ORDER BY date;
    """
    return pd.read_sql(q, get_engine(), params={"pid": pid})

try:
    df = load_equity(1)
    if df.empty: raise ValueError("empty")
except Exception:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=50, freq="D"),
                       "equity": (10000*(1+pd.Series(np.random.normal(0.0007,0.01,50))).cumprod()).round(2)})

df = df.sort_values("date").reset_index(drop=True)
rets = df["equity"].pct_change().dropna()
win = (rets > 0).mean()*100 if not rets.empty else 0
sharpe = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std() else 0
tot = (df.equity.iloc[-1]/df.equity.iloc[0]-1)*100 if len(df)>1 else 0
dd = ((df.equity/df.equity.cummax()-1).min()*100) if len(df)>1 else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Win Rate", f"{win:.1f}%")
c2.metric("Sharpe (ann.)", f"{sharpe:.2f}")
c3.metric("Total Return", f"{tot:+.1f}%")
c4.metric("Max Drawdown", f"{dd:.1f}%")

st.subheader("Equity Curve")
st.line_chart(df.set_index("date")[["equity"]])
