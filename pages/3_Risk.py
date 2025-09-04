# pages/3_Risk.py
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text
from lib.db import get_engine

st.set_page_config(page_title="Risk", page_icon="🛡️", layout="wide")
st.title("🛡️ Risk Dashboard")

# ---- Sidebar controls
with st.sidebar:
    st.header("Filters")
    pid = st.number_input("Portfolio ID", 1, step=1, value=1)
    lookback = st.number_input("Lookback (trading days)", 50, 2000, value=252, step=10)
    conf = st.select_slider("Confidence (for VaR/ES)", options=[0.90, 0.95, 0.975, 0.99, 0.995], value=0.99)
    horizon = st.number_input("Horizon (days)", 1, 30, value=1)
    show_worst = st.number_input("Show worst N days", 5, 50, value=10)


@st.cache_data(ttl=300)
def load_equity_series(portfolio_id: int) -> pd.DataFrame:
    q = text("""
        SELECT date::date AS date,
               (realized + unrealized - fees) AS equity
        FROM daily_pnl
        WHERE portfolio_id = :pid
        ORDER BY date
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"pid": portfolio_id})
    # drop NaNs/negatives defensively
    df = df.dropna(subset=["equity"]).copy()
    df["equity"] = df["equity"].astype(float)
    return df


def compute_var_es(returns: pd.Series, conf_level: float = 0.99, horizon_days: int = 1):
    """
    Historical VaR / ES on daily returns with square-root-of-time scaling.
    Returns (var_pct, es_pct) as positive loss percentages.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan, np.nan
    scale = float(np.sqrt(max(horizon_days, 1)))
    q = float(np.quantile(r, 1.0 - conf_level))  # negative number (loss) for lower tail
    var_pct = -q * scale
    es_slice = r[r <= q]
    es_pct = float(-es_slice.mean() * scale) if len(es_slice) else np.nan
    return var_pct, es_pct


# ---- Load data
df = load_equity_series(pid)

if df.empty or len(df) < 3:
    st.info("Not enough PnL/equity data yet. Once `daily_pnl` has rows, risk will populate.")
    st.stop()

# Respect lookback
df = df.tail(int(lookback)).reset_index(drop=True)

# ---- Returns & drawdowns
df["ret"] = df["equity"].pct_change()
df["peak"] = df["equity"].cummax()
df["dd"] = df["equity"] / df["peak"] - 1.0
max_dd = float(df["dd"].min())

# VaR / ES (historical)
var_pct, es_pct = compute_var_es(df["ret"], conf_level=conf, horizon_days=horizon)
last_equity = float(df["equity"].iloc[-1])
var_amt = last_equity * var_pct if np.isfinite(var_pct) else np.nan
es_amt = last_equity * es_pct if np.isfinite(es_pct) else np.nan

# Volatility (annualized) over lookback
daily_vol = float(df["ret"].std())
ann_vol = daily_vol * np.sqrt(252) if np.isfinite(daily_vol) else np.nan

# Some extras
ytd = None
try:
    ytd_df = df.copy()
    ytd_df = ytd_df[ytd_df["date"] >= pd.Timestamp(pd.Timestamp.today().year, 1, 1)]
    if len(ytd_df) >= 2:
        ytd = float(ytd_df["equity"].iloc[-1] / ytd_df["equity"].iloc[0] - 1.0)
except Exception:
    pass

# ---- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity (last)", f"{last_equity:,.0f}")
c2.metric(f"VaR {int(conf*100)}% ({horizon}d)", f"-{var_pct*100:.2f}%", f"{-var_amt:,.0f}")
c3.metric(f"ES {int(conf*100)}% ({horizon}d)", f"-{es_pct*100:.2f}%", f"{-es_amt:,.0f}")
c4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
c5.metric("Vol (ann.)", f"{ann_vol*100:.2f}%")

# ---- Charts
st.subheader("Equity")
st.line_chart(df.set_index("date")[["equity"]])

st.subheader("Drawdown")
st.area_chart(df.set_index("date")[["dd"]])

# Return histogram (coarse but helpful)
st.subheader("Return distribution")
hist = np.histogram(df["ret"].dropna(), bins=30)
hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
st.bar_chart(hist_df.set_index("bin_left"))

# ---- Worst days table
st.subheader(f"Worst {show_worst} daily returns")
worst = df.dropna(subset=["ret"]).nsmallest(int(show_worst), "ret")[["date", "ret", "equity", "dd"]]
worst.rename(columns={"ret": "ret (d)", "dd": "drawdown"}, inplace=True)
st.dataframe(worst, use_container_width=True)

# ---- Download button
with st.expander("Download data"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (risk window)", csv, file_name=f"risk_window_pid{pid}.csv", mime="text/csv")
