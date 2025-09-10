# app/dashboard.py
import os
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import sqlalchemy as sa
import streamlit as st


# ----------------------------
# Config & connection helpers
# ----------------------------
def _build_db_url() -> Optional[str]:
    """
    Priority:
      1) TEST_DB_URL (full SQLAlchemy URL)
      2) Compose from DB_* envs (Postgres)
      3) Fallback to SQLite (local dev / demo)
    """
    url = os.getenv("TEST_DB_URL")
    if url:
        return url

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")
    sslmode = os.getenv("DB_SSLMODE", "prefer")  # "require" on many hosted DBs

    if all([host, port, name, user, pwd]):
        return (
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
            f"?sslmode={sslmode}"
        )

    # Fallback: local demo (requires your tests or ETL to have run at least once)
    st.warning("DB_* env vars not set; using local SQLite (./app_local.db).")
    return "sqlite:///app_local.db"


@st.cache_resource(show_spinner=False)
def get_engine():
    url = _build_db_url()
    return sa.create_engine(url, pool_pre_ping=True)


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_pnl(
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load PnL from daily_pnl, optional date filtering.
    Returns columns: portfolio_id, date, realized, unrealized, fees, net
    """
    eng = get_engine()
    where = []
    params = {}

    if start:
        where.append("date >= :start")
        params["start"] = start
    if end:
        where.append("date <= :end")
        params["end"] = end

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = sa.text(f"""
        SELECT portfolio_id, date, realized, unrealized, fees
        FROM daily_pnl
        {where_sql}
        ORDER BY date ASC, portfolio_id ASC
    """)

    with eng.connect() as c:
        df = pd.read_sql(sql, c, params=params)

    if df.empty:
        return df

    # Ensure 'date' is datetime64[ns] for charts
    df["date"] = pd.to_datetime(df["date"])

    # Compute net
    df["net"] = df["realized"] + df["unrealized"] - df["fees"]

    return df


def daterange_preset(preset: str) -> Tuple[Optional[date], Optional[date]]:
    today = date.today()
    if preset == "Last 30 days":
        return today - timedelta(days=30), today
    if preset == "Last 90 days":
        return today - timedelta(days=90), today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    if preset == "All":
        return None, None
    return None, None


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="PnL Dashboard", layout="wide")
st.title("📈 Daily PnL Dashboard")

# Sidebar: DB info + filters
with st.sidebar:
    st.header("Database")
    db_url_preview = _build_db_url()
    st.caption("Using:")
    st.code(db_url_preview, language="text")

    # Date filters
    st.header("Filters")
    preset = st.selectbox("Date range", ["Last 30 days", "Last 90 days", "YTD", "All", "Custom"])
    start, end = daterange_preset(preset)
    if preset == "Custom":
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=start or date.today() - timedelta(days=30))
        with c2:
            end = st.date_input("End", value=end or date.today())

    refresh = st.button("🔄 Refresh")

# Load data
if refresh:
    st.cache_data.clear()

df = load_pnl(start, end)

if df.empty:
    st.info("No PnL rows found for the selected range.")
    st.stop()

# Portfolio filter
pids = sorted(df["portfolio_id"].unique())
pid = st.selectbox("Portfolio", pids)

dff = df[df["portfolio_id"] == pid].copy().sort_values("date").reset_index(drop=True)

# Metrics row
latest = dff.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio", f"{pid}")
c2.metric("Last Net", f"{latest['net']:,.2f}")
c3.metric("Last Realized", f"{latest['realized']:,.2f}")
c4.metric("Last Unrealized", f"{latest['unrealized']:,.2f}")
c5.metric("Last Fees", f"{latest['fees']:,.2f}")

# Equity curve (cumulative net)
dff["cum_net"] = dff["net"].cumsum()

st.subheader("Equity Curve (Cumulative Net)")
st.line_chart(dff.set_index("date")[["cum_net"]])

st.subheader("PnL Components")
st.line_chart(dff.set_index("date")[["realized", "unrealized", "net"]])

with st.expander("Raw data"):
    st.dataframe(dff, use_container_width=True)
