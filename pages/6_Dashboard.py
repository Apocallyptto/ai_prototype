# pages/6_Dashboard.py
import os
import pandas as pd
import streamlit as st
import sqlalchemy as sa
from datetime import date, timedelta

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

@st.cache_resource(show_spinner=False)
def get_engine():
    # Prefer Streamlit secrets, fallback to env vars
    if "DB_HOST" in st.secrets:
        host = st.secrets["DB_HOST"]; port = st.secrets["DB_PORT"]
        name = st.secrets["DB_NAME"]; user = st.secrets["DB_USER"]; pw = st.secrets["DB_PASSWORD"]
        ssl = st.secrets.get("DB_SSLMODE", "require")
    else:
        host = os.getenv("DB_HOST"); port = os.getenv("DB_PORT")
        name = os.getenv("DB_NAME"); user = os.getenv("DB_USER"); pw = os.getenv("DB_PASSWORD")
        ssl = os.getenv("DB_SSLMODE", "prefer")

    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"
    if ssl:
        url += f"?sslmode={ssl}"
    return sa.create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=60, show_spinner=False)
def load_kpis(days=30, portfolio_id=1):
    eng = get_engine()
    with eng.connect() as c:
        pnl = pd.read_sql(
            sa.text("""
              select date, realized, unrealized, fees,
                     (realized + unrealized - fees) as total
              from daily_pnl
              where portfolio_id = :pid and date >= current_date - interval :days
              order by date
            """),
            c,
            params={"pid": portfolio_id, "days": f"{int(days)} days"},
        )
        orders = pd.read_sql(
            sa.text("""
              select id, ts, symbol, side, qty, price, status, filled_at
              from orders
              order by ts desc
              limit 100
            """), c
        )
        signals = pd.read_sql(
            sa.text("""
              select ts, symbol, signal, strength
              from signals
              order by ts desc
              limit 100
            """), c
        )
    return pnl, orders, signals

left, right = st.columns([1, 3])
with left:
    pid = st.number_input("Portfolio ID", min_value=1, value=1, step=1)
    lookback = st.slider("Lookback (days)", 7, 180, 30)

pnl, orders, signals = load_kpis(days=lookback, portfolio_id=pid)

# Top KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
if not pnl.empty:
    total_pnl = float(pnl["total"].sum())
    last_day = float(pnl["total"].iloc[-1])
    realized_sum = float(pnl["realized"].sum())
    unrealized_last = float(pnl["unrealized"].iloc[-1])
else:
    total_pnl = last_day = realized_sum = unrealized_last = 0.0

kpi1.metric("Total PnL", f"{total_pnl:,.2f}")
kpi2.metric("Last Day PnL", f"{last_day:,.2f}")
kpi3.metric("Realized (sum)", f"{realized_sum:,.2f}")
kpi4.metric("Unrealized (last)", f"{unrealized_last:,.2f}")

# Charts + Tables
tab1, tab2, tab3 = st.tabs(["PnL", "Orders", "Signals"])

with tab1:
    if pnl.empty:
        st.warning("No PnL data yet.")
    else:
        st.line_chart(pnl.set_index("date")[["realized", "unrealized", "fees", "total"]])
        st.dataframe(pnl, use_container_width=True)

with tab2:
    if orders.empty:
        st.info("No orders.")
    else:
        st.dataframe(orders, use_container_width=True)

with tab3:
    if signals.empty:
        st.info("No signals.")
    else:
        st.dataframe(signals, use_container_width=True)
