# pages/6_Dashboard.py
import os
from datetime import date, timedelta

import pandas as pd
import sqlalchemy as sa
import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

def _require(value, name: str):
    if value in (None, "", "None"):
        st.error(f"Missing database setting: `{name}`")
        st.stop()
    return value

@st.cache_resource(show_spinner=False)
def get_engine():
    # Prefer .streamlit/secrets.toml -> [db] section; fall back to env vars.
    cfg = st.secrets.get("db", {})

    host = cfg.get("host") or os.getenv("DB_HOST")
    port = cfg.get("port") or os.getenv("DB_PORT", 5432)
    db   = cfg.get("dbname") or cfg.get("name") or os.getenv("DB_NAME")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASSWORD")
    ssl  = cfg.get("sslmode") or os.getenv("DB_SSLMODE")  # e.g. "require" for Neon

    host = _require(host, "host")
    db   = _require(db, "dbname")
    user = _require(user, "user")
    pw   = _require(pw, "password")
    try:
        port = int(port)
    except Exception:
        st.error(f"DB `port` must be an integer, got: {port!r}")
        st.stop()

    url = sa.engine.URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=pw,
        host=host,
        port=port,
        database=db,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=60, show_spinner=False)
def load_kpis(days=30, portfolio_id=1):
    eng = get_engine()
    with eng.connect() as c:
        pnl = pd.read_sql(
            sa.text("""
                SELECT
                    date, realized, unrealized, fees,
                    (realized + unrealized - fees) AS total
                FROM daily_pnl
                WHERE portfolio_id = :pid
                  AND date >= current_date - (:days || ' days')::interval
                ORDER BY date
            """),
            c,
            params={"pid": int(portfolio_id), "days": int(days)},
        )

        orders = pd.read_sql(
            sa.text("""
                SELECT id, ts, symbol, side, qty, price, status, filled_at
                FROM orders
                ORDER BY ts DESC
                LIMIT 100
            """),
            c
        )

        signals = pd.read_sql(
            sa.text("""
                SELECT ts, symbol, signal, strength
                FROM signals
                ORDER BY ts DESC
                LIMIT 100
            """),
            c
        )
    return pnl, orders, signals

left, right = st.columns([1, 3])
with left:
    pid = st.number_input("Portfolio ID", min_value=1, value=1, step=1)
    lookback = st.slider("Lookback (days)", 7, 180, 30)

pnl, orders, signals = load_kpis(days=lookback, portfolio_id=pid)

# KPIs
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
