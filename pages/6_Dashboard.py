# pages/6_Dashboard.py
import os
from datetime import date, timedelta
import pandas as pd
import sqlalchemy as sa
import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

def _require(v, name):
    if not v: st.error(f"Missing DB setting: {name}"); st.stop()
    return v

@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets.get("db", {})
    host = _require(cfg.get("host") or os.getenv("DB_HOST"), "host")
    port = int(cfg.get("port") or os.getenv("DB_PORT", 5432))
    db   = _require(cfg.get("dbname") or os.getenv("DB_NAME"), "dbname")
    user = _require(cfg.get("user") or os.getenv("DB_USER"), "user")
    pw   = _require(cfg.get("password") or os.getenv("DB_PASSWORD"), "password")
    ssl  = cfg.get("sslmode") or os.getenv("DB_SSLMODE", "require")
    url = sa.engine.URL.create("postgresql+psycopg2", user, pw, host, port, db,
                               query={"sslmode": ssl} if ssl else None)
    return sa.create_engine(url, pool_pre_ping=True)

def _safe_read_sql(sql, conn, **kwargs):
    try:
        return pd.read_sql(sa.text(sql), conn, **kwargs)
    except Exception:
        # Most likely: relation does not exist yet
        st.info("No data yet. Go to **Admin Import** → **Create/Update Tables** (and optionally **Seed Demo Data**).")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def load_kpis(days=30, portfolio_id=1):
    eng = get_engine()
    with eng.connect() as c:
        pnl = _safe_read_sql(
            """
            SELECT date, realized, unrealized, fees,
                   (realized + unrealized - fees) AS total
            FROM daily_pnl
            WHERE portfolio_id = :pid
              AND date >= current_date - (:days || ' days')::interval
            ORDER BY date
            """,
            c, params={"pid": int(portfolio_id), "days": int(days)}
        )
        orders = _safe_read_sql(
            "SELECT id, ts, symbol, side, qty, price, status, filled_at FROM orders ORDER BY ts DESC LIMIT 100",
            c
        )
        signals = _safe_read_sql(
            "SELECT ts, symbol, signal, strength FROM signals ORDER BY ts DESC LIMIT 100",
            c
        )
    return pnl, orders, signals

left, right = st.columns([1, 3])
with left:
    pid = st.number_input("Portfolio ID", min_value=1, value=1, step=1)
    lookback = st.slider("Lookback (days)", 7, 180, 30)

pnl, orders, signals = load_kpis(days=lookback, portfolio_id=pid)

k1, k2, k3, k4 = st.columns(4)
if not pnl.empty:
    k1.metric("Total PnL", f"{float(pnl['total'].sum()):,.2f}")
    k2.metric("Last Day PnL", f"{float(pnl['total'].iloc[-1]):,.2f}")
    k3.metric("Realized (sum)", f"{float(pnl['realized'].sum()):,.2f}")
    k4.metric("Unrealized (last)", f"{float(pnl['unrealized'].iloc[-1]):,.2f}")
else:
    for col in (k1, k2, k3, k4):
        col.metric("-", "0.00")

tab1, tab2, tab3 = st.tabs(["PnL", "Orders", "Signals"])
with tab1:
    if pnl.empty: st.warning("No PnL data yet.")
    else:
        st.line_chart(pnl.set_index("date")[["realized","unrealized","fees","total"]])
        st.dataframe(pnl, use_container_width=True)
with tab2:
    st.dataframe(orders if not orders.empty else pd.DataFrame(), use_container_width=True)
with tab3:
    st.dataframe(signals if not signals.empty else pd.DataFrame(), use_container_width=True)
