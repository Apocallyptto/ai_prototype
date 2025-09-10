# pages/9_Admin_Import.py
import os
from datetime import date, timedelta
import random

import pandas as pd
import sqlalchemy as sa
import streamlit as st

st.set_page_config(page_title="Admin Import", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin / Import")

@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets.get("db", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    port = int(cfg.get("port") or os.getenv("DB_PORT", 5432))
    db   = cfg.get("dbname") or os.getenv("DB_NAME")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASSWORD")
    ssl  = cfg.get("sslmode") or os.getenv("DB_SSLMODE", "require")
    url = sa.engine.URL.create(
        "postgresql+psycopg2", user, pw, host, port, db,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)

eng = get_engine()

DDL = [
    """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        portfolio_id INTEGER NOT NULL,
        date         DATE     NOT NULL,
        realized     DOUBLE PRECISION NOT NULL,
        unrealized   DOUBLE PRECISION NOT NULL,
        fees         DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (portfolio_id, date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id         BIGSERIAL PRIMARY KEY,
        ts         TIMESTAMP NOT NULL DEFAULT now(),
        symbol     TEXT      NOT NULL,
        side       TEXT      NOT NULL,  -- BUY / SELL
        qty        DOUBLE PRECISION NOT NULL,
        price      DOUBLE PRECISION NOT NULL,
        status     TEXT      NOT NULL DEFAULT 'filled',
        filled_at  TIMESTAMP NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        ts        TIMESTAMP NOT NULL DEFAULT now(),
        symbol    TEXT      NOT NULL,
        signal    TEXT      NOT NULL,   -- e.g. BUY / SELL / HOLD
        strength  DOUBLE PRECISION NOT NULL DEFAULT 1.0
    );
    """,
]

def ensure_schema():
    with eng.begin() as c:
        for stmt in DDL:
            c.execute(sa.text(stmt))

def seed_demo():
    today = date.today()
    # daily_pnl for last 60 days
    rows = []
    eq = 0.0
    for i in range(60, -1, -1):
        d = today - timedelta(days=i)
        realized = round(random.uniform(-50, 50), 2)
        unrealized = round(random.uniform(-20, 20), 2)
        fees = round(random.uniform(0, 2), 2)
        rows.append({
            "portfolio_id": 1,
            "date": d,
            "realized": realized,
            "unrealized": unrealized,
            "fees": fees,
        })
    df = pd.DataFrame(rows)
    with eng.begin() as c:
        df.to_sql("daily_pnl", c, if_exists="append", index=False)

    # 40 orders
    orders = []
    for i in range(40):
        orders.append({
            "ts": f"{today} 10:{i:02d}:00",
            "symbol": random.choice(["AAPL","MSFT","NVDA","TSLA"]),
            "side": random.choice(["BUY","SELL"]),
            "qty": round(random.uniform(1, 5), 2),
            "price": round(random.uniform(50, 500), 2),
            "status": "filled",
            "filled_at": f"{today} 10:{i:02d}:30",
        })
    df_o = pd.DataFrame(orders)
    with eng.begin() as c:
        df_o.to_sql("orders", c, if_exists="append", index=False)

    # 40 signals
    sigs = []
    for i in range(40):
        sigs.append({
            "ts": f"{today} 09:{i:02d}:00",
            "symbol": random.choice(["AAPL","MSFT","NVDA","TSLA"]),
            "signal": random.choice(["BUY","SELL","HOLD"]),
            "strength": round(random.uniform(0.3, 1.0), 2),
        })
    df_s = pd.DataFrame(sigs)
    with eng.begin() as c:
        df_s.to_sql("signals", c, if_exists="append", index=False)

col1, col2 = st.columns(2)
with col1:
    if st.button("Create/Update Tables", use_container_width=True, type="primary"):
        ensure_schema()
        st.success("Schema ensured ✅")

with col2:
    if st.button("Seed Demo Data", use_container_width=True):
        ensure_schema()
        seed_demo()
        st.success("Demo data inserted ✅")
