# pages/9_Admin_Import.py
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import sqlalchemy as sa
import streamlit as st


# ---------- Page chrome ----------
st.set_page_config(page_title="Admin / Import", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin / Import")
st.caption("Create/Update tables, reset demo data, and seed examples.")


# ---------- DB utilities ----------
def _require(value, name: str):
    if value in (None, "", "None"):
        st.error(f"Missing database setting: `{name}`")
        st.stop()
    return value


@st.cache_resource(show_spinner=False)
def get_engine():
    """
    Prefer .streamlit/secrets.toml -> [db]; fall back to env vars.
    """
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


def create_or_update_tables(engine):
    """
    Minimal, compatible schemas for the Streamlit pages you have.
    Uses IF NOT EXISTS so it won't clobber existing tables.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        portfolio_id INTEGER NOT NULL,
        date DATE NOT NULL,
        realized DOUBLE PRECISION NOT NULL,
        unrealized DOUBLE PRECISION NOT NULL,
        fees DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (portfolio_id, date)
    );

    CREATE TABLE IF NOT EXISTS signals (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL,
        ticker TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        model TEXT NOT NULL,
        side TEXT NOT NULL,              -- e.g., 'buy' / 'sell'
        strength DOUBLE PRECISION
    );

    CREATE TABLE IF NOT EXISTS orders (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        ticker TEXT NOT NULL,
        side TEXT NOT NULL,              -- 'buy'/'sell'
        qty DOUBLE PRECISION NOT NULL,
        order_type TEXT,                 -- 'market'/'limit'
        limit_price DOUBLE PRECISION,
        status TEXT NOT NULL DEFAULT 'new',
        filled_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS symbols (
        ticker TEXT PRIMARY KEY,
        name TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(sa.text(ddl))


def reset_tables(conn):
    """
    Try fast TRUNCATE; if permissions block it (e.g., in some managed Postgres),
    fall back to DELETE.
    """
    try:
        conn.execute(sa.text("""
            TRUNCATE TABLE orders, signals, daily_pnl RESTART IDENTITY;
        """))
    except Exception:
        conn.execute(sa.text("DELETE FROM orders;"))
        conn.execute(sa.text("DELETE FROM signals;"))
        conn.execute(sa.text("DELETE FROM daily_pnl;"))


# ---------- Demo data ----------
def make_demo_pnl(portfolio_id=1, days=120, seed=7):
    """
    Generate semi-realistic daily PnL for the last `days` business days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.utcnow().date(), periods=days)

    # Random daily returns around 0 with small variance
    # Build a simple equity curve, then decompose into realized/unrealized
    daily_ret = rng.normal(loc=0.0005, scale=0.01, size=len(idx))  # ~0.05% avg daily
    equity = 100_000 * (1 + pd.Series(daily_ret, index=idx)).cumprod()

    # Daily pnl ≈ equity change
    pnl_total = equity.diff().fillna(0.0)

    # Split into realized/unrealized (arbitrary but stable)
    realized = pnl_total * 0.6 + rng.normal(0, 10, size=len(idx))
    unrealized = pnl_total - realized
    fees = np.clip(rng.normal(2.0, 1.0, size=len(idx)), 0.0, None)

    df = pd.DataFrame({
        "portfolio_id": portfolio_id,
        "date": idx.date,
        "realized": realized.round(2),
        "unrealized": unrealized.round(2),
        "fees": fees.round(2),
    })
    return df


def make_demo_signals(ticker="AAPL", n=12, timeframe="1d", model="baseline", seed=11):
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc).replace(hour=17, minute=0, second=0, microsecond=0)

    rows = []
    for i in range(n):
        ts = base_ts - timedelta(days=(n - i))
        side = "buy" if i % 2 == 0 else "sell"
        strength = float(rng.uniform(0.51, 0.85)) if side == "buy" else float(rng.uniform(0.15, 0.49))
        rows.append({
            "ts": ts,
            "ticker": ticker,
            "timeframe": timeframe,
            "model": model,
            "side": side,
            "strength": strength,
        })
    return pd.DataFrame(rows)


def make_demo_orders(ticker="AAPL", n=6):
    base_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    for i in range(n):
        ts = base_ts - timedelta(days=(n - i))
        side = "buy" if i % 2 == 0 else "sell"
        order_type = "limit" if i % 3 == 0 else "market"
        limit_price = 230 + (i * 0.75) if order_type == "limit" else None
        rows.append({
            "ts": ts,
            "ticker": ticker,
            "side": side,
            "qty": 10 + i,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "filled" if i < n - 1 else "new",
            "filled_at": (ts + timedelta(minutes=5)) if i < n - 1 else None,
        })
    return pd.DataFrame(rows)


def seed_demo(engine):
    """
    Reset (truncate/delete) tables, then insert a fresh set of demo rows.
    """
    with engine.begin() as conn:
        reset_tables(conn)

        # Ensure symbols exist (optional)
        symbols = pd.DataFrame([
            {"ticker": "AAPL", "name": "Apple Inc."},
            {"ticker": "MSFT", "name": "Microsoft Corp."},
            {"ticker": "SPY",  "name": "SPDR S&P 500 ETF"},
        ])
        symbols.to_sql("symbols", conn, if_exists="append", index=False)

        # Insert demo PnL, Signals, Orders
        make_demo_pnl().to_sql("daily_pnl", conn, if_exists="append", index=False)
        make_demo_signals().to_sql("signals", conn, if_exists="append", index=False)
        make_demo_orders().to_sql("orders", conn, if_exists="append", index=False)


def count_rows(engine):
    with engine.connect() as c:
        q = lambda t: c.execute(sa.text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
        return {
            "daily_pnl": q("daily_pnl"),
            "signals": q("signals"),
            "orders": q("orders"),
            "symbols": q("symbols"),
        }


# ---------- UI actions ----------
eng = get_engine()

col1, col2 = st.columns(2)

with col1:
    if st.button("Create/Update Tables", type="primary", use_container_width=True):
        try:
            create_or_update_tables(eng)
            st.success("Tables created/verified ✅")
            st.json(count_rows(eng))
        except Exception as e:
            st.exception(e)

with col2:
    if st.button("Seed Demo Data", type="secondary", use_container_width=True):
        try:
            create_or_update_tables(eng)
            seed_demo(eng)
            st.success("Demo data seeded ✅")
            st.json(count_rows(eng))
        except Exception as e:
            st.exception(e)

st.divider()

with st.expander("Table row counts"):
    try:
        st.json(count_rows(eng))
    except Exception as e:
        st.warning("Could not fetch counts yet.")
        st.caption(str(e))
