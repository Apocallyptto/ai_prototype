# pages/9_Admin_Import.py
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import sqlalchemy as sa
import streamlit as st

st.set_page_config(page_title="Admin / Import", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin / Import")
st.caption("Create/Update tables, migrate schema if needed, and seed demo data.")

# ---------- DB helpers ----------
def _require(value, name: str):
    if value in (None, "", "None"):
        st.error(f"Missing database setting: `{name}`")
        st.stop()
    return value

@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets.get("db", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    port = cfg.get("port") or os.getenv("DB_PORT", 5432)
    db   = cfg.get("dbname") or cfg.get("name") or os.getenv("DB_NAME")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASSWORD")
    ssl  = cfg.get("sslmode") or os.getenv("DB_SSLMODE")

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

def get_table_columns(conn, table: str) -> set[str]:
    rows = conn.execute(sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
    """), {"t": table}).fetchall()
    return {r[0] for r in rows}

def get_column_types(conn, table: str) -> dict:
    rows = conn.execute(sa.text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
    """), {"t": table}).fetchall()
    return {r[0]: r[1] for r in rows}

# ---------- Create / migrate schema ----------
def create_or_update_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        portfolio_id INTEGER NOT NULL,
        date DATE NOT NULL,
        realized DOUBLE PRECISION NOT NULL,
        unrealized DOUBLE PRECISION NOT NULL,
        fees DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (portfolio_id, date)
    );

    CREATE TABLE IF NOT EXISTS orders (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now()
        -- other columns added or migrated below
    );

    CREATE TABLE IF NOT EXISTS signals (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL
        -- other columns added or migrated below
    );

    CREATE TABLE IF NOT EXISTS symbols (
        ticker TEXT PRIMARY KEY
    );
    """
    with engine.begin() as conn:
        conn.execute(sa.text(ddl))
        # Ensure symbols.name exists (ignore if no perms)
        try:
            conn.execute(sa.text("ALTER TABLE symbols ADD COLUMN IF NOT EXISTS name TEXT;"))
        except Exception:
            pass
        migrate_orders_schema(conn)
        migrate_signals_schema(conn)

def migrate_orders_schema(conn):
    cols = get_table_columns(conn, "orders")
    # Legacy rename: symbol -> ticker
    if "ticker" not in cols and "symbol" in cols:
        try:
            conn.execute(sa.text('ALTER TABLE orders RENAME COLUMN symbol TO ticker;'))
        except Exception:
            pass
        cols = get_table_columns(conn, "orders")

    # Ensure required columns exist
    needed = {
        "ts": "TIMESTAMPTZ",
        "ticker": "TEXT",
        "side": "TEXT",
        "qty": "DOUBLE PRECISION",
        "order_type": "TEXT",
        "limit_price": "DOUBLE PRECISION",
        "status": "TEXT",
        "filled_at": "TIMESTAMPTZ",
    }
    for name, typ in needed.items():
        if name not in cols:
            try:
                conn.execute(sa.text(f"ALTER TABLE orders ADD COLUMN {name} {typ};"))
            except Exception:
                pass

    # Normalize types (JSON/JSONB/text, numeric)
    types = get_column_types(conn, "orders")
    for name in ("ticker", "side", "order_type", "status"):
        t = types.get(name)
        if t in ("json", "jsonb"):
            try:
                conn.execute(sa.text(
                    f"ALTER TABLE orders ALTER COLUMN {name} TYPE TEXT "
                    f"USING trim(both '\"' from {name}::text)"
                ))
            except Exception:
                pass

    # Cast qty/limit_price to double precision if odd types
    types = get_column_types(conn, "orders")
    if types.get("qty") not in ("double precision", "numeric", "real", "integer", "bigint", "smallint"):
        try:
            conn.execute(sa.text(
                "ALTER TABLE orders ALTER COLUMN qty TYPE DOUBLE PRECISION USING qty::double precision"
            ))
        except Exception:
            pass
    types = get_column_types(conn, "orders")
    if "limit_price" in types and types.get("limit_price") not in ("double precision", "numeric", "real", "integer", "bigint", "smallint"):
        try:
            conn.execute(sa.text(
                "ALTER TABLE orders ALTER COLUMN limit_price TYPE DOUBLE PRECISION USING limit_price::double precision"
            ))
        except Exception:
            pass

def migrate_signals_schema(conn):
    cols = get_table_columns(conn, "signals")

    # Legacy renames
    if "ticker" not in cols and "symbol" in cols:
        try:
            conn.execute(sa.text('ALTER TABLE signals RENAME COLUMN symbol TO ticker;'))
        except Exception:
            pass
        cols = get_table_columns(conn, "signals")

    if "side" not in cols and "signal" in cols:
        try:
            conn.execute(sa.text('ALTER TABLE signals RENAME COLUMN signal TO side;'))
        except Exception:
            pass
        cols = get_table_columns(conn, "signals")

    # Ensure required columns exist
    needed = {
        "ticker": "TEXT",
        "timeframe": "TEXT",
        "model": "TEXT",
        "side": "TEXT",
        "strength": "DOUBLE PRECISION",
    }
    for name, typ in needed.items():
        if name not in cols:
            try:
                conn.execute(sa.text(f"ALTER TABLE signals ADD COLUMN {name} {typ};"))
            except Exception:
                pass

    # Convert JSON-ish columns to TEXT
    types = get_column_types(conn, "signals")
    for name in ("ticker", "timeframe", "model", "side"):
        t = types.get(name)
        if t in ("json", "jsonb"):
            try:
                conn.execute(sa.text(
                    f"ALTER TABLE signals ALTER COLUMN {name} TYPE TEXT "
                    f"USING trim(both '\"' from {name}::text)"
                ))
            except Exception:
                pass

    # Strength numeric
    t = get_column_types(conn, "signals").get("strength")
    if t not in (None, "double precision", "numeric", "real", "integer", "bigint", "smallint"):
        try:
            conn.execute(sa.text(
                "ALTER TABLE signals ALTER COLUMN strength TYPE DOUBLE PRECISION USING strength::double precision"
            ))
        except Exception:
            pass

def reset_tables(conn):
    try:
        conn.execute(sa.text("TRUNCATE TABLE orders, signals, daily_pnl RESTART IDENTITY;"))
    except Exception:
        conn.execute(sa.text("DELETE FROM orders;"))
        conn.execute(sa.text("DELETE FROM signals;"))
        conn.execute(sa.text("DELETE FROM daily_pnl;"))

def upsert_symbols(conn, rows):
    cols = get_table_columns(conn, "symbols")
    if "name" in cols:
        conn.execute(sa.text("""
            INSERT INTO symbols (ticker, name)
            VALUES (:ticker, :name)
            ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name;
        """), rows)
    else:
        conn.execute(sa.text("""
            INSERT INTO symbols (ticker) VALUES (:ticker)
            ON CONFLICT (ticker) DO NOTHING;
        """), [{"ticker": r["ticker"]} for r in rows])

# ---------- Demo data ----------
def make_demo_pnl(portfolio_id=1, days=120, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.utcnow().date(), periods=days)

    daily_ret = rng.normal(0.0005, 0.01, size=len(idx))
    equity = 100_000 * (1 + pd.Series(daily_ret, index=idx)).cumprod()
    pnl_total = equity.diff().fillna(0.0)

    realized = pnl_total * 0.6 + rng.normal(0, 10, size=len(idx))
    unrealized = pnl_total - realized
    fees = np.clip(rng.normal(2.0, 1.0, size=len(idx)), 0.0, None)

    return pd.DataFrame({
        "portfolio_id": portfolio_id,
        "date": idx.date,
        "realized": realized.round(2),
        "unrealized": unrealized.round(2),
        "fees": fees.round(2),
    })

def make_demo_signals(ticker="AAPL", n=12, timeframe="1d", model="baseline", seed=11):
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc).replace(hour=17, minute=0, second=0, microsecond=0)
    rows = []
    for i in range(n):
        ts = base_ts - timedelta(days=(n - i))
        side = "buy" if i % 2 == 0 else "sell"
        strength = float(rng.uniform(0.51, 0.85)) if side == "buy" else float(rng.uniform(0.15, 0.49))
        rows.append({"ts": ts, "ticker": ticker, "timeframe": timeframe, "model": model,
                     "side": side, "strength": strength})
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
            "ts": ts, "ticker": ticker, "side": side, "qty": 10 + i,
            "order_type": order_type, "limit_price": limit_price,
            "status": "filled" if i < n - 1 else "new",
            "filled_at": (ts + timedelta(minutes=5)) if i < n - 1 else None,
        })
    return pd.DataFrame(rows)

# ---------- Seeding workflow ----------
def seed_demo(engine):
    with engine.begin() as conn:
        migrate_orders_schema(conn)
        migrate_signals_schema(conn)
        reset_tables(conn)

        # Symbols
        demo_symbols = [
            {"ticker": "AAPL", "name": "Apple Inc."},
            {"ticker": "MSFT", "name": "Microsoft Corp."},
            {"ticker": "SPY",  "name": "SPDR S&P 500 ETF"},
        ]
        upsert_symbols(conn, demo_symbols)

        # PnL
        make_demo_pnl().to_sql("daily_pnl", conn, if_exists="append", index=False)

        # Signals — adapt to actual columns/types
        sig_cols = get_table_columns(conn, "signals")
        sig_types = get_column_types(conn, "signals")
        df_sig = make_demo_signals()

        if "ticker" not in sig_cols and "symbol" in sig_cols:
            df_sig = df_sig.rename(columns={"ticker": "symbol"})
        if "side" not in sig_cols and "signal" in sig_cols:
            df_sig = df_sig.rename(columns={"side": "signal"})

        for col in ("ticker", "timeframe", "model", "side"):
            if col in sig_cols and sig_types.get(col) in ("json", "jsonb"):
                df_sig[col] = df_sig[col].map(lambda x: f"\"{x}\"" if x is not None and not str(x).startswith('"') else x)

        keep = [c for c in df_sig.columns if c in sig_cols]
        if "ts" in sig_cols and "ts" not in keep:
            keep = ["ts"] + keep
        df_sig = df_sig[keep]
        df_sig.to_sql("signals", conn, if_exists="append", index=False)

        # Orders — adapt to actual columns/types
        ord_cols = get_table_columns(conn, "orders")
        ord_types = get_column_types(conn, "orders")
        df_ord = make_demo_orders()

        if "ticker" not in ord_cols and "symbol" in ord_cols:
            df_ord = df_ord.rename(columns={"ticker": "symbol"})

        # Fallback for JSON-ish text columns
        for col in ("ticker", "side", "order_type", "status"):
            if col in ord_cols and ord_types.get(col) in ("json", "jsonb"):
                df_ord[col] = df_ord[col].map(lambda x: f"\"{x}\"" if x is not None and not str(x).startswith('"') else x)

        keep = [c for c in df_ord.columns if c in ord_cols]
        if "ts" in ord_cols and "ts" not in keep:
            keep = ["ts"] + keep
        df_ord = df_ord[keep]
        df_ord.to_sql("orders", conn, if_exists="append", index=False)

def count_rows(engine):
    with engine.connect() as c:
        def q(t): return c.execute(sa.text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
        return {"symbols": q("symbols"), "signals": q("signals"),
                "orders": q("orders"), "daily_pnl": q("daily_pnl")}

# ---------- UI ----------
eng = get_engine()
c1, c2 = st.columns(2)

with c1:
    if st.button("Create/Update Tables", type="primary", use_container_width=True):
        try:
            create_or_update_tables(eng)
            st.success("Tables created / verified ✅")
            st.json(count_rows(eng))
        except Exception as e:
            st.exception(e)

with c2:
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
