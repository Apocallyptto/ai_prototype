# lib/db_orders.py
from __future__ import annotations
import os
import psycopg2
from typing import Any, Dict, List, Iterable

# Prefer a single env var DB_URL; fall back to PG* variables; finally localhost.
def _dsn_from_env() -> str:
    db_url = os.getenv("DB_URL")
    if db_url:
        return db_url
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "postgres")
    db   = os.getenv("PGDATABASE", "trader")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

def _pg_conn():
    return psycopg2.connect(_dsn_from_env())

# ----------------- Example upsert logic (unchanged API) -----------------
def upsert_orders(orders: Iterable[Dict[str, Any]]) -> int:
    sql = """
    INSERT INTO orders (
        id, client_order_id, symbol, side, order_type, qty, filled_qty,
        limit_price, stop_price, status, created_at, updated_at
    )
    VALUES (
        %(id)s, %(client_order_id)s, %(symbol)s, %(side)s, %(order_type)s, %(qty)s, %(filled_qty)s,
        %(limit_price)s, %(stop_price)s, %(status)s, %(created_at)s, %(updated_at)s
    )
    ON CONFLICT (id) DO UPDATE SET
        client_order_id = EXCLUDED.client_order_id,
        symbol          = EXCLUDED.symbol,
        side            = EXCLUDED.side,
        order_type      = EXCLUDED.order_type,
        qty             = EXCLUDED.qty,
        filled_qty      = EXCLUDED.filled_qty,
        limit_price     = EXCLUDED.limit_price,
        stop_price      = EXCLUDED.stop_price,
        status          = EXCLUDED.status,
        created_at      = EXCLUDED.created_at,
        updated_at      = EXCLUDED.updated_at
    ;
    """
    orders = list(orders)
    if not orders:
        return 0
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, orders)
    return len(orders)
