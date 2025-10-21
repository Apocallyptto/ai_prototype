# lib/db_orders.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Prefer psycopg3, fallback to psycopg2
try:
    import psycopg  # psycopg3
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2

def _pg_conn():
    dsn = os.getenv("DATABASE_URL")
    if HAVE3:
        return psycopg.connect(dsn) if dsn else psycopg.connect(
            host=os.getenv("PGHOST","localhost"),
            user=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD","postgres"),
            dbname=os.getenv("PGDATABASE","ai_prototype"),
            port=os.getenv("PGPORT","5432"),
        )
    else:
        return psycopg2.connect(dsn) if dsn else psycopg2.connect(
            host=os.getenv("PGHOST","localhost"),
            user=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD","postgres"),
            dbname=os.getenv("PGDATABASE","ai_prototype"),
            port=os.getenv("PGPORT","5432"),
        )

def _ts(x: Any) -> Optional[datetime]:
    """Return a UTC-aware datetime or None."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    if isinstance(x, str):
        s = x.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None

def upsert_orders(orders: List[Dict[str, Any]]) -> int:
    """
    Upsert a list of Alpaca order dicts into the 'orders' table.
    Assumes the table exists (created by tools.init_orders_db).
    """
    if not orders:
        return 0

    cols = [
        "id", "client_order_id", "symbol", "side", "order_type", "order_class",
        "qty", "filled_qty", "status", "limit_price", "stop_price",
        "filled_avg_price", "time_in_force", "extended_hours",
        "created_at", "updated_at", "submitted_at", "filled_at",
        "canceled_at", "expired_at", "failed_at",
    ]
    placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"

    sql = f"""
    INSERT INTO orders ({",".join(cols)})
    VALUES {placeholders}
    ON CONFLICT (id) DO UPDATE SET
      client_order_id = EXCLUDED.client_order_id,
      symbol          = EXCLUDED.symbol,
      side            = EXCLUDED.side,
      order_type      = EXCLUDED.order_type,
      order_class     = EXCLUDED.order_class,
      qty             = EXCLUDED.qty,
      filled_qty      = EXCLUDED.filled_qty,
      status          = EXCLUDED.status,
      limit_price     = EXCLUDED.limit_price,
      stop_price      = EXCLUDED.stop_price,
      filled_avg_price= EXCLUDED.filled_avg_price,
      time_in_force   = EXCLUDED.time_in_force,
      extended_hours  = EXCLUDED.extended_hours,
      created_at      = EXCLUDED.created_at,
      updated_at      = EXCLUDED.updated_at,
      submitted_at    = EXCLUDED.submitted_at,
      filled_at       = EXCLUDED.filled_at,
      canceled_at     = EXCLUDED.canceled_at,
      expired_at      = EXCLUDED.expired_at,
      failed_at       = EXCLUDED.failed_at
    """

    def row(o: Dict[str, Any]) -> tuple:
        return (
            o.get("id"),
            o.get("client_order_id"),
            o.get("symbol"),
            o.get("side"),
            o.get("type") or o.get("order_type"),
            o.get("order_class"),
            o.get("qty"),
            o.get("filled_qty"),
            o.get("status"),
            o.get("limit_price"),
            o.get("stop_price"),
            o.get("filled_avg_price"),
            o.get("time_in_force"),
            bool(o.get("extended_hours", False)),
            _ts(o.get("created_at")),
            _ts(o.get("updated_at")),
            _ts(o.get("submitted_at")),
            _ts(o.get("filled_at")),
            _ts(o.get("canceled_at")),
            _ts(o.get("expired_at")),
            _ts(o.get("failed_at")),
        )

    rows = [row(o) for o in orders]

    with _pg_conn() as c:
        with c.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
        c.commit()
    return len(rows)
