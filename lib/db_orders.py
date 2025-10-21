# lib/db_orders.py
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime

# prefer psycopg3; fallback psycopg2
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

def _conn():
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

UPSERT_SQL = """
INSERT INTO orders (id, client_order_id, symbol, side, order_type, order_class,
                    qty, filled_qty, filled_avg_price, status, time_in_force,
                    limit_price, stop_price, extended_hours,
                    created_at, submitted_at, updated_at, filled_at, canceled_at, expires_at)
VALUES (%(id)s, %(client_order_id)s, %(symbol)s, %(side)s, %(order_type)s, %(order_class)s,
        %(qty)s, %(filled_qty)s, %(filled_avg_price)s, %(status)s, %(time_in_force)s,
        %(limit_price)s, %(stop_price)s, %(extended_hours)s,
        %(created_at)s, %(submitted_at)s, %(updated_at)s, %(filled_at)s, %(canceled_at)s, %(expires_at)s)
ON CONFLICT (id) DO UPDATE SET
  client_order_id=EXCLUDED.client_order_id,
  symbol=EXCLUDED.symbol,
  side=EXCLUDED.side,
  order_type=EXCLUDED.order_type,
  order_class=EXCLUDED.order_class,
  qty=EXCLUDED.qty,
  filled_qty=EXCLUDED.filled_qty,
  filled_avg_price=EXCLUDED.filled_avg_price,
  status=EXCLUDED.status,
  time_in_force=EXCLUDED.time_in_force,
  limit_price=EXCLUDED.limit_price,
  stop_price=EXCLUDED.stop_price,
  extended_hours=EXCLUDED.extended_hours,
  created_at=EXCLUDED.created_at,
  submitted_at=EXCLUDED.submitted_at,
  updated_at=EXCLUDED.updated_at,
  filled_at=EXCLUDED.filled_at,
  canceled_at=EXCLUDED.canceled_at,
  expires_at=EXCLUDED.expires_at;
"""

def _ts(x):
    if not x:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).replace("Z", "+00:00")
    try:
        from datetime import datetime
        return datetime.fromisoformat(s)
    except Exception:
        return None

def upsert_orders(rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with _conn() as c:
        if HAVE3:
            c.execute("SET TIME ZONE 'UTC';")
        with c.cursor() as cur:
            for o in rows:
                params = {
                    "id": o.get("id"),
                    "client_order_id": o.get("client_order_id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "order_type": (o.get("type") or o.get("order_type")),
                    "order_class": o.get("order_class"),
                    "qty": float(o["qty"]) if o.get("qty") is not None else None,
                    "filled_qty": float(o["filled_qty"]) if o.get("filled_qty") is not None else None,
                    "filled_avg_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                    "status": o.get("status"),
                    "time_in_force": o.get("time_in_force"),
                    "limit_price": float(o["limit_price"]) if o.get("limit_price") else None,
                    "stop_price": float(o["stop_price"]) if o.get("stop_price") else None,
                    "extended_hours": bool(o.get("extended_hours")),
                    "created_at": _ts(o.get("created_at")),
                    "submitted_at": _ts(o.get("submitted_at")),
                    "updated_at": _ts(o.get("updated_at")),
                    "filled_at": _ts(o.get("filled_at")),
                    "canceled_at": _ts(o.get("canceled_at")),
                    "expires_at": _ts(o.get("expires_at")),
                }
                cur.execute(UPSERT_SQL, params)
                count += 1
    return count
