# tools/init_or_migrate_orders.py
from __future__ import annotations
import os
from typing import Optional

# try psycopg3 first, fall back to psycopg2
try:
    import psycopg  # psycopg3
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    id               text PRIMARY KEY,
    client_order_id  text,
    symbol           text,
    side             text,
    order_type       text,            -- aligns with lib/db_orders.py
    order_class      text,
    qty              text,
    filled_qty       text,
    status           text,
    limit_price      text,
    stop_price       text,
    filled_avg_price text,
    time_in_force    text,
    extended_hours   boolean,
    created_at       timestamptz,
    updated_at       timestamptz,
    submitted_at     timestamptz,
    filled_at        timestamptz,
    canceled_at      timestamptz,
    expired_at       timestamptz,
    failed_at        timestamptz
);
"""

# column -> PostgreSQL type
COLUMNS = {
    "id":               "text",
    "client_order_id":  "text",
    "symbol":           "text",
    "side":             "text",
    "order_type":       "text",
    "order_class":      "text",
    "qty":              "text",
    "filled_qty":       "text",
    "status":           "text",
    "limit_price":      "text",
    "stop_price":       "text",
    "filled_avg_price": "text",
    "time_in_force":    "text",
    "extended_hours":   "boolean",
    "created_at":       "timestamptz",
    "updated_at":       "timestamptz",
    "submitted_at":     "timestamptz",
    "filled_at":        "timestamptz",
    "canceled_at":      "timestamptz",
    "expired_at":       "timestamptz",
    "failed_at":        "timestamptz",
}

DDL_PK = """
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'orders_pkey'
      AND conrelid = 'orders'::regclass
  ) THEN
    ALTER TABLE orders ADD CONSTRAINT orders_pkey PRIMARY KEY (id);
  END IF;
END$$;
"""

DDL_IDX = """
CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders (client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
"""

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

def ensure_table_and_columns():
    with _conn() as c:
        with c.cursor() as cur:
            # 1) create table if missing
            cur.execute(DDL_TABLE)

            # 2) ensure all columns exist
            for col, typ in COLUMNS.items():
                cur.execute("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='orders' AND column_name=%s
                """, (col,))
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(f"ALTER TABLE orders ADD COLUMN {col} {typ};")

            # 3) ensure PK + helpful indices
            cur.execute(DDL_PK)
            for stmt in DDL_IDX.strip().split(";"):
                s = stmt.strip()
                if s:
                    cur.execute(s + ";")
        c.commit()

def main():
    ensure_table_and_columns()
    print("orders table is initialized and migrated (columns ensured).")

if __name__ == "__main__":
    main()
