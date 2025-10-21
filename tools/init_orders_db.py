# tools/init_orders_db.py
from __future__ import annotations
import os

# prefer psycopg3; fallback to psycopg2
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

DDL = """
CREATE TABLE IF NOT EXISTS orders (
  id              TEXT PRIMARY KEY,             -- alpaca order id (uuid)
  client_order_id TEXT,
  symbol          TEXT NOT NULL,
  side            TEXT NOT NULL,
  order_type      TEXT NOT NULL,
  order_class     TEXT,
  qty             DOUBLE PRECISION,
  filled_qty      DOUBLE PRECISION,
  filled_avg_price DOUBLE PRECISION,
  status          TEXT,
  time_in_force   TEXT,
  limit_price     DOUBLE PRECISION,
  stop_price      DOUBLE PRECISION,
  extended_hours  BOOLEAN,
  created_at      TIMESTAMPTZ,
  submitted_at    TIMESTAMPTZ,
  updated_at      TIMESTAMPTZ,
  filled_at       TIMESTAMPTZ,
  canceled_at     TIMESTAMPTZ,
  expires_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders (symbol, created_at DESC);
"""

def main():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv("PGHOST","localhost")
        user = os.getenv("PGUSER","postgres")
        pw   = os.getenv("PGPASSWORD","postgres")
        db   = os.getenv("PGDATABASE","ai_prototype")
        port = os.getenv("PGPORT","5432")
        if HAVE3: conn = psycopg.connect(host=host, user=user, password=pw, dbname=db, port=port, autocommit=True)
        else:
            conn = psycopg2.connect(host=host, user=user, password=pw, dbname=db, port=port)
            conn.autocommit = True
    else:
        if HAVE3: conn = psycopg.connect(dsn, autocommit=True)
        else:
            conn = psycopg2.connect(dsn); conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(DDL)
        print("Ensured table 'orders'.")
    conn.close()
    print("OK")

if __name__ == "__main__":
    main()
