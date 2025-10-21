# tools/migrate_orders_table.py
from __future__ import annotations
import os
from datetime import datetime
try:
    import psycopg
    HAVE3=True
except Exception:
    HAVE3=False
    import psycopg2

def conn():
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

SQL_ADD = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='orders' AND column_name='order_type'
    ) THEN
        ALTER TABLE orders ADD COLUMN order_type text;
    END IF;
END$$;
"""

def main():
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(SQL_ADD)
        c.commit()
    print("orders table migration OK (ensured order_type column).")

if __name__ == "__main__":
    main()
