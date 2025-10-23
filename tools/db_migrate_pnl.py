# tools/db_migrate_pnl.py
from __future__ import annotations
import os, sys
import psycopg2

DDL = """
CREATE TABLE IF NOT EXISTS public.daily_pnl (
    id           BIGSERIAL PRIMARY KEY,
    as_of_date   DATE NOT NULL UNIQUE,
    equity       NUMERIC(18,6) NOT NULL,
    profit       NUMERIC(18,6),
    profit_pct   NUMERIC(9,6),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(2)

    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()
        print("OK: ensured table public.daily_pnl exists.")

if __name__ == "__main__":
    main()
