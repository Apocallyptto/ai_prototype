# tools/db_migrate_pnl.py
from __future__ import annotations
import os, sys
import psycopg2

DDL_CREATE = """
CREATE TABLE IF NOT EXISTS public.daily_pnl (
    as_of_date  DATE PRIMARY KEY,
    equity      NUMERIC(18,6) NOT NULL,
    profit      NUMERIC(18,6),
    profit_pct  NUMERIC(9,6),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

DDL_ALTER = [
    # Add missing columns safely
    "ALTER TABLE public.daily_pnl ADD COLUMN IF NOT EXISTS as_of_date DATE;",
    "ALTER TABLE public.daily_pnl ADD COLUMN IF NOT EXISTS equity NUMERIC(18,6);",
    "ALTER TABLE public.daily_pnl ADD COLUMN IF NOT EXISTS profit NUMERIC(18,6);",
    "ALTER TABLE public.daily_pnl ADD COLUMN IF NOT EXISTS profit_pct NUMERIC(9,6);",
    "ALTER TABLE public.daily_pnl ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",

    # Ensure NOT NULLs where we rely on them
    "ALTER TABLE public.daily_pnl ALTER COLUMN as_of_date SET NOT NULL;",
    "ALTER TABLE public.daily_pnl ALTER COLUMN equity SET NOT NULL;",

    # Make as_of_date unique (use DO block because IF NOT EXISTS is not supported for constraints)
    """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'daily_pnl_as_of_date_key'
      ) THEN
        BEGIN
          ALTER TABLE public.daily_pnl
          ADD CONSTRAINT daily_pnl_as_of_date_key UNIQUE (as_of_date);
        EXCEPTION WHEN duplicate_table THEN
          -- ignore race
          NULL;
        END;
      END IF;
    END$$;
    """,
]

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(2)

    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        # Create if missing (with full schema)
        cur.execute(DDL_CREATE)
        # Then bring any existing table up to spec
        for stmt in DDL_ALTER:
            cur.execute(stmt)
        conn.commit()
        print("OK: public.daily_pnl is migrated/normalized.")

if __name__ == "__main__":
    main()
