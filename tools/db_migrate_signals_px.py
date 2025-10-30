# tools/db_migrate_signals_px.py
import os
import logging
import psycopg2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("db_migrate_signals_px")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='signals'
          AND column_name='px'
    ) THEN
        ALTER TABLE public.signals ADD COLUMN px NUMERIC;
    END IF;

    -- optional: if you also want a fast ts index (some repos already have)
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname='public' AND tablename='signals' AND indexname='idx_signals_ts'
    ) THEN
        CREATE INDEX idx_signals_ts ON public.signals(ts);
    END IF;
END$$;
"""

def main():
    log.info("Connecting to %s", DB_URL)
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(SQL)
        conn.commit()
    log.info("OK: signals table now has 'px' column (and index if missing).")

if __name__ == "__main__":
    main()
