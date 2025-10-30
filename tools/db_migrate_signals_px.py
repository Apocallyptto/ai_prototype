# tools/db_migrate_signals_px.py
import os, psycopg2, logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("db_migrate_signals_px")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='px'
    ) THEN
        ALTER TABLE public.signals ADD COLUMN px NUMERIC;
    END IF;
END$$;
"""

def main():
    log.info(f"Connecting to {DB_URL}")
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(SQL)
        conn.commit()
    log.info("OK: signals table now has 'px' column.")

if __name__ == "__main__":
    main()
