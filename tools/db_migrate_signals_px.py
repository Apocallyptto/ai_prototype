# tools/db_migrate_signals_px.py
import psycopg2, os, logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("db_migrate_signals_px")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

def main():
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='signals' AND column_name='px'
                ) THEN
                    ALTER TABLE public.signals ADD COLUMN px NUMERIC;
                END IF;
            END$$;
        """)
        conn.commit()
    log.info("OK: signals table now has 'px' column.")

if __name__ == "__main__":
    main()
