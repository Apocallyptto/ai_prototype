# tools/db_migrate_models.py
import os, psycopg2, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("ERROR: set DB_URL or DATABASE_URL")

    ddl = """
    CREATE TABLE IF NOT EXISTS models_meta (
        id SERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        artifact_path TEXT NOT NULL,
        notes TEXT,
        is_active BOOLEAN NOT NULL DEFAULT FALSE
    );

    -- only one active per model_name
    CREATE UNIQUE INDEX IF NOT EXISTS ux_models_meta_active
      ON models_meta (model_name) WHERE is_active = TRUE;
    """

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    logging.info("OK: ensured table public.models_meta exists.")

if __name__ == "__main__":
    main()
