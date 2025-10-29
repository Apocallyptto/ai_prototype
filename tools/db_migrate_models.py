# tools/db_migrate_models.py
import os, psycopg2

def _dsn():
    dsn = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("DB_HOST","postgres")
    user = os.getenv("DB_USER","postgres")
    pw   = os.getenv("DB_PASSWORD","postgres")
    db   = os.getenv("DB_NAME","trader")
    port = os.getenv("DB_PORT","5432")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

DDL = """
CREATE TABLE IF NOT EXISTS public.models_meta (
    id           BIGSERIAL PRIMARY KEY,
    model_type   TEXT NOT NULL,             -- e.g. 'gbc_5m'
    path        TEXT NOT NULL,              -- absolute path inside container, e.g. /app/models/gbc_5m.pkl
    features    TEXT[] NOT NULL DEFAULT '{}',
    symbols     TEXT[] NOT NULL DEFAULT '{}',
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active   BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add missing columns if table existed previously
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='path')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN path TEXT NOT NULL DEFAULT '';
        -- wipe default to avoid empty strings on future inserts
        ALTER TABLE public.models_meta ALTER COLUMN path DROP DEFAULT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='model_type')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN model_type TEXT NOT NULL DEFAULT 'gbc_5m';
        ALTER TABLE public.models_meta ALTER COLUMN model_type DROP DEFAULT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='is_active')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT false;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='features')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN features TEXT[] NOT NULL DEFAULT '{}';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='symbols')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN symbols TEXT[] NOT NULL DEFAULT '{}';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='params')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN params JSONB NOT NULL DEFAULT '{}'::jsonb;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='metrics')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN metrics JSONB NOT NULL DEFAULT '{}'::jsonb;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='models_meta' AND column_name='created_at')
    THEN
        ALTER TABLE public.models_meta ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_models_meta_active ON public.models_meta (is_active, created_at DESC);
"""

def main():
    dsn = _dsn()
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
    print("OK: models_meta migrated.")

if __name__ == "__main__":
    main()
