# tools/init_db.py
from __future__ import annotations
import os, sys, logging
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("init_db")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

CORE_DDL = [
    # signals table (simple superset; safe if already exists)
    """
    CREATE TABLE IF NOT EXISTS signals (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        symbol TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('buy','sell')),
        strength NUMERIC,
        source TEXT DEFAULT 'rule',
        portfolio_id TEXT NULL
    );
    """,
    # daily_pnl (coarse; migrations may add columns later)
    """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        id SERIAL PRIMARY KEY,
        day DATE UNIQUE NOT NULL,
        equity NUMERIC,
        realized NUMERIC DEFAULT 0,
        unrealized NUMERIC DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """
]

def main() -> None:
    log.info("INIT_DB | connecting %s", DB_URL)
    with psycopg2.connect(DB_URL) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for stmt in CORE_DDL:
                cur.execute(stmt)
    log.info("INIT_DB | done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("INIT_DB failed: %s", e)
        sys.exit(1)
