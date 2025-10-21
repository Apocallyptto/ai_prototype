# tools/init_db.py
from __future__ import annotations
import os, sys
from datetime import datetime, timezone

# prefer psycopg3; fallback psycopg2
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

DDL = """
CREATE TABLE IF NOT EXISTS signals (
  id           BIGSERIAL PRIMARY KEY,
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL CHECK (side IN ('buy','sell')),
  strength     DOUBLE PRECISION NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  portfolio_id INT NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_created ON signals (symbol, created_at DESC);
"""

SAMPLE = """
INSERT INTO signals (symbol, side, strength, created_at, portfolio_id)
VALUES ('AAPL','buy',0.65, NOW(), 1),
       ('MSFT','buy',0.58, NOW(), 1)
RETURNING id, symbol, side, strength, created_at, portfolio_id;
"""

def main():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv("PGHOST","localhost")
        user = os.getenv("PGUSER","postgres")
        pw   = os.getenv("PGPASSWORD","postgres")
        db   = os.getenv("PGDATABASE","ai_prototype")
        port = os.getenv("PGPORT","5432")
        if HAVE3:
            conn = psycopg.connect(host=host, user=user, password=pw, dbname=db, port=port, autocommit=True)
        else:
            conn = psycopg2.connect(host=host, user=user, password=pw, dbname=db, port=port)
            conn.autocommit = True
    else:
        if HAVE3:
            conn = psycopg.connect(dsn, autocommit=True)
        else:
            conn = psycopg2.connect(dsn)
            conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(DDL)
        print("Ensured table 'signals'.")
        cur.execute(SAMPLE)
        rows = cur.fetchall()
        print("Inserted sample signals:")
        for r in rows:
            print(r)

    conn.close()
    print("OK")

if __name__ == "__main__":
    main()
