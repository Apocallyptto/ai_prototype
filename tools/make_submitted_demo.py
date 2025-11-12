#!/usr/bin/env python3
import os, psycopg2, sys
from datetime import timedelta

db_url = os.environ["DB_URL"]
symbol = os.getenv("SYMBOL", "AAPL")
side   = os.getenv("SIDE",   "buy")
age_m  = int(os.getenv("AGE_MINUTES", "25"))  # > STALE_MINUTES to see it skip

row = {
    "symbol": symbol,
    "side": side,
    "strength": 0.95,
    "source": "demo",
    "portfolio_id": "paper-core",
    "status": "submitted",
}

with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    # make sure created_at exists (idempotent)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='signals' AND column_name='created_at'
            ) THEN
                ALTER TABLE signals
                    ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
            END IF;
        END$$;
    """)
    # insert demo row
    cur.execute("""
        INSERT INTO signals(symbol, side, strength, source, portfolio_id, status)
        VALUES (%s,%s,%s,%s,%s,%s)
        RETURNING id, created_at;
    """, (row["symbol"], row["side"], row["strength"], row["source"], row["portfolio_id"], row["status"]))
    sid, created_at = cur.fetchone()

    # age it backwards so sync can treat it as stale
    cur.execute("UPDATE signals SET created_at = NOW() - (%s || ' minutes')::INTERVAL WHERE id=%s;",
                (age_m, sid))
    conn.commit()

print("inserted demo submitted row; to see it:", """
  docker compose exec postgres psql -U postgres -d trader -c "
    SELECT id,symbol,side,status,created_at,processed_at,status_reason
      FROM signals
     WHERE id = (SELECT max(id) FROM signals)
  "
""")
