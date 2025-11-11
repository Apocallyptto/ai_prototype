# tools/unlock_symbol.py (enhanced)
import os, sys, psycopg2
from datetime import timedelta

symbol = os.getenv("SYMBOL") or (len(sys.argv) > 1 and sys.argv[1])
side   = os.getenv("SIDE")   or (len(sys.argv) > 2 and sys.argv[2])
age_min = int(os.getenv("SUBMITTED_AGE_MINUTES", "15"))

if not (symbol and side):
    raise SystemExit("Usage: SYMBOL=AAPL SIDE=buy [SUBMITTED_AGE_MINUTES=15]")

db_url = os.environ["DB_URL"]
with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    # 1) Prefer unlocking true 'pending' rows (classic lock)
    cur.execute("""
        UPDATE signals
           SET status='skipped',
               processed_at=NOW(),
               status_reason='manual unlock'
         WHERE symbol=%s AND side=%s
           AND processed_at IS NULL AND status='pending';
    """, (symbol, side))
    unlocked_pending = cur.rowcount

    # 2) If nothing was pending, optionally mark stale 'submitted' rows as skipped
    unlocked_submitted = 0
    if unlocked_pending == 0 and age_min > 0:
        cur.execute(f"""
            UPDATE signals
               SET status='skipped',
                   status_reason=concat('manual unlock (stale submitted > {age_min}m)'),
                   processed_at=COALESCE(processed_at, NOW())
             WHERE symbol=%s AND side=%s
               AND status='submitted'
               AND processed_at <= NOW() - INTERVAL '{age_min} minutes';
        """, (symbol, side))
        unlocked_submitted = cur.rowcount

    conn.commit()

print(f"unlocked pending={unlocked_pending}, stale_submitted={unlocked_submitted} for {symbol}/{side}")
