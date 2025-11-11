# tools/unlock_symbol.py
import os, sys, psycopg2

symbol = os.getenv("SYMBOL") or (len(sys.argv) > 1 and sys.argv[1])
side   = os.getenv("SIDE")   or (len(sys.argv) > 2 and sys.argv[2])
if not (symbol and side):
    raise SystemExit("Usage: SYMBOL=AAPL SIDE=buy  or  python unlock_symbol.py AAPL buy")

db_url = os.environ["DB_URL"]
with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    cur.execute("""
        UPDATE signals
           SET status='skipped',
               processed_at=NOW(),
               status_reason='manual unlock'
         WHERE symbol=%s AND side=%s
           AND processed_at IS NULL AND status='pending';
    """, (symbol, side))
    n = cur.rowcount
    conn.commit()
print(f"unlocked {n} row(s) for {symbol}/{side}")
