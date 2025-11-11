# tools/reset_signal.py
import os, sys, psycopg2

sid = os.getenv("SIGNAL_ID") or (len(sys.argv) > 1 and sys.argv[1])
if not sid:
    raise SystemExit("Provide SIGNAL_ID env or argv (e.g., SIGNAL_ID=12)")
sid = int(sid)
force = os.getenv("FORCE", "0") == "1"

db_url = os.environ["DB_URL"]
with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    # find symbol/side of this row
    cur.execute("SELECT symbol, side FROM signals WHERE id=%s;", (sid,))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"Signal {sid} not found")
    symbol, side = row

    # check if another pending exists for the same lock key
    cur.execute("""
        SELECT id FROM signals
         WHERE symbol=%s AND side=%s
           AND processed_at IS NULL
           AND status='pending'
           AND id <> %s
         LIMIT 1;
    """, (symbol, side, sid))
    other = cur.fetchone()

    if other and not force:
        other_id = other[0]
        raise SystemExit(
            f"Another pending exists for {symbol}/{side}: id={other_id}. "
            f"Re-run with FORCE=1 to supersede it."
        )

    if other and force:
        # mark the other pending as skipped, so the unique lock frees up
        cur.execute("""
            UPDATE signals
               SET status='skipped',
                   processed_at=NOW(),
                   status_reason='superseded by reset of id %s'
             WHERE id=%s;
        """, (sid, other[0]))

    # finally reset the target row to pending
    cur.execute("""
        UPDATE signals
           SET status='pending',
               processed_at=NULL,
               error=NULL,
               order_id=NULL,
               client_order_id=NULL,
               exec_order_id=NULL
         WHERE id=%s;
    """, (sid,))
    conn.commit()

print(f"reset {sid} → pending  (symbol={symbol} side={side} force={force})")
