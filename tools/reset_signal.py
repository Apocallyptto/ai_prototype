# tools/reset_signal.py
import os, psycopg2, sys
sid = int(os.getenv("SIGNAL_ID") or (len(sys.argv) > 1 and sys.argv[1]) or 0)
if not sid:
    raise SystemExit("Provide SIGNAL_ID env or argv")
db_url = os.environ["DB_URL"]
with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
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
print("reset", sid, "to pending")
