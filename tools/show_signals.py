# tools/show_signals.py
from __future__ import annotations
import os, sys
import psycopg2

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    limit = int(os.environ.get("LIMIT", "12"))
    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, side, strength, created_at
            FROM public.signals
            ORDER BY created_at DESC
            LIMIT %s;
        """, (limit,))
        rows = cur.fetchall()
        if not rows:
            print("No signals found.")
            return
        for r in rows:
            sym, side, strength, ts = r
            print(f"{ts.isoformat()}  {sym:5s}  {side:4s}  {strength:.2f}")

if __name__ == "__main__":
    main()
