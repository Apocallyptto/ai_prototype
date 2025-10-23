# tools/show_daily_pnl_schema.py
from __future__ import annotations
import os, psycopg2

def main():
    dsn = os.environ["DATABASE_URL"]
    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='daily_pnl'
            ORDER BY ordinal_position;
        """)
        rows = cur.fetchall()
    print("public.daily_pnl columns:")
    for name, dtype, nullable in rows:
        print(f" - {name:12s} {dtype:18s} null={nullable}")

if __name__ == "__main__":
    main()
