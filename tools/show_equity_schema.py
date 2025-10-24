# tools/show_equity_schema.py
from __future__ import annotations
import os, psycopg2

def main():
    dsn = os.environ["DATABASE_URL"]
    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='equity_snapshots'
            ORDER BY ordinal_position;
        """)
        for name, dtype, nullable in cur.fetchall():
            print(f"{name:12s} {dtype:18s} null={nullable}")

if __name__ == "__main__":
    main()
