# tools/pnl_snapshot.py
from __future__ import annotations
import os, sys, requests
from datetime import datetime, timezone, date
import psycopg2

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def get_equity() -> float:
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=15)
    r.raise_for_status()
    return float(r.json()["equity"])

def upsert_daily_pnl(dsn: str, as_of: date, equity: float) -> None:
    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT equity
            FROM public.daily_pnl
            WHERE as_of_date = %s
            ORDER BY id ASC
            LIMIT 1;
        """, (as_of,))
        row = cur.fetchone()
        if row is None:
            profit = 0.0
            profit_pct = 0.0
            cur.execute("""
                INSERT INTO public.daily_pnl(as_of_date, equity, profit, profit_pct)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (as_of_date) DO UPDATE
                SET equity = EXCLUDED.equity,
                    profit = EXCLUDED.profit,
                    profit_pct = EXCLUDED.profit_pct;
            """, (as_of, equity, profit, profit_pct))
            conn.commit()
            print(f"{as_of}  equity={equity:.2f}  profit={profit:.2f} (first snapshot)")
        else:
            first_equity = float(row[0])
            profit = equity - first_equity
            profit_pct = (profit / first_equity) if first_equity != 0 else 0.0
            cur.execute("""
                UPDATE public.daily_pnl
                SET equity = %s,
                    profit = %s,
                    profit_pct = %s
                WHERE as_of_date = %s;
            """, (equity, profit, profit_pct, as_of))
            conn.commit()
            print(f"{as_of}  equity={equity:.2f}  profit={profit:.2f}  profit_pct={profit_pct*100:.3f}% (updated)")

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(2)
    equity = get_equity()
    today = datetime.now(timezone.utc).date()
    upsert_daily_pnl(dsn, today, equity)

if __name__ == "__main__":
    main()
