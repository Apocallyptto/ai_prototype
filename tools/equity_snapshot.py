# tools/equity_snapshot.py
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
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=20)
    r.raise_for_status()
    return float(r.json()["equity"])

def upsert_equity(dsn: str, as_of: date, equity: float) -> None:
    with psycopg2.connect(dsn=dsn) as conn, conn.cursor() as cur:
        # Baseline = first row of the day; subsequent updates compute PnL vs baseline.
        cur.execute("""
            SELECT equity
            FROM public.equity_snapshots
            WHERE as_of_date = %s
            LIMIT 1;
        """, (as_of,))
        row = cur.fetchone()
        if row is None:
            cur.execute("""
                INSERT INTO public.equity_snapshots(as_of_date, equity, profit, profit_pct)
                VALUES (%s, %s, 0, 0)
                ON CONFLICT (as_of_date) DO UPDATE
                SET equity = EXCLUDED.equity;
            """, (as_of, equity))
            conn.commit()
            print(f"{as_of} equity={equity:.2f} profit=0.00 (first of day)")
        else:
            baseline = float(row[0])
            profit = equity - baseline
            profit_pct = (profit / baseline) if baseline else 0.0
            cur.execute("""
                UPDATE public.equity_snapshots
                SET equity=%s, profit=%s, profit_pct=%s
                WHERE as_of_date=%s;
            """, (equity, profit, profit_pct, as_of))
            conn.commit()
            print(f"{as_of} equity={equity:.2f} profit={profit:.2f} profit_pct={profit_pct*100:.3f}% (updated)")

def main():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    today = datetime.now(timezone.utc).date()
    eq = get_equity()
    upsert_equity(dsn, today, eq)

if __name__ == "__main__":
    main()
