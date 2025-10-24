# tools/show_recent_signals.py
import os
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
WINDOW_MIN = int(os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN", "10"))
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")

def _utcnow():
    return datetime.now(timezone.utc)

def main():
    since_dt = _utcnow() - timedelta(minutes=WINDOW_MIN)
    and_portfolio = "AND portfolio_id = %s" if PORTFOLIO_ID else ""
    params = [SYMBOLS, MIN_STRENGTH, since_dt]
    if PORTFOLIO_ID:
        params.append(int(PORTFOLIO_ID))

    sql = f"""
        SELECT DISTINCT ON (symbol)
            symbol, side, strength, created_at, portfolio_id, source
        FROM public.signals
        WHERE symbol = ANY(%s)
          AND strength >= %s
          AND created_at >= %s
          {and_portfolio}
        ORDER BY symbol, created_at DESC
    """

    with psycopg2.connect(DATABASE_URL) as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        print(f"No rows within {WINDOW_MIN}m window (min_strength={MIN_STRENGTH}).")
        return

    rows.sort(key=lambda r: r["created_at"], reverse=True)
    for r in rows:
        print(f"{r['symbol']:5s} {r['side']:4s} strength={r['strength']:.2f} at={r['created_at']} src={r.get('source') or 'unknown'}")

if __name__ == "__main__":
    main()
