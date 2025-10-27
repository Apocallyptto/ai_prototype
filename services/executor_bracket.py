# services/executor_bracket.py
from __future__ import annotations

import os, sys, argparse, logging
from datetime import datetime, timedelta, timezone
import psycopg2

from services.bracket_helper import submit_bracket

log = logging.getLogger("executor_bracket")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DB_DSN = os.getenv("DATABASE_URL")

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))

def fetch_recent_signals(window_min: int) -> list[tuple[str,str,float,datetime]]:
    """Return most recent signal per symbol within window."""
    since = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    sql = """
        SELECT DISTINCT ON (symbol)
               symbol, side, strength, created_at
        FROM public.signals
        WHERE created_at >= %s
          AND strength >= %s
          AND symbol = ANY(%s)
        ORDER BY symbol, created_at DESC;
    """
    with psycopg2.connect(DB_DSN) as c, c.cursor() as cur:
        cur.execute(sql, (since, MIN_STRENGTH, SYMBOLS))
        rows = cur.fetchall()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-min", type=int, default=int(os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN","20")))
    ap.add_argument("--min-strength", type=float, default=MIN_STRENGTH)
    ap.add_argument("--portfolio-id", type=str, default=None)
    ap.add_argument("--max-positions", type=int, default=None)
    args = ap.parse_args()

    log.info("symbols=%s min_strength=%.2f window_min=%d portfolio_id=%s",
             SYMBOLS, args.min_strength, args.since_min, args.portfolio_id or "ANY")

    rows = fetch_recent_signals(args.since_min)
    if not rows:
        log.info("No signals within window.")
        return

    placed = 0
    for sym, side, strength, created_at in rows:
        try:
            resp = submit_bracket(sym, side, qty=None, strength=float(strength))
            log.info("Placed %s %s (strength=%.2f) id=%s source=%s at=%s",
                     sym, side, strength, resp.get("id"), None, created_at)
            placed += 1
        except Exception as e:
            log.error("%s %s: submit failed -> %s", sym, side, e)
    log.info("Done. Placed %d bracket order(s).", placed)

if __name__ == "__main__":
    main()
