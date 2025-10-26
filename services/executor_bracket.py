# services/executor_bracket.py
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from services.bracket_helper import submit_bracket

LOG = logging.getLogger("executor_bracket")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s %(name)s | %(message)s"))
if not LOG.handlers:
    LOG.addHandler(ch)

DATABASE_URL = os.getenv("DATABASE_URL")

EXECUTOR_SIGNAL_WINDOW_MIN = int(os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN", "20"))
DEFAULT_MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _fetch_recent_signals(
    since_min: int,
    min_strength: float,
    portfolio_id: Optional[int] = None,
) -> List[Dict]:
    """
    Return latest signal per symbol within window, >= min_strength.
    """
    window_since = _now() - timedelta(minutes=since_min)
    sql = """
    with ranked as (
        select s.*
             , row_number() over (partition by s.symbol order by s.created_at desc) as rn
        from public.signals s
        where s.created_at >= %s
          and s.strength >= %s
          and (%s is null or s.portfolio_id = %s)
    )
    select symbol, side, strength, created_at, source
    from ranked
    where rn = 1
    order by created_at desc
    """
    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (window_since, float(min_strength), portfolio_id, portfolio_id))
        rows = cur.fetchall()
    # Keep only symbols we care about
    rows = [r for r in rows if r["symbol"].upper() in SYMBOLS]
    return rows

def _place_from_rows(rows: List[Dict]) -> int:
    placed = 0
    for r in rows:
        sym = r["symbol"].upper()
        side = r["side"].lower()
        strength = float(r.get("strength", 0.0))
        try:
            resp = submit_bracket(
                symbol=sym,
                side=side,
                qty=None,                 # let dynamic sizing + strength decide (if enabled)
                time_in_force="day",
                order_type="market",      # auto-limit if market closed
                client_id=None,
                strength=strength,
            )
            oid = resp.get("id") or resp.get("order", {}).get("id")
            LOG.info("Placed %s %s (strength=%.2f) id=%s source=%s at=%s",
                     sym, side, strength, oid, r.get("source", "unknown"), r.get("created_at"))
            placed += 1
        except Exception as e:
            LOG.error("%s %s: submit failed -> %s", sym, side, str(e))
    return placed

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Place bracket orders from recent signals.")
    ap.add_argument("--since-min", type=int, default=EXECUTOR_SIGNAL_WINDOW_MIN, help="Window (minutes) to read signals")
    ap.add_argument("--min-strength", type=float, default=DEFAULT_MIN_STRENGTH, help="Minimum strength to consider")
    ap.add_argument("--portfolio-id", type=int, default=None, help="Optional portfolio filter")
    ap.add_argument("--max-positions", type=int, default=None, help="(reserved) Max positions")
    args = ap.parse_args()

    LOG.info(
        "symbols=%s min_strength=%.2f window_min=%d portfolio_id=%s",
        SYMBOLS, args.min_strength, args.since_min, "ANY" if args.portfolio_id is None else args.portfolio_id,
    )

    rows = _fetch_recent_signals(args.since_min, args.min_strength, args.portfolio_id)
    if not rows:
        LOG.info("No signals within window.")
        return

    placed = _place_from_rows(rows)
    LOG.info("Done. Placed %d bracket order(s).", placed)

if __name__ == "__main__":
    main()
