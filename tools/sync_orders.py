#!/usr/bin/env python3
"""
Sync local 'signals' table with Alpaca orders.

Schema expected:
  signals(..., created_at, processed_status, processed_note, processed_at, alpaca_order_id, ...)

What it does:
- Repeats every SYNC_POLL seconds (default 30).
- Finds signals where processed_status='submitted'.
- Looks up the Alpaca order by alpaca_order_id (exact).
- If Alpaca order is terminal (filled/canceled/expired/rejected/replaced),
  updates processed_status/processed_note/processed_at.

Env:
  DB_URL (optional; default postgresql://postgres:postgres@postgres:5432/trader)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  ALPACA_PAPER (default "1")
  LOOKBACK_DAYS (default "30")   # just for order fetching, not for DB selection
  SYNC_POLL (default "30")
"""

import os
import sys
import time
import psycopg2
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def _normalize_db_url() -> str:
    url = (os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "").strip()
    if not url:
        # default local in compose
        return "postgresql://postgres:postgres@postgres:5432/trader"

    # Common alias in some setups
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    # If URL requests psycopg (v3) but it's not installed, fall back to psycopg2.
    # This prevents crashes when DB_URL is set to postgresql+psycopg:// but only psycopg2-binary is installed.
    if url.startswith("postgresql+psycopg://"):
        try:
            import psycopg  # type: ignore  # noqa: F401
        except Exception:
            url = "postgresql+psycopg2://" + url.split("postgresql+psycopg://", 1)[1]

    return url


def _sval(x) -> str:
    """
    Alpaca SDK often returns enums for fields like status/order_class.
    This converts them to their string 'value' (e.g. OrderStatus.FILLED -> 'filled').
    """
    if x is None:
        return ""
    v = getattr(x, "value", None)
    if v is not None:
        return str(v)
    return str(x)


def _load_columns(cur) -> set[str]:
    cur.execute("""
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name='signals';
    """)
    return {r[0] for r in cur.fetchall()}


def _alpaca_to_local(status: str) -> tuple[str | None, str | None]:
    s = (status or "").strip().lower()

    # Sometimes enum->str becomes like "orderstatus.filled" if not normalized.
    # Defensive fallback:
    if "." in s:
        s = s.split(".")[-1]

    if s in ("new", "accepted", "partially_filled", "held"):
        return (None, None)  # keep submitted
    if s == "filled":
        return ("filled", "order filled")
    if s in ("canceled", "expired"):
        return ("skipped", f"broker {s}")
    if s in ("rejected", "replaced"):
        return ("error", f"broker {s}")
    return (None, None)


def main_loop():
    paper = os.getenv("ALPACA_PAPER", "1") != "0"
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "30"))
    poll = int(os.getenv("SYNC_POLL", "30"))

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET")
        sys.exit(2)

    tc = TradingClient(api_key, api_secret, paper=paper)
    db_url = _normalize_db_url()

    print(f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | paper={paper}")

    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
            closed_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500))
            all_orders = list(open_orders) + list(closed_orders)

            # Index by id (include everything we fetched)
            orders_by_id: dict[str, object] = {}
            for o in all_orders:
                oid = _sval(getattr(o, "id", None)).strip()
                if not oid:
                    continue

                # Optional filter: keep recent-ish only (helps memory), but don't break things
                sub_at = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
                if sub_at is not None:
                    try:
                        if sub_at.tzinfo is None:
                            sub_at = sub_at.replace(tzinfo=timezone.utc)
                        sub_at = sub_at.astimezone(timezone.utc)
                        if sub_at < cutoff:
                            # still keep it? let's keep it, because your DB contains Dec 30 too.
                            pass
                    except Exception:
                        pass

                orders_by_id[oid] = o

            checked = 0
            updated = 0
            not_found = 0

            with psycopg2.connect(db_url) as conn:
                conn.autocommit = False
                with conn.cursor() as cur:
                    cols = _load_columns(cur)
                    required = {"id", "created_at", "symbol", "side", "processed_status", "alpaca_order_id", "processed_at", "processed_note"}
                    if not required.issubset(cols):
                        print(f"ERROR: signals schema missing columns: {sorted(list(required - cols))}")
                        conn.rollback()
                        time.sleep(poll)
                        continue

                    cur.execute("""
                        SELECT id, created_at, symbol, side, processed_status, alpaca_order_id, processed_note
                          FROM signals
                         WHERE processed_status='submitted'
                      ORDER BY id DESC;
                    """)
                    rows = cur.fetchall()
                    checked = len(rows)

                    for sid, created_at, sym, side, st, oid, note in rows:
                        oid = (oid or "").strip()
                        if not oid:
                            not_found += 1
                            continue

                        o = orders_by_id.get(oid)
                        if o is None:
                            # Not in our fetched window. Try direct fetch by id via SDK? (SDK doesn't have get_order_by_id)
                            # We'll just mark as not_found for now.
                            not_found += 1
                            continue

                        alp_status = _sval(getattr(o, "status", None)).strip().lower()
                        new_status, reason = _alpaca_to_local(alp_status)
                        if not new_status:
                            # still open / not terminal
                            continue

                        cur.execute("""
                            UPDATE signals
                               SET processed_status=%s,
                                   processed_at=NOW(),
                                   processed_note=%s
                             WHERE id=%s;
                        """, (new_status, reason, sid))
                        updated += 1

                    conn.commit()

            print(f"checked={checked} submitted | updated={updated} | not_found_in_cache={not_found}")

        except Exception as e:
            print("sync_orders ERROR:", repr(e))

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
