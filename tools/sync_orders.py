#!/usr/bin/env python3
"""
Sync local 'signals' table with Alpaca orders.

Supports your schema:
  signals(..., created_at, processed_status, processed_note, processed_at, alpaca_order_id, ...)

What it does:
- Repeats every SYNC_POLL seconds (default 30).
- Finds signals where processed_status='submitted'.
- Tries to locate matching Alpaca order:
    1) by alpaca_order_id (exact)
    2) fallback by (symbol, side, created_at within MATCH_WINDOW_SEC), ignoring order_class='oco'
       and avoids linking already-linked Alpaca ids.
- Updates processed_status/processed_note/processed_at when Alpaca order is closed (filled/canceled/expired/rejected/replaced).
- If a signal is older than STALE_MINUTES and still not resolved -> marks skipped (optional).

Env:
  DB_URL (optional; default postgresql://postgres:postgres@postgres:5432/trader)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  ALPACA_PAPER (default "1")
  LOOKBACK_DAYS (default "3")
  SYNC_POLL (default "30")
  MATCH_WINDOW_SEC (default "300")   # 5 min
  STALE_MINUTES (default "0")        # disabled by default
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
    raw_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@postgres:5432/trader"
    if raw_url.startswith("postgresql+psycopg2://"):
        raw_url = raw_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    return raw_url


def _to_dt_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # Alpaca gives ISO8601 like 2026-01-07T15:55:17.899228817Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _load_columns(cur) -> set[str]:
    cur.execute("""
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name='signals';
    """)
    return {r[0] for r in cur.fetchall()}


def _is_entry_order(order_obj) -> bool:
    # Ignore exit OCO orders when fallback matching
    oc = getattr(order_obj, "order_class", None)
    if oc is None:
        return True
    oc = str(oc).strip().lower()
    return oc == ""


def _alpaca_to_local(status: str) -> tuple[str | None, str | None]:
    s = (status or "").lower()
    if s in ("new", "accepted", "partially_filled"):
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
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
    poll = int(os.getenv("SYNC_POLL", "30"))
    match_window_sec = int(os.getenv("MATCH_WINDOW_SEC", "300"))
    stale_minutes = int(os.getenv("STALE_MINUTES", "0"))

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET")
        sys.exit(2)

    tc = TradingClient(api_key, api_secret, paper=paper)
    db_url = _normalize_db_url()

    print(f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | match_window={match_window_sec}s | stale_minutes={stale_minutes}")

    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            # Fetch OPEN + CLOSED
            open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
            closed_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500))

            all_orders = list(open_orders) + list(closed_orders)

            # Index orders
            orders_by_id: dict[str, object] = {}
            orders_by_sym_side: dict[tuple[str, str], list[object]] = {}

            for o in all_orders:
                oid = str(getattr(o, "id", "") or "")
                if oid:
                    orders_by_id[oid] = o

                # build list for fallback matching
                if not _is_entry_order(o):
                    continue

                sym = str(getattr(o, "symbol", "") or "")
                side = str(getattr(o, "side", "") or "")
                sub_at = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
                if sub_at is not None:
                    try:
                        if sub_at.tzinfo is None:
                            sub_at = sub_at.replace(tzinfo=timezone.utc)
                        sub_at = sub_at.astimezone(timezone.utc)
                    except Exception:
                        sub_at = None

                # filter recent
                if sub_at is not None and sub_at < cutoff:
                    continue

                if sym and side:
                    orders_by_sym_side.setdefault((sym, side), []).append(o)

            # sort fallback lists by submitted_at
            for k, lst in orders_by_sym_side.items():
                def keyfn(o):
                    dt = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
                    try:
                        if dt and dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(timezone.utc) if dt else datetime.min.replace(tzinfo=timezone.utc)
                    except Exception:
                        return datetime.min.replace(tzinfo=timezone.utc)
                lst.sort(key=keyfn, reverse=True)

            checked = 0
            updated = 0
            linked = 0
            not_found = 0
            stale_updated = 0

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

                    # already linked alpaca ids, so fallback matching doesn't reuse same order
                    cur.execute("""
                        SELECT alpaca_order_id
                          FROM signals
                         WHERE alpaca_order_id IS NOT NULL AND alpaca_order_id <> '';
                    """)
                    used_ids = {r[0] for r in cur.fetchall()}

                    cur.execute("""
                        SELECT id, created_at, symbol, side, processed_status, alpaca_order_id, processed_note
                          FROM signals
                         WHERE processed_status='submitted'
                      ORDER BY id DESC;
                    """)
                    rows = cur.fetchall()
                    checked = len(rows)

                    now_utc = datetime.now(timezone.utc)

                    for sid, created_at, sym, side, st, oid, note in rows:
                        # normalize created_at
                        if created_at is not None and created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        if created_at is not None:
                            created_at = created_at.astimezone(timezone.utc)

                        o = None

                        # 1) exact by alpaca_order_id
                        if oid:
                            o = orders_by_id.get(str(oid))

                        # 2) fallback by (symbol, side, time window) if no oid or not found
                        if o is None and sym and side and created_at is not None:
                            candidates = orders_by_sym_side.get((sym, side), [])
                            best = None
                            best_dt = None
                            for cand in candidates:
                                cid = str(getattr(cand, "id", "") or "")
                                if not cid or cid in used_ids:
                                    continue
                                sub_at = getattr(cand, "submitted_at", None) or getattr(cand, "created_at", None)
                                if sub_at is None:
                                    continue
                                if sub_at.tzinfo is None:
                                    sub_at = sub_at.replace(tzinfo=timezone.utc)
                                sub_at = sub_at.astimezone(timezone.utc)
                                delta = abs((sub_at - created_at).total_seconds())
                                if delta <= match_window_sec:
                                    if best is None or delta < best_dt:
                                        best = cand
                                        best_dt = delta
                            if best is not None:
                                o = best
                                # link alpaca_order_id in DB
                                new_oid = str(getattr(o, "id", "") or "")
                                if new_oid:
                                    cur.execute("""
                                        UPDATE signals
                                           SET alpaca_order_id=%s,
                                               processed_note=COALESCE(processed_note,'') || %s
                                         WHERE id=%s;
                                    """, (new_oid, " | linked_by_fallback", sid))
                                    used_ids.add(new_oid)
                                    linked += 1

                        if o is None:
                            not_found += 1
                            # optionally stale skip
                            if stale_minutes > 0 and created_at is not None:
                                age_min = (now_utc - created_at).total_seconds() / 60.0
                                if age_min >= stale_minutes:
                                    cur.execute("""
                                        UPDATE signals
                                           SET processed_status='skipped',
                                               processed_at=NOW(),
                                               processed_note='stale submitted > %s minutes'
                                         WHERE id=%s;
                                    """, (stale_minutes, sid))
                                    updated += 1
                                    stale_updated += 1
                            continue

                        alp_status = str(getattr(o, "status", "") or "").lower()
                        new_status, reason = _alpaca_to_local(alp_status)
                        if not new_status:
                            continue  # still open, keep submitted

                        cur.execute("""
                            UPDATE signals
                               SET processed_status=%s,
                                   processed_at=NOW(),
                                   processed_note=%s
                             WHERE id=%s;
                        """, (new_status, reason, sid))
                        updated += 1

                    conn.commit()

            print(f"checked={checked} submitted | updated={updated} (stale={stale_updated}) | linked_by_fallback={linked} | not_found_in_alpaca={not_found}")

        except Exception as e:
            print("sync_orders ERROR:", repr(e))

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
