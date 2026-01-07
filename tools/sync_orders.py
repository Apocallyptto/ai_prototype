#!/usr/bin/env python3
"""
Sync local 'signals' table with Alpaca orders.

Works with TWO possible schemas:

A) Newer schema (your current one):
  signals(..., processed_status, processed_note, processed_at, alpaca_order_id, created_at, ...)

B) Older schema:
  signals(..., status, status_reason, processed_at, order_id, client_order_id, created_at, ...)

What it does:
- Fetches OPEN + CLOSED Alpaca orders (within LOOKBACK_DAYS).
- For local rows with status "submitted", tries to find matching Alpaca order by:
    - alpaca_order_id / order_id (preferred)
    - client_order_id (if column exists)
- Maps Alpaca status -> local status:
    - filled   -> filled
    - canceled/expired -> skipped
    - rejected/replaced -> error
    - new/accepted/partially_filled -> keep submitted (no change)
- Optional: marks stale submitted signals older than STALE_MINUTES as skipped.
- If DB has filled_qty / avg_fill_price columns, updates them for filled orders.

Env:
  DB_URL (required; can be postgresql+psycopg2://... or postgresql://...)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  ALPACA_PAPER (default "1")
  LOOKBACK_DAYS (default "3")
  STALE_MINUTES (default "0" = disabled)
"""

import os
import sys
import psycopg2
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def _to_str(x):
    return str(x) if x is not None else None


def _to_float_or_none(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _normalize_db_url() -> str:
    raw_url = (
        os.getenv("DB_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@postgres:5432/trader"
    )
    if raw_url.startswith("postgresql+psycopg2://"):
        raw_url = raw_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    return raw_url


def _load_signals_columns(cur) -> set[str]:
    cur.execute("""
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name='signals';
    """)
    return {r[0] for r in cur.fetchall()}


def main():
    paper = os.getenv("ALPACA_PAPER", "1") != "0"
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
    stale_minutes = int(os.getenv("STALE_MINUTES", "0"))

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET")
        sys.exit(2)

    tc = TradingClient(api_key, api_secret, paper=paper)

    # Pull OPEN + CLOSED
    open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    closed_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500))

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    def recent(o) -> bool:
        try:
            ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
            if ts is None:
                return True
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= cutoff
        except Exception:
            return True

    orders_by_id: dict[str, object] = {}
    orders_by_coid: dict[str, object] = {}

    for o in list(open_orders) + list(closed_orders):
        if not recent(o):
            continue
        oid = _to_str(getattr(o, "id", None))
        coid = _to_str(getattr(o, "client_order_id", None))
        if oid:
            orders_by_id[oid] = o
        if coid:
            orders_by_coid[coid] = o

    # Alpaca status -> local status, reason
    status_map = {
        "new": None,
        "accepted": None,
        "partially_filled": None,
        "filled": ("filled", "order filled"),
        "canceled": ("skipped", "broker canceled"),
        "expired": ("skipped", "broker expired"),
        "rejected": ("error", "broker rejected"),
        "replaced": ("error", "broker replaced"),
    }

    db_url = _normalize_db_url()

    checked = 0
    updated = 0
    stale_updated = 0
    not_found = 0

    with psycopg2.connect(db_url) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cols = _load_signals_columns(cur)

            # Decide schema
            status_col = "processed_status" if "processed_status" in cols else ("status" if "status" in cols else None)
            note_col = "processed_note" if "processed_note" in cols else ("status_reason" if "status_reason" in cols else None)
            processed_at_col = "processed_at" if "processed_at" in cols else None
            created_at_col = "created_at" if "created_at" in cols else None

            # order id column (your DB uses alpaca_order_id)
            order_id_col = "alpaca_order_id" if "alpaca_order_id" in cols else ("order_id" if "order_id" in cols else None)
            client_order_id_col = "client_order_id" if "client_order_id" in cols else None

            has_filled_qty = "filled_qty" in cols
            has_avg_fill_price = "avg_fill_price" in cols

            if not status_col or not order_id_col:
                print("ERROR: signals table schema not supported.")
                print(f"  missing status_col or order_id_col. detected cols: {sorted(list(cols))[:25]} ...")
                sys.exit(3)

            # Build SELECT dynamically (safe: only from known column names)
            select_parts = [
                "id",
                "symbol",
                "side",
                f"{order_id_col} AS order_id_value",
                (f"{client_order_id_col} AS client_order_id_value" if client_order_id_col else "NULL AS client_order_id_value"),
                f"{status_col} AS status_value",
                (f"{processed_at_col} AS processed_at_value" if processed_at_col else "NULL AS processed_at_value"),
                (f"{created_at_col} AS created_at_value" if created_at_col else "NULL AS created_at_value"),
            ]

            cur.execute(f"""
                SELECT {", ".join(select_parts)}
                  FROM signals
                 WHERE {status_col} = 'submitted'
              ORDER BY id DESC;
            """)

            rows = cur.fetchall()
            checked = len(rows)

            for (sid, symbol, side, oid, coid, db_status, processed_at, created_at) in rows:
                o = None
                if oid:
                    o = orders_by_id.get(str(oid))
                if (o is None) and coid:
                    o = orders_by_coid.get(str(coid))

                new_status = None
                reason = None
                fill_qty = None
                fill_avg = None

                if o is not None:
                    broker_status = str(getattr(o, "status", "")).lower()
                    mapped = status_map.get(broker_status)
                    if mapped:
                        new_status, reason = mapped
                        if new_status == "filled":
                            fill_qty = _to_float_or_none(getattr(o, "filled_qty", None))
                            fill_avg = _to_float_or_none(getattr(o, "filled_avg_price", None))
                else:
                    not_found += 1

                # stale detection (only if still not decided)
                if not new_status and stale_minutes > 0 and created_at is not None:
                    now = datetime.now(timezone.utc)
                    if getattr(created_at, "tzinfo", None) is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    age_min = (now - created_at).total_seconds() / 60.0
                    if age_min >= stale_minutes:
                        new_status = "skipped"
                        reason = f"stale submitted > {stale_minutes}m"
                        stale_updated += 1

                if not new_status:
                    continue  # keep submitted

                # Build UPDATE
                set_clauses = [f"{status_col}=%s"]
                params = [new_status]

                if processed_at_col:
                    set_clauses.append(f"{processed_at_col}=NOW()")

                # Note/reason field (append fill info if present)
                if note_col:
                    note = reason or ""
                    if new_status == "filled":
                        if fill_qty is not None:
                            note += f" | filled_qty={fill_qty}"
                        if fill_avg is not None:
                            note += f" | avg_fill_price={fill_avg}"
                    set_clauses.append(f"{note_col}=%s")
                    params.append(note)

                # Optional fill fields
                if new_status == "filled":
                    if has_filled_qty and fill_qty is not None:
                        set_clauses.append("filled_qty=%s")
                        params.append(fill_qty)
                    if has_avg_fill_price and fill_avg is not None:
                        set_clauses.append("avg_fill_price=%s")
                        params.append(fill_avg)

                params.append(sid)

                cur.execute(f"""
                    UPDATE signals
                       SET {", ".join(set_clauses)}
                     WHERE id=%s;
                """, params)

                updated += 1

            conn.commit()

    print(f"checked={checked} submitted rows | updated={updated} (stale={stale_updated}) | not_found_in_alpaca={not_found}")


if __name__ == "__main__":
    main()
