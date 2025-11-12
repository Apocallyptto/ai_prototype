#!/usr/bin/env python3
"""
Sync local 'signals' table with Alpaca orders.

- Pulls OPEN + CLOSED orders (within LOOKBACK_DAYS).
- Matches signals by order_id OR client_order_id.
- Maps broker statuses to local statuses.
- Optionally marks 'submitted' older than STALE_MINUTES as 'skipped'.
- If the signals table has filled_qty / avg_fill_price columns, they are updated for filled orders.

Env:
  DB_URL (required)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  ALPACA_PAPER (default "1")
  LOOKBACK_DAYS (default "3")
  STALE_MINUTES (default "0" = disabled)
"""

import os
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

def main():
    paper = os.getenv("ALPACA_PAPER", "1") != "0"
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
    stale_minutes = int(os.getenv("STALE_MINUTES", "0"))

    tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=paper)

    # Fetch both OPEN and CLOSED orders
    open_orders   = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN,   limit=500))
    closed_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500))

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    def recent(o):
        try:
            ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
            if ts is None:
                return True
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= cutoff
        except Exception:
            return True

    # Index orders by id and client_order_id
    orders_by_id = {}
    orders_by_coid = {}
    for o in list(open_orders) + list(closed_orders):
        if not recent(o):
            continue
        oid = _to_str(getattr(o, "id", None))
        coid = _to_str(getattr(o, "client_order_id", None))
        if oid:
            orders_by_id[oid] = o
        if coid:
            orders_by_coid[coid] = o

    # Map Alpaca status -> (local_status, reason)
    status_map = {
        "new":                 None,
        "accepted":            None,
        "partially_filled":    None,
        "filled":              ("filled",  "order filled"),
        "canceled":            ("skipped", "broker canceled"),
        "expired":             ("skipped", "broker expired"),
        "rejected":            ("error",   "broker rejected"),
        "replaced":            ("error",   "broker replaced"),
    }

    db_url = os.environ["DB_URL"]
    checked = 0
    updated = 0
    stale_updated = 0

    with psycopg2.connect(db_url) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            # Detect optional columns (filled_qty / avg_fill_price)
            cur.execute("""
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_name='signals'
                   AND column_name IN ('filled_qty','avg_fill_price');
            """)
            cols = {r[0] for r in cur.fetchall()}
            has_filled_qty = 'filled_qty' in cols
            has_avg_fill_price = 'avg_fill_price' in cols

            # Load all submitted rows
            cur.execute("""
                SELECT id, symbol, side, order_id, client_order_id, status, processed_at, created_at
                  FROM signals
                 WHERE status='submitted'
              ORDER BY id DESC;
            """)
            rows = cur.fetchall()
            checked = len(rows)

            for sid, symbol, side, oid, coid, db_status, processed_at, created_at in rows:
                o = None
                if oid:
                    o = orders_by_id.get(str(oid))
                if (o is None) and coid:
                    o = orders_by_coid.get(str(coid))

                new_status = None
                reason = None
                set_fill_fields = {}

                if o is not None:
                    broker_status = str(getattr(o, "status", "")).lower()
                    mapped = status_map.get(broker_status)
                    if mapped:
                        new_status, reason = mapped
                        if new_status == "filled":
                            # best-effort pull of fills
                            fq = _to_float_or_none(getattr(o, "filled_qty", None))
                            fp = _to_float_or_none(getattr(o, "filled_avg_price", None))
                            if has_filled_qty and fq is not None:
                                set_fill_fields["filled_qty"] = fq
                            if has_avg_fill_price and fp is not None:
                                set_fill_fields["avg_fill_price"] = fp

                # If still not decided and it's old enough, mark as stale
                if not new_status and stale_minutes > 0 and created_at is not None:
                    now = datetime.now(timezone.utc)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    age_min = (now - created_at).total_seconds() / 60.0
                    if age_min >= stale_minutes:
                        new_status = "skipped"
                        reason = f"stale submitted > {stale_minutes}m"
                        stale_updated += 1

                if new_status:
                    if set_fill_fields:
                        # Update including fill fields if available
                        set_clauses = ["status=%s", "processed_at=NOW()", "status_reason=%s"]
                        params = [new_status, reason]
                        if has_filled_qty and "filled_qty" in set_fill_fields:
                            set_clauses.append("filled_qty=%s")
                            params.append(set_fill_fields["filled_qty"])
                        if has_avg_fill_price and "avg_fill_price" in set_fill_fields:
                            set_clauses.append("avg_fill_price=%s")
                            params.append(set_fill_fields["avg_fill_price"])
                        params.append(sid)
                        cur.execute(f"""
                            UPDATE signals
                               SET {", ".join(set_clauses)}
                             WHERE id=%s;
                        """, params)
                    else:
                        # Basic update without fill fields
                        cur.execute("""
                            UPDATE signals
                               SET status=%s,
                                   processed_at=NOW(),
                                   status_reason=%s
                             WHERE id=%s;
                        """, (new_status, reason, sid))
                    updated += 1

            conn.commit()

    print(f"checked {checked} submitted rows, updated {updated} (stale={stale_updated})")

if __name__ == "__main__":
    main()
