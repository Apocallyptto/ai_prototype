#!/usr/bin/env python3
"""
Sync local 'signals' with Alpaca orders.
- Looks at both OPEN and CLOSED orders (within LOOKBACK_DAYS).
- Matches by order_id OR client_order_id.
- Optionally marks 'submitted' older than STALE_MINUTES as skipped.
Env:
  DB_URL (required)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  ALPACA_PAPER (default 1)
  LOOKBACK_DAYS (default 3)
  STALE_MINUTES (optional; if set>0, mark submitted older than this as skipped)
"""

import os, psycopg2
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

paper = os.getenv("ALPACA_PAPER", "1") != "0"
lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
stale_minutes = int(os.getenv("STALE_MINUTES", "0"))

tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=paper)

# Pull recent orders: OPEN + CLOSED (filled/canceled/expired/rejected)
req_open   = GetOrdersRequest(status=QueryOrderStatus.OPEN,   limit=500)
req_closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
open_orders   = tc.get_orders(req_open)
closed_orders = tc.get_orders(req_closed)

def to_str(x):
    return str(x) if x is not None else None

# Map by both id and client_order_id for quick lookup
orders_by_id  = {}
orders_by_coid = {}

cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
def recent(o):
    try:
        # alpaca Order has submitted_at or created_at
        ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
        if ts is None:
            return True  # keep if unsure
        # ensure tz-aware to compare
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= cutoff
    except Exception:
        return True

for o in list(open_orders) + list(closed_orders):
    if not recent(o):
        continue
    orders_by_id[to_str(o.id)] = o
    if getattr(o, "client_order_id", None):
        orders_by_coid[to_str(o.client_order_id)] = o

status_map = {
    "new":       None,          # still open
    "accepted":  None,          # still open
    "partially_filled": None,   # still working
    "filled":    ("filled",  "order filled"),
    "canceled":  ("skipped", "broker canceled"),
    "expired":   ("skipped", "broker expired"),
    "rejected":  ("error",   "broker rejected"),
    "replaced":  ("error",   "broker replaced"),
}

db_url = os.environ["DB_URL"]
checked = 0
updated = 0
stale_updated = 0

with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    # 1) Reconcile all 'submitted'
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

        if o is not None:
            broker_status = str(getattr(o, "status", "")).lower()
            mapped = status_map.get(broker_status)
            if mapped:
                new_status, reason = mapped

        # If enabled, mark stale submitted older than X minutes
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
