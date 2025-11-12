#!/usr/bin/env python3
"""
tools/sync_orders.py
Sync local 'signals' table with Alpaca order status.
"""

import os, psycopg2
from datetime import datetime, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

lookback_days = int(os.getenv("LOOKBACK_DAYS", "3"))
paper = os.getenv("ALPACA_PAPER", "1") != "0"

tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=paper)
orders = tc.get_orders(GetOrdersRequest(status=None, limit=500))
orders_map = {str(o.id): o for o in orders}

db_url = os.environ["DB_URL"]
updated = 0

with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT id, symbol, side, order_id, status
          FROM signals
         WHERE status='submitted'
      ORDER BY id DESC;
    """)
    rows = cur.fetchall()

    for row in rows:
        sid, symbol, side, order_id, db_status = row
        o = orders_map.get(str(order_id))
        if not o:
            continue

        new_status = None
        reason = None
        if o.status == "filled":
            new_status = "filled"
            reason = "order filled"
        elif o.status in ("canceled", "expired"):
            new_status = "skipped"
            reason = f"broker {o.status}"
        elif o.status in ("replaced", "rejected"):
            new_status = "error"
            reason = f"broker {o.status}"
        if new_status and new_status != db_status:
            cur.execute("""
                UPDATE signals
                   SET status=%s,
                       processed_at=NOW(),
                       status_reason=%s
                 WHERE id=%s;
            """, (new_status, reason, sid))
            updated += 1

    conn.commit()

print(f"checked {len(rows)} submitted rows, updated {updated}")
