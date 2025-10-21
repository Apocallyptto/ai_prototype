# tools/sync_orders.py
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone

from lib.broker_alpaca import list_orders
from lib.db_orders import upsert_orders

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def log(msg, level="INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} sync_orders | {msg}", flush=True)

def main():
    lookback_days = int(os.getenv("ORDERS_LOOKBACK_DAYS","7"))
    after = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    orders = list_orders(status="all", nested=True, limit=500, after=after)
    log(f"fetched {len(orders)} orders since {after.isoformat()}")
    n = upsert_orders(orders)
    log(f"sync complete (upserted {n})")

if __name__ == "__main__":
    main()
