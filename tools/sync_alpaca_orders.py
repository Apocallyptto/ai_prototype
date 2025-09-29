# tools/sync_alpaca_orders.py
from __future__ import annotations
import os, requests, datetime as dt
from lib.db_orders import log_order_row

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
K = os.environ["ALPACA_API_KEY"]
S = os.environ["ALPACA_API_SECRET"]
HDR = {"APCA-API-KEY-ID": K, "APCA-API-SECRET-KEY": S, "accept": "application/json"}

def main():
    since_days = int(os.getenv("SYNC_SINCE_DAYS", "2"))
    since = (dt.datetime.utcnow() - dt.timedelta(days=since_days)).isoformat() + "Z"

    r = requests.get(
        f"{BASE}/v2/orders",
        params={"status": "all", "after": since, "limit": 500},
        headers=HDR,
        timeout=30,
    )
    r.raise_for_status()
    orders = r.json()
    ok = 0
    for o in orders:
        ok += 1 if log_order_row(
            ts_iso=o.get("submitted_at") or o.get("created_at"),
            ticker=o["symbol"],
            side=o["side"],
            qty=float(o.get("qty") or o.get("filled_qty") or 0) or 0.0,
            order_type=o.get("type") or o.get("order_type") or "limit",
            limit_price=float(o["limit_price"]) if o.get("limit_price") else None,
            status=o.get("status", "submitted"),
        ) else 0
    print(f"Synced {ok}/{len(orders)} orders into DB.")

if __name__ == "__main__":
    main()
