# tools/list_open_orders.py
from __future__ import annotations
import os, sys, requests
from datetime import datetime, timezone

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def main():
    symbols = os.environ.get("SYMBOLS", "").strip()
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&direction=desc&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(s.strip().upper() for s in symbols.split(",") if s.strip())
    r = S.get(url, timeout=15)
    r.raise_for_status()
    orders = r.json()
    if not orders:
        print("No open orders.")
        return
    print(f"Open orders: {len(orders)}")
    for o in orders:
        side = o.get("side")
        sym = o.get("symbol")
        oclass = o.get("order_class")
        typ = o.get("type")
        qty = o.get("qty")
        lp = o.get("limit_price")
        st = o.get("stop_price")
        cid = o.get("client_order_id")
        created = o.get("created_at")
        print(f"- {created}  {sym}  {side}  {oclass}/{typ}  qty={qty}  L={lp}  S={st}  cid={cid}")

if __name__ == "__main__":
    main()
