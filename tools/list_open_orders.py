# tools/list_open_orders.py
from __future__ import annotations

import os
import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
S.headers.update(
    {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
)

def _fmt(v):
    return "None" if v is None or v == "" else str(v)

def _print_order(prefix: str, o: dict):
    created = o.get("created_at") or o.get("submitted_at") or o.get("updated_at")
    sym = o.get("symbol")
    side = o.get("side")
    oclass = o.get("order_class")
    typ = o.get("type")
    qty = o.get("qty")
    lp = o.get("limit_price")
    stp = o.get("stop_price")
    status = o.get("status")
    oid = o.get("id")
    cid = o.get("client_order_id")

    print(
        f"{prefix}{_fmt(created)}  {sym}  {side}  {oclass}/{typ}  "
        f"qty={_fmt(qty)}  L={_fmt(lp)}  S={_fmt(stp)}  st={_fmt(status)}  "
        f"id={_fmt(oid)}  cid={_fmt(cid)}"
    )

def main():
    # env:
    #   SYMBOLS="AAPL,MSFT,SPY" (optional filter)
    #   NESTED="true|false"     (default true)
    symbols_env = os.environ.get("SYMBOLS", "").strip()
    nested_env = os.environ.get("NESTED", "true").strip().lower()
    nested = nested_env not in ("0", "false", "no")

    params = {
        "status": "open",
        "direction": "desc",
        "nested": "true" if nested else "false",
    }
    if symbols_env:
        params["symbols"] = ",".join(
            s.strip().upper() for s in symbols_env.split(",") if s.strip()
        )

    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = S.get(url, params=params, timeout=15)
    r.raise_for_status()

    orders = r.json()
    if not orders:
        print("No open orders.")
        return

    print(f"Open orders: {len(orders)} (nested={nested})")
    for o in orders:
        _print_order("- ", o)

        # If nested=true, Alpaca may include legs under "legs"
        legs = o.get("legs") or []
        if legs:
            print(f"  legs: {len(legs)}")
            for l in legs:
                _print_order("    LEG - ", l)

if __name__ == "__main__":
    main()
