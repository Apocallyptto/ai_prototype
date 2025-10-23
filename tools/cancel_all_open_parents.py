# tools/cancel_all_open_parents.py
from __future__ import annotations
import os, sys, requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
FOR_REAL   = os.getenv("FOR_REAL", "0").lower() in {"1","true","yes","y"}

S = requests.Session()
S.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
})

def main():
    symbols = [s.strip().upper() for s in os.environ.get("SYMBOLS","").split(",") if s.strip()]
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&direction=desc&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(symbols)
    r = S.get(url, timeout=15); r.raise_for_status()
    orders = r.json() or []

    # only cancel parent orders (have order_class == 'bracket' and no parent_id)
    parents = [o for o in orders if o.get("order_class") == "bracket" and not o.get("parent_order_id")]
    if not parents:
        print("No open parent bracket orders to cancel.")
        return

    print(f"{'DRY-RUN' if not FOR_REAL else 'CANCEL'}: {len(parents)} parent orders")
    for p in parents:
        oid = p.get("id"); sym = p.get("symbol"); side = p.get("side"); cid = p.get("client_order_id")
        print(f"- {sym} {side} parent id={oid} cid={cid}")
        if FOR_REAL and oid:
            rr = S.delete(f"{ALPACA_BASE_URL}/v2/orders/{oid}", timeout=15)
            if rr.status_code >= 300:
                print(f"  ! cancel failed: {rr.status_code} {rr.text}")
            else:
                print("  cancelled")

if __name__ == "__main__":
    main()
