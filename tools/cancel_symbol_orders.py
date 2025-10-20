# tools/cancel_symbol_orders.py
from __future__ import annotations
import os, sys, json, requests
from datetime import datetime, timezone

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} cancel_symbol | {msg}", flush=True)

def http(method, url, **kwargs):
    return requests.request(method, url, timeout=15, headers={
        "APCA-API-KEY-ID": API_KEY or "",
        "APCA-API-SECRET-KEY": API_SECRET or "",
        "Content-Type": "application/json",
    }, **kwargs)

def cancel_all(symbol: str):
    r = http("GET", f"{ALPACA_BASE_URL}/v2/orders", params={"status":"open","nested":"true","symbols":symbol})
    r.raise_for_status()
    orders = r.json()
    if not orders:
        log(f"No open orders for {symbol}")
        return
    for o in orders:
        oid = o["id"]
        log(f"Cancel {symbol} order {oid} ({o.get('type')}/{o.get('order_class')}, qty={o.get('qty')})")
        rr = http("DELETE", f"{ALPACA_BASE_URL}/v2/orders/{oid}")
        if rr.status_code not in (200,204):
            log(f"  -> cancel failed {rr.status_code} {rr.text}")
    log("Done.")

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Cancel all open orders for a symbol.")
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args(argv)
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")
    cancel_all(args.symbol)

if __name__ == "__main__":
    main(sys.argv[1:])
