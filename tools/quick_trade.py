# tools/quick_trade.py
from __future__ import annotations
import os, sys, json, requests
from datetime import datetime, timezone

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} quick_trade | {msg}")

def place_market(symbol: str, side: str, qty: str = "1", tif: str = "day"):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")
    h = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Content-Type": "application/json",
    }
    p = {"symbol": symbol, "side": side, "type": "market", "time_in_force": tif, "qty": qty}
    r = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=h, data=json.dumps(p), timeout=15)
    log(f"POST /v2/orders -> {r.status_code}")
    try:
        js = r.json()
    except Exception:
        js = {"text": r.text}
    print(json.dumps(js, indent=2))
    r.raise_for_status()
    return js

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Quick market trade helper (paper).")
    ap.add_argument("--symbol", required=True, help="Ticker, e.g., AAPL")
    ap.add_argument("--side", required=True, choices=["buy","sell"], help="buy or sell")
    ap.add_argument("--qty", default="1", help="Quantity, default 1")
    ap.add_argument("--tif", default="day", choices=["day","gtc"], help="Time in force")
    args = ap.parse_args(argv)
    place_market(args.symbol, args.side, args.qty, args.tif)

if __name__ == "__main__":
    main(sys.argv[1:])
