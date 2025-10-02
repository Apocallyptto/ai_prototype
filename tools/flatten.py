# tools/flatten.py
import os, sys, time, math, requests

ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
K = os.environ["ALPACA_API_KEY"]
S = os.environ["ALPACA_API_SECRET"]
HDR = {"APCA-API-KEY-ID": K, "APCA-API-SECRET-KEY": S, "accept": "application/json", "content-type":"application/json"}

def latest_price(ticker: str) -> float:
    r = requests.get(f"{ALPACA_DATA}/v2/stocks/{ticker}/trades/latest", headers=HDR, timeout=15)
    r.raise_for_status()
    px = r.json().get("trade", {}).get("p")
    if not px:
        raise RuntimeError(f"No last trade for {ticker}")
    return float(px)

def cancel_open_orders(symbols=None):
    r = requests.get(f"{ALPACA_BASE}/v2/orders?status=open&limit=500", headers=HDR, timeout=20)
    r.raise_for_status()
    opens = r.json() if isinstance(r.json(), list) else []
    count = 0
    for o in opens:
        if symbols and o.get("symbol") not in symbols:
            continue
        oid = o["id"]
        requests.delete(f"{ALPACA_BASE}/v2/orders/{oid}", headers=HDR, timeout=15)
        count += 1
    print(f"Canceled {count} open order(s).")

def flatten_symbol(ticker: str, pad_up=1.02, pad_down=0.98):
    # get position
    r = requests.get(f"{ALPACA_BASE}/v2/positions/{ticker}", headers=HDR, timeout=15)
    if r.status_code == 404:
        print(f"{ticker}: no position.")
        return
    r.raise_for_status()
    pos = r.json()
    qty = float(pos["qty"])
    if qty == 0:
        print(f"{ticker}: qty already 0.")
        return

    side = "sell" if qty > 0 else "buy"  # close long -> sell, close short -> buy
    qty_abs = abs(int(qty))  # paper acct uses whole shares in your setup

    last = latest_price(ticker)
    lim = last * (pad_up if side=="buy" else pad_down)
    lim = round(lim + 1e-9, 2)

    payload = {
        "symbol": ticker,
        "qty": qty_abs,
        "side": side,
        "type": "limit",
        "limit_price": lim,
        "time_in_force": "day",
        "extended_hours": True
    }
    rr = requests.post(f"{ALPACA_BASE}/v2/orders", json=payload, headers=HDR, timeout=20)
    text = rr.text
    try:
        js = rr.json()
    except Exception:
        js = None
    print(f"{ticker}: close {side} {qty_abs} @ {lim} -> HTTP {rr.status_code} | {text[:200]}")
    if rr.status_code >= 300:
        rr.raise_for_status()

def main():
    # symbols from CLI, e.g.:  python tools/flatten.py AAPL MSFT SPY
    syms = sys.argv[1:]
    if not syms:
        # If none given, close everything that has a position
        r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=HDR, timeout=15)
        r.raise_for_status()
        syms = [p["symbol"] for p in r.json()]
    if not syms:
        print("No positions to flatten.")
        return

    cancel_open_orders(set(syms))
    # small pause so server side releases reserved qty
    time.sleep(0.8)

    for t in syms:
        try:
            flatten_symbol(t)
        except Exception as e:
            print(f"{t}: error -> {e}")

if __name__ == "__main__":
    main()
