# jobs/manage_stale_orders.py
from __future__ import annotations
import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

REPRICE_RTH_SEC = int(os.getenv("REPRICE_RTH_SEC", "60"))
REPRICE_ETH_SEC = int(os.getenv("REPRICE_ETH_SEC", "180"))
PRICE_PAD_CENTS = int(os.getenv("PRICE_PAD_CENTS", "2"))
MAX_REPRICES_PER_ORDER = int(os.getenv("MAX_REPRICES_PER_ORDER", "3"))
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1", "true", "yes", "y"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} manage_stale | {msg}")

def http(method: str, url: str, **kwargs):
    for attempt in range(5):
        try:
            resp = SESSION.request(method, url, timeout=15, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} {resp.text}")
            return resp
        except Exception as e:
            wait = min(2 ** attempt, 8)
            log(f"HTTP error {e} -> retrying in {wait}s", level="WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

# ---- Alpaca helpers ----

def get_clock() -> dict:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/clock")
    r.raise_for_status()
    return r.json()

def is_rth_now() -> bool:
    try:
        clock = get_clock()
        return bool(clock.get("is_open"))
    except Exception as e:
        log(f"clock check failed: {e}", level="WARN")
        # If uncertain, assume RTH so we reprice more responsively
        return True

def list_open_orders(symbols: Optional[List[str]] = None) -> List[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbols:
        url += f"&symbols={','.join(symbols)}"
    r = http("GET", url)
    r.raise_for_status()
    return r.json()

def cancel_order(order_id: str):
    r = http("DELETE", f"{ALPACA_BASE_URL}/v2/orders/{order_id}")
    if r.status_code not in (200, 204):
        log(f"cancel failed {order_id}: {r.status_code} {r.text}", level="WARN")
    return r

def place_order(
    symbol: str,
    side: str,
    qty: str,
    tif: str,
    order_type: str,
    limit_price: Optional[str],
    stop_price: Optional[str],
    client_id: str,
):
    payload = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "time_in_force": tif,
        "type": order_type,
        "client_order_id": client_id,
        "extended_hours": EXTENDED_HOURS,
    }
    if order_type in ("limit", "stop_limit") and limit_price is not None:
        payload["limit_price"] = limit_price
    if order_type in ("stop", "stop_limit") and stop_price is not None:
        payload["stop_price"] = stop_price

    r = http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"repost failed: {r.status_code} {r.text}", level="WARN")
    return r

# ---- Data helpers ----

def _q2(price: float) -> str:
    return str(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def latest_quote(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = http("GET", url, params={"feed": ALPACA_DATA_FEED})
    if r.status_code == 403 and ALPACA_DATA_FEED.lower() != "iex":
        # Fallback to iex if current feed is not allowed
        r = http("GET", url, params={"feed": "iex"})
    if r.status_code >= 300:
        log(f"quote fetch failed for {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("quote") or js
    except Exception as e:
        log(f"quote json parse failed for {symbol}: {e}", level="WARN")
        return None

def calc_new_limit(symbol: str, side: str, order_type: str) -> Optional[str]:
    q = latest_quote(symbol)
    if not q:
        return None
    # Common Alpaca fields: bp/ap (best bid/ask), lp (last price)
    bid = float(q.get("bp") or q.get("bid_price") or 0)
    ask = float(q.get("ap") or q.get("ask_price") or 0)
    last = float(q.get("lp") or q.get("last") or 0)
    pad = PRICE_PAD_CENTS / 100.0

    if order_type == "limit":
        if side == "buy":
            target = ask if ask > 0 else (last + pad)
            return _q2(target)
        else:
            target = bid if bid > 0 else (last - pad)
            return _q2(target)

    # For stop-limit, we adjust only the limit leg; leave stop untouched
    if order_type == "stop_limit":
        if side == "buy":
            target = max(ask, last + pad) if ask > 0 else (last + pad)
            return _q2(target)
        else:
            target = min(bid, last - pad) if bid > 0 else (last - pad)
            return _q2(target)

    return None

# ---- Reprice bookkeeping ----

def parse_reprice_count(client_order_id: str) -> int:
    # Only count if '-RP' appears; otherwise 0
    if "-RP" not in client_order_id:
        return 0
    parts = client_order_id.split("-RP")
    try:
        tail = parts[1] if len(parts) > 1 else ""
        digits = []
        for ch in tail:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        return int("".join(digits)) if digits else 0
    except Exception:
        return 0

# ---- Main logic ----

def manage_stale(symbols: Optional[List[str]] = None):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    rth = is_rth_now()
    threshold = REPRICE_RTH_SEC if rth else REPRICE_ETH_SEC
    log(f"mode={'RTH' if rth else 'ETH'} threshold={threshold}s feed={ALPACA_DATA_FEED}")

    orders = list_open_orders(symbols)
    if not orders:
        log("No open orders to manage.")
        return

    now = datetime.now(timezone.utc)

    for o in orders:
        # Skip OCO/bracket/child orders — only reprice standalone entries/adjustments
        if (o.get("order_class") in ("oco", "bracket")) or o.get("parent_order_id") or o.get("legs"):
            continue

        typ = (o.get("type") or o.get("order_type") or "").lower()
        if typ not in ("limit", "stop_limit"):
            continue
        if o.get("status") not in ("new", "accepted", "open", "partially_filled"):
            continue

        # Age
        created = o.get("created_at") or o.get("submitted_at")
        try:
            created_dt = datetime.fromisoformat((created or "").replace("Z", "+00:00"))
        except Exception:
            created_dt = now - timedelta(hours=1)
        age = (now - created_dt).total_seconds()
        if age < threshold:
            continue

        # Reprice count guard
        cid = o.get("client_order_id", "")
        rp_count = parse_reprice_count(cid)
        if rp_count >= MAX_REPRICES_PER_ORDER:
            log(f"{o['symbol']} order {o['id']} hit max reprice count; skipping.")
            continue

        new_limit = calc_new_limit(o["symbol"], o["side"], typ)
        if not new_limit:
            log(f"{o['symbol']} unable to compute new limit; skipping.")
            continue

        log(f"Repricing {o['symbol']} {typ} {o['side']} qty={o['qty']} age={int(age)}s -> new limit {new_limit}")
        # Cancel old
        cancel_order(o["id"])
        # Repost
        new_cid = f"RP-{cid}-RP{rp_count+1}-{int(time.time())}"
        place_order(
            symbol=o["symbol"],
            side=o["side"],
            qty=o["qty"],
            tif=o.get("time_in_force", "day"),
            order_type=typ,
            limit_price=new_limit,
            stop_price=o.get("stop_price"),
            client_id=new_cid,
        )

def print_state(symbols: Optional[List[str]] = None):
    log("Clock/venue state:")
    try:
        log(json.dumps(get_clock(), indent=2))
    except Exception as e:
        log(f"clock fetch failed: {e}", level="WARN")
    log("Open orders:")
    try:
        js = list_open_orders(symbols)
        log(json.dumps(js, indent=2))
    except Exception as e:
        log(f"list_open_orders failed: {e}", level="WARN")

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Manage stale open limit orders by cancel+reprice.")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated whitelist, e.g., AAPL,MSFT,SPY")
    ap.add_argument("--debug", action="store_true", help="Just print clock and open orders, no actions")
    args = ap.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    return args, syms

if __name__ == "__main__":
    args, symbols = parse_cli(sys.argv[1:])
    if args.debug:
        print_state(symbols)
        sys.exit(0)
    manage_stale(symbols)
