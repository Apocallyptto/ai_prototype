"""
jobs/manage_stale_orders.py

Stale open-order management for ai_prototype (Alpaca paper bot).

Goal
- Detect limit orders that have been sitting unfilled longer than a threshold and nudge them by
  cancelling + re-submitting at a refreshed price.
- Keep logic simple & robust: we avoid the replace endpoint and do cancel+new to sidestep nuances.
- Respect your existing guardrails (wash-trade avoidance, dedupe window) by tagging client IDs.

Key ideas
- Thresholds: RTH ~60s, ETH ~180s (configurable via env/CLI).
- Only touches ACTIVE limit orders (status=open, type in {limit, stop_limit}). Market/stop orders are skipped.
- Price refresh uses latest quote (bid/ask) from data API feed (iex by default) + optional pad.
- For buys: new_limit = min(ask, last_price + pad). For sells: new_limit = max(bid, last_price - pad).
- Idempotent: per-order max reprice count (tracked via client id suffix `RP#`).

Usage
    $env:PYTHONPATH = "$PWD"
    python -m jobs.manage_stale_orders --symbols AAPL,MSFT,SPY

Optional env/CLI
- REPRICE_RTH_SEC  (int, default 60)
- REPRICE_ETH_SEC  (int, default 180)
- PRICE_PAD_CENTS  (int, default 2)  # small nudge to cross the spread if needed
- ALPACA_DATA_FEED ("iex" default)
- EXTENDED_HOURS   (bool) when submitting replacement orders
- MAX_REPRICES_PER_ORDER (int, default 3)

Notes
- Uses cancel + new order to simplify behavior across statuses. New order inherits side/qty/TIF/type.
- Sets `client_order_id` = f"RP-{orig_client_id}-RP{n}-{int(ts)}".
- Leaves dedupe/wash-trade logic in your executor intact; if you rely on dedupe over (ticker, side),
  consider excluding client IDs beginning with `RP-` from dedupe.
"""
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
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1","true","yes","y"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})


def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
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


# ---- Alpaca helpers ---- #

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
        # Assume RTH if can't tell; keeps system lively
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


def place_order(symbol: str, side: str, qty: str, tif: str, order_type: str, limit_price: Optional[str], stop_price: Optional[str], client_id: str):
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


# ---- Data helpers ---- #

def latest_quote(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    params = {"feed": ALPACA_DATA_FEED}
    r = http("GET", url, params=params)
    if r.status_code == 403 and ALPACA_DATA_FEED.lower() != "iex":
        # fallback to iex
        r = http("GET", url, params={"feed": "iex"})
    if r.status_code >= 300:
        log(f"quote fetch failed for {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("quote") or js  # defensive for possible shapes
    except Exception as e:
        log(f"quote json parse failed for {symbol}: {e}", level="WARN")
        return None


def q2(price: float) -> str:
    return str(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calc_new_limit(symbol: str, side: str, order_type: str) -> Optional[str]:
    q = latest_quote(symbol)
    if not q:
        return None
    bid = float(q.get("bp") or q.get("bid_price") or 0)
    ask = float(q.get("ap") or q.get("ask_price") or 0)
    last = float(q.get("lp") or q.get("last") or 0)
    pad = PRICE_PAD_CENTS / 100.0
    if order_type == "limit":
        if side == "buy":
            target = ask if ask > 0 else (last + pad)
            return q2(target)
        else:
            target = bid if bid > 0 else (last - pad)
            return q2(target)
    # stop-limit: we only adjust the limit leg; leave stop as-is
    if order_type == "stop_limit":
        if side == "buy":
            target = max(ask, last + pad) if ask > 0 else (last + pad)
            return q2(target)
        else:
            target = min(bid, last - pad) if bid > 0 else (last - pad)
            return q2(target)
    return None


def parse_reprice_count(client_order_id: str) -> int:
    # Look for pattern RP<digit> at the end of client id
    # e.g., RP-<orig>-RP2-<ts>
    parts = client_order_id.split("-RP")
    try:
        tail = parts[-1]
        n = int(''.join(ch for ch in tail if ch.isdigit()))
        return n
    except Exception:
        return 0


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
        typ = (o.get("type") or o.get("order_type") or "").lower()
        if typ not in ("limit", "stop_limit"):
            continue
        if o.get("status") not in ("new", "accepted", "open", "partially_filled"):
            continue
        # age
        created = o.get("created_at") or o.get("submitted_at")
        try:
            created_dt = datetime.fromisoformat(created.replace("Z","+00:00"))
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
        # Cancel
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


def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Manage stale open limit orders by cancel+reprice.")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated whitelist, e.g., AAPL,MSFT,SPY")
    ap.add_argument("--debug", action="store_true", help="Just print clock and open orders, no actions")
    args = ap.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(',') if s.strip()] if args.symbols else None
    return args, syms


def print_state(symbols: Optional[List[str]] = None):
    log("Clock/venue state:")
    log(json.dumps(get_clock(), indent=2))
    log("Open orders:")
    js = list_open_orders(symbols)
    log(json.dumps(js, indent=2))


if __name__ == "__main__":
    args, symbols = parse_cli(sys.argv[1:])
    if args.debug:
        print_state(symbols)
        sys.exit(0)
    manage_stale(symbols)
