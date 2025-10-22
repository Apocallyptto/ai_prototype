# jobs/manage_stale_orders.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# -------- Behavior knobs --------
# Mode affects default age threshold (ETH typically slower fills)
MODE = os.getenv("STALE_MODE", os.getenv("MODE", "ETH")).upper()  # "RTH" or "ETH"
FEED = os.getenv("ALPACA_DATA_FEED", "iex")

# Reprice when order age >= threshold
REPRICE_THRESHOLD_SECONDS_RTH = int(os.getenv("REPRICE_THRESHOLD_SECONDS_RTH", "60"))
REPRICE_THRESHOLD_SECONDS_ETH = int(os.getenv("REPRICE_THRESHOLD_SECONDS_ETH", "180"))

# NEW: Cool-down so we don't churn on tiny quote wiggles
MIN_COOLDOWN_SEC = int(os.getenv("REPRICE_MIN_COOLDOWN_SEC", "45"))

# Only reprice if the limit is "far" from the quote mid by more than this pct
# Example: 0.001 => 0.10% away from mid
REPRICE_AWAY_PCT = float(os.getenv("REPRICE_AWAY_PCT", "0.001"))

# Safety: don't reprice more frequently than this per order (hard floor)
PER_ORDER_MIN_SECONDS = int(os.getenv("PER_ORDER_MIN_SECONDS", "30"))

# Where to peg the new price:
#   For buys => min(ask, last) + SLIPPAGE_TICKS * tick
#   For sells => max(bid, last) - SLIPPAGE_TICKS * tick
SLIPPAGE_TICKS = int(os.getenv("SLIPPAGE_TICKS", "0"))
DEFAULT_TICK = float(os.getenv("DEFAULT_TICK", "0.01"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def log(msg: str, level: str = "INFO"):
    order = ["DEBUG", "INFO", "WARN", "ERROR"]
    if order.index(level) >= order.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} manage_stale | {msg}", flush=True)

def _http(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            wait = min(2 ** attempt, 8)
            log(f"HTTP {method} {url} failed: {e} -> retry {wait}s", "WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _threshold_sec() -> int:
    return REPRICE_THRESHOLD_SECONDS_RTH if MODE == "RTH" else REPRICE_THRESHOLD_SECONDS_ETH

def _tick_size(symbol: str) -> float:
    # If you want per-symbol ticks later, plug a map here.
    return DEFAULT_TICK

def _get_quote(symbol: str) -> Optional[dict]:
    # Alpaca v2 market data (free) — use 'iex' unless you pay for 'sip'
    # https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest?feed=iex
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
        r = _http("GET", url, params={"feed": FEED})
        if r.status_code != 200:
            return None
        return r.json().get("quote")
    except Exception:
        return None

def _get_last(symbol: str) -> Optional[float]:
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
        r = _http("GET", url, params={"feed": FEED})
        if r.status_code != 200:
            return None
        t = r.json().get("trade")
        if not t:
            return None
        return float(t.get("p", 0) or 0)
    except Exception:
        return None

def _quote_mid(symbol: str) -> Optional[float]:
    q = _get_quote(symbol)
    if not q:
        return _get_last(symbol)
    bid = float(q.get("bp", 0) or 0)
    ask = float(q.get("ap", 0) or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return _get_last(symbol)

def _list_open_orders(symbols: Optional[List[str]]) -> List[dict]:
    # Returns open, top-level orders (parents and any single orders)
    params = {"status": "open", "nested": "false", "limit": 500}
    if symbols:
        params["symbols"] = ",".join(symbols)
    r = _http("GET", f"{ALPACA_BASE_URL}/v2/orders", params=params)
    r.raise_for_status()
    return r.json()

def _cancel_order(order_id: str) -> bool:
    r = _http("DELETE", f"{ALPACA_BASE_URL}/v2/orders/{order_id}")
    # 204 normally
    if r.status_code not in (200, 204):
        log(f"cancel {order_id} -> {r.status_code} {r.text}", "WARN")
    return r.status_code in (200, 204)

def _submit_replacement(o: dict, new_limit: float) -> Optional[dict]:
    # Re-post a similar parent (or single) with updated limit
    payload = {
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "type": "limit",
        "qty": o.get("qty", "1"),
        "time_in_force": o.get("time_in_force", "day"),
        "extended_hours": bool(o.get("extended_hours", False)),
        "limit_price": f"{new_limit:.2f}",
        # Mark as a replacement to help dedupe/analytics
        "client_order_id": f"REPRICE-{o.get('symbol')}-{int(time.time())}"
    }
    # Preserve order_class only for non-brackets (we *never* rebuild bracket children here)
    oc = o.get("order_class")
    if oc and oc != "bracket":
        payload["order_class"] = oc

    r = _http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"repost failed: {r.status_code} {r.text}", "ERROR")
        return None
    return r.json()

def _is_parent_bracket(o: dict) -> bool:
    return (o.get("order_class") == "bracket") and (not o.get("parent_order_id"))

def _is_child_of_bracket(o: dict) -> bool:
    return (o.get("order_class") == "bracket") and bool(o.get("parent_order_id"))

def _should_consider(o: dict) -> bool:
    # Consider only top-level **limit** orders we created as parents (either bracket parents or vanilla limit)
    if o.get("status") != "open":
        return False
    if (o.get("type") or "").lower() != "limit":
        return False
    if _is_child_of_bracket(o):
        return False  # let the parent logic manage children
    return True

def _price(o: dict, key: str) -> Optional[float]:
    v = o.get(key)
    try:
        return None if v in (None, "") else float(v)
    except Exception:
        return None

def _compute_new_limit(symbol: str, side: str, current_limit: float) -> Optional[float]:
    mid = _quote_mid(symbol)
    if not mid or mid <= 0:
        return None
    away = abs(current_limit - mid) / mid
    if away < REPRICE_AWAY_PCT:
        return None  # close enough; no need to churn

    tick = _tick_size(symbol)
    last = _get_last(symbol) or mid
    if side == "buy":
        peg = min(last, mid) + SLIPPAGE_TICKS * tick
        return round(peg / tick) * tick
    else:
        peg = max(last, mid) - SLIPPAGE_TICKS * tick
        return round(peg / tick) * tick

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=False, help="Comma-separated tickers, optional")
    args = ap.parse_args()
    syms = None
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    threshold = _threshold_sec()
    log(f"mode={MODE} threshold={threshold}s feed={FEED}")

    opened = _list_open_orders(syms)
    now = _now()
    acted = 0

    for o in opened:
        if not _should_consider(o):
            continue

        symbol = o.get("symbol")
        side = (o.get("side") or "").lower()  # 'buy'/'sell'
        if not symbol or side not in ("buy", "sell"):
            continue

        created_dt = _iso_to_dt(o.get("created_at")) or now
        updated_dt = _iso_to_dt(o.get("updated_at")) or created_dt
        submitted_dt = _iso_to_dt(o.get("submitted_at")) or created_dt

        age = (now - created_dt).total_seconds()

        # -------- NEW: Cool-down guard (prevents flapping) --------
        last_update = _iso_to_dt(o.get("updated_at")) or _iso_to_dt(o.get("created_at")) or _iso_to_dt(o.get("submitted_at")) or created_dt
        cool = (now - last_update).total_seconds()
        if cool < MIN_COOLDOWN_SEC:
            # too fresh to touch
            continue
        # -------- END Cool-down guard --------

        # Hard floor so we don't touch hyper-fresh orders even if created_at parsing was odd
        if (now - submitted_dt).total_seconds() < PER_ORDER_MIN_SECONDS:
            continue

        if age < threshold:
            continue

        limit_px = _price(o, "limit_price")
        if not limit_px or limit_px <= 0:
            continue

        # Decide if price is far enough from mid to bother
        new_limit = _compute_new_limit(symbol, side, limit_px)
        if new_limit is None or abs(new_limit - limit_px) < 1e-9:
            continue  # close enough; skip

        # Cancel + repost
        if not _cancel_order(o["id"]):
            # If cancel failed (race with fill or already canceled), skip
            continue

        new_o = _submit_replacement(o, new_limit)
        if new_o:
            acted += 1
            log(f"{symbol} {side}: repriced {limit_px:.2f} -> {new_limit:.2f} (age={int(age)}s)")

    if acted == 0:
        log("no stale orders to reprice")
    else:
        log(f"repriced {acted} order(s)")

if __name__ == "__main__":
    main()
