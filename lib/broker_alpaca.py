# lib/broker_alpaca.py
from __future__ import annotations
import os, time, json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' for paper/free
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def _http(method: str, url: str, **kwargs) -> requests.Response:
    backoff = 1.0
    for _ in range(5):
        try:
            r = SESSION.request(method, url, timeout=20, **kwargs)
            # retry 5xx and transient 4xx from data feed
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

# ---------- Trading API ----------
def list_positions() -> List[Dict[str, Any]]:
    r = _http("GET", f"{ALPACA_BASE_URL}/v2/positions")
    r.raise_for_status()
    return r.json()

def list_orders(status: str = "all", nested: bool = True, limit: int = 200, after: Optional[datetime] = None) -> List[Dict[str, Any]]:
    params = {"status": status, "limit": str(limit), "nested": "true" if nested else "false"}
    if after:
        params["after"] = after.isoformat()
    r = _http("GET", f"{ALPACA_BASE_URL}/v2/orders", params=params)
    r.raise_for_status()
    return r.json()

def cancel_order(order_id: str) -> requests.Response:
    return _http("DELETE", f"{ALPACA_BASE_URL}/v2/orders/{order_id}")

def submit_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = _http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    r.raise_for_status()
    return r.json()

# ---------- Market Data ----------
def get_bars(symbol: str, timeframe: str, start: datetime, end: datetime, limit: int = 400) -> List[Dict[str, Any]]:
    """Tries configured feed then falls back to IEX if needed."""
    feeds = [ALPACA_DATA_FEED] + (["iex"] if ALPACA_DATA_FEED.lower() != "iex" else [])
    last_err = None
    for feed in feeds:
        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": limit,
            "adjustment": "split",
            "feed": feed,
        }
        r = _http("GET", f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", params=params)
        if r.status_code == 403:
            last_err = RuntimeError(f"403 on feed '{feed}' — not enabled")
            continue
        try:
            r.raise_for_status()
            js = r.json()
            bars = js.get("bars", [])
            if not bars:
                last_err = RuntimeError(f"no bars for {symbol} on feed={feed}")
                continue
            return bars
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"bar fetch failed for {symbol}: {last_err}")

# ---------- Utils ----------
def quantize_price(price: float, tick: Decimal = Decimal("0.01")) -> float:
    return float(Decimal(str(price)).quantize(tick, rounding=ROUND_HALF_UP))
