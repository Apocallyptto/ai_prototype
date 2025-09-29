# lib/broker_alpaca.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry


# --- config from env (paper by default)
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
ALPACA_KEY  = os.environ["ALPACA_API_KEY"]
ALPACA_SEC  = os.environ["ALPACA_API_SECRET"]

HDR = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SEC,
    "accept": "application/json",
    "content-type": "application/json",
}

# --- a resilient HTTP session (retries + backoff)
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,                  # 1 try + 4 retries
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.6,       # 0.6, 1.2, 2.4, 4.8...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "DELETE"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SES = _session()


# --- price helpers -----------------------------------------------------------
def latest_price(ticker: str, timeout: float = 25.0) -> float:
    """
    Get the most recent trade price. Falls back to quotes/latest.
    Raises RuntimeError if neither endpoint yields a price.
    """
    # Try trades/latest
    try:
        r = _SES.get(
            f"{ALPACA_DATA}/v2/stocks/{ticker}/trades/latest",
            headers=HDR,
            timeout=timeout,
        )
        if r.ok:
            px = r.json().get("trade", {}).get("p")
            if px:
                return float(px)
    except requests.RequestException:
        pass

    # Fallback: quotes/latest (use mid of bid/ask if both present; else the one we have)
    try:
        r = _SES.get(
            f"{ALPACA_DATA}/v2/stocks/{ticker}/quotes/latest",
            headers=HDR,
            timeout=timeout,
        )
        if r.ok:
            q = r.json().get("quote", {}) or {}
            bp = q.get("bp")
            ap = q.get("ap")
            if bp and ap:
                return (float(bp) + float(ap)) / 2.0
            if ap:
                return float(ap)
            if bp:
                return float(bp)
    except requests.RequestException:
        pass

    raise RuntimeError(f"Could not fetch latest price for {ticker}")


# --- order placement ---------------------------------------------------------
def place_marketable_limit(
    symbol: str,
    side: str,
    qty: int,
    *,
    pad_up: float = 1.05,
    pad_down: float = 0.95,
    time_in_force: str = "day",
    extended_hours: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Place a DAY+LIMIT order with a 'marketable' price:
      - BUY at last * pad_up
      - SELL at last * pad_down

    Returns:
        {
          "http_status": int|None,
          "json": dict|None,
          "text": str,
          "last": float|None,
          "limit_price": float|None,
        }
    """
    result: Dict[str, Any] = {
        "http_status": None,
        "json": None,
        "text": "",
        "last": None,
        "limit_price": None,
    }

    # Get a resilient last price
    try:
        last = latest_price(symbol)
        result["last"] = last
    except Exception as e:
        result["text"] = f"price_error: {e}"
        return result  # executor will log and skip

    # Compute marketable limit
    if side.lower() == "buy":
        limit_px = round(last * pad_up + 1e-9, 2)
    else:
        limit_px = round(last * pad_down + 1e-9, 2)
    result["limit_price"] = limit_px

    payload = {
        "symbol": symbol,
        "qty": int(qty),
        "side": side.lower(),
        "type": "limit",
        "limit_price": limit_px,
        "time_in_force": time_in_force,
        "extended_hours": bool(extended_hours),
    }

    try:
        r = _SES.post(f"{ALPACA_BASE}/v2/orders", json=payload, headers=HDR, timeout=timeout)
        result["http_status"] = r.status_code
        result["text"] = r.text
        if r.ok:
            result["json"] = r.json()
    except requests.RequestException as e:
        result["text"] = f"post_error: {e}"

    return result
