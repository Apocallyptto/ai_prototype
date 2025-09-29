# lib/broker_alpaca.py
from __future__ import annotations

import os
import requests
from decimal import Decimal, ROUND_HALF_UP

# ---- Config (env) ----------------------------------------------------------
ALPACA_BASE  = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA  = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
ALPACA_KEY   = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET= os.environ["ALPACA_API_SECRET"]

HDR = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "accept": "application/json",
}

def _round_cents(x: float) -> float:
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def latest_price(ticker: str) -> float:
    r = requests.get(f"{ALPACA_DATA}/v2/stocks/{ticker}/trades/latest",
                     headers=HDR, timeout=15)
    r.raise_for_status()
    p = r.json().get("trade", {}).get("p")
    if not p:
        raise RuntimeError(f"No last trade for {ticker}")
    return float(p)

def place_marketable_limit(
    symbol: str,
    side: str,                  # "buy" | "sell"
    qty: int,
    pad_up: float = float(os.getenv("ALPACA_PAD_UP", "1.05")),     # buy 5% above last
    pad_down: float = float(os.getenv("ALPACA_PAD_DOWN", "0.95")), # sell 5% below last
    extended_hours: bool = True,
) -> dict:
    """
    Submits a DAY+LIMIT that’s deliberately *marketable* so it works
    during RTH and extended hours. Returns a dict with HTTP code,
    raw text/json, last/limit price.
    """
    last = latest_price(symbol)
    lim  = last * (pad_up if side == "buy" else pad_down)
    lim  = _round_cents(lim)

    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "limit",
        "limit_price": lim,
        "time_in_force": "day",
        "extended_hours": extended_hours,
    }

    r = requests.post(f"{ALPACA_BASE}/v2/orders",
                      json=payload,
                      headers={**HDR, "content-type": "application/json"},
                      timeout=20)

    return {
        "http_status": r.status_code,
        "text": r.text,
        "json": (r.json() if r.headers.get("content-type","").startswith("application/json") else None),
        "limit_price": lim,
        "last": last,
    }
