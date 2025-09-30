# lib/broker_alpaca.py
import os
import math
import logging
import requests
from requests.adapters import HTTPAdapter, Retry

log = logging.getLogger(__name__)

# -------- retrying HTTP session (handles brief network blips) ----------
def _retrying_session(total: int = 3, backoff: float = 0.5) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


SESSION = _retrying_session()

# -------- env / endpoints ----------
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
KEY = os.environ["ALPACA_API_KEY"]
SEC = os.environ["ALPACA_API_SECRET"]

HDR = {
    "APCA-API-KEY-ID": KEY,
    "APCA-API-SECRET-KEY": SEC,
    "accept": "application/json",
}

# -------- data helpers ----------
def latest_price(ticker: str) -> float:
    """Return the latest trade price from Alpaca Data API."""
    r = SESSION.get(
        f"{ALPACA_DATA}/v2/stocks/{ticker}/trades/latest",
        headers=HDR,
        timeout=15,
    )
    r.raise_for_status()
    px = r.json().get("trade", {}).get("p")
    if not px:
        raise RuntimeError(f"No last trade for {ticker}")
    return float(px)


# -------- order helper ----------
def place_marketable_limit(
    symbol: str,
    side: str,
    qty: int,
    pad_up: float = 1.05,
    pad_down: float = 0.95,
    extended_hours: bool = True,
):
    """
    Submit a DAY + LIMIT 'marketable' order that works during RTH and extended hours.
    We compute a limit just beyond the last trade (buy: above / sell: below).

    Returns a dict with:
      http_status, json (response JSON or None), text (raw text),
      last (float), limit_price (float)
    """
    side = side.lower().strip()
    assert side in ("buy", "sell"), f"invalid side: {side}"

    last = latest_price(symbol)

    # compute marketable limit
    limit_price = last * (pad_up if side == "buy" else pad_down)
    # round to cents
    limit_price = round(limit_price + 1e-9, 2)

    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "limit",
        "limit_price": limit_price,
        "time_in_force": "day",
        "extended_hours": bool(extended_hours),
    }

    r = SESSION.post(
        f"{ALPACA_BASE}/v2/orders",
        json=payload,
        headers={**HDR, "content-type": "application/json"},
        timeout=20,
    )

    text = r.text
    data = None
    try:
        data = r.json()
    except Exception:
        pass

    log.info(
        "alpaca order -> %s | last=%.2f lim=%.2f | %s",
        r.status_code,
        last,
        limit_price,
        text[:500],
    )

    return {
        "http_status": r.status_code,
        "json": data,
        "text": text,
        "last": last,
        "limit_price": limit_price,
    }
