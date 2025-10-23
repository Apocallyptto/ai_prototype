# services/bracket_helper.py
from __future__ import annotations
import os, time, json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, List, Dict, Any

import requests

# === Alpaca endpoints & auth ===
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")  # paper/free: iex (or sip for paid)

# === Exit config (simple, safe defaults) ===
# If you prefer ATR-based sizing, you can switch to your ATR utils here.
BASE_TP_PCT = float(os.getenv("BASE_TP_PCT", "0.006"))  # 0.6%
BASE_SL_PCT = float(os.getenv("BASE_SL_PCT", "0.004"))  # 0.4%

# Bracket orders MUST have extended_hours = False on Alpaca
EXTENDED_HOURS = False

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} INFO bracket_helper | {msg}")

def _err(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} ERROR bracket_helper | {msg}")

def _http(method: str, url: str, **kwargs) -> requests.Response:
    r = SESSION.request(method, url, timeout=15, **kwargs)
    return r

# --------------- Market utilities ---------------

def _clock_is_open() -> bool:
    try:
        r = _http("GET", f"{ALPACA_BASE_URL}/v2/clock")
        r.raise_for_status()
        j = r.json()
        return bool(j.get("is_open"))
    except Exception as e:
        _err(f"clock check failed: {e}")
        # Fail-safe: treat as closed (force limit entries) to avoid market orders at odd hours
        return False

def _latest_trade_price(symbol: str) -> float:
    # Alpaca data v2: GET /v2/stocks/{symbol}/trades/latest
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": DATA_FEED}
    r = _http("GET", url, params=params)
    try:
        r.raise_for_status()
        j = r.json()
        p = j.get("trade", {}).get("p")
        if p is None:
            raise ValueError("missing trade price")
        return float(p)
    except Exception as e:
        _err(f"latest trade fetch failed for {symbol}: {e}")
        # very conservative fallback
        raise

def _q(x: float) -> float:
    # round to cents
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _compute_exits(side: str, base_price: float) -> Tuple[float, float]:
    if side.lower() == "buy":
        tp = base_price * (1.0 + BASE_TP_PCT)
        sl = base_price * (1.0 - BASE_SL_PCT)
    else:
        tp = base_price * (1.0 - BASE_TP_PCT)
        sl = base_price * (1.0 + BASE_SL_PCT)
    return _q(tp), _q(sl)

# --------------- Public helpers ---------------

def list_open_orders(symbols: Optional[List[str]] = None) -> List[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(s.upper() for s in symbols)
    r = _http("GET", url)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------
# BACKWARD-COMPAT: Old executor calls this signature (no 'api', no price)
# submit_bracket(symbol, side, qty, prefer_limit_when_closed=True, ref_atr=None)
# ---------------------------------------------------------------------
def submit_bracket(symbol: str,
                   side: str,
                   qty: Optional[int],
                   *,
                   prefer_limit_when_closed: bool = True,
                   ref_atr: Optional[float] = None) -> dict:
    """
    Old-style entry used by services.executor_bracket.
    - Detects market open/closed
    - Fetches latest price
    - Builds bracket with extended_hours=False
    - Posts via raw HTTP (requests)

    Returns Alpaca order JSON on success (raises on error).
    """
    qty_int = int(qty or 1)
    is_open = _clock_is_open()
    base = _latest_trade_price(symbol)

    tp, sl = _compute_exits(side, base)

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side.lower(),
        "qty": str(qty_int),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": EXTENDED_HOURS,  # MUST be False for brackets
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
    }

    if is_open:
        payload["type"] = "market"
    else:
        payload["type"] = "limit" if prefer_limit_when_closed else "market"
        # small nudge around last price for after-hours limit anchoring (defensive)
        if side.lower() == "buy":
            payload["limit_price"] = f"{_q(base + 0.05):.2f}"
        else:
            payload["limit_price"] = f"{_q(base - 0.05):.2f}"

    _log(f"submit bracket: {payload}")
    r = _http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        _log(f"submit failed: {r.status_code} {r.text}")
        r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------
# NEW-STYLE convenience: for callers that already have an SDK client & price
# submit_bracket_v2(api, symbol, side, qty, last_price, client_id)
# ---------------------------------------------------------------------
def submit_bracket_v2(api,
                      symbol: str,
                      side: str,
                      qty: int | float,
                      last_price: float,
                      client_id: str) -> dict:
    """
    New-style submit using an Alpaca SDK client:
        api.submit_order(**payload)
    """
    qty_int = int(qty or 1)
    tp, sl = _compute_exits(side, float(last_price))

    payload = {
        "symbol": symbol,
        "side": side.lower(),
        "type": "limit",                # SDK path: keep limit with explicit anchor
        "qty": str(qty_int),
        "time_in_force": "day",
        "order_class": "bracket",
        "limit_price": f"{float(last_price):.2f}",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": EXTENDED_HOURS,     # MUST be False
        "client_order_id": client_id,
    }
    _log(f"submit bracket (v2): {payload}")
    try:
        return api.submit_order(**payload)
    except Exception as e:
        # best-effort surfacing of Alpaca error bodies
        try:
            msg = getattr(e, "response", None)
            if msg is not None:
                _log(f"submit failed: {msg.text}")
        except Exception:
            pass
        _err(f"submit failed: {e}")
        raise

# Alias kept for older code that imported this name by mistake
def submit_bracket_entry(symbol: str, side: str, qty: Optional[int], prefer_limit_when_closed: bool = True) -> dict:
    return submit_bracket(symbol, side, qty, prefer_limit_when_closed=prefer_limit_when_closed)
