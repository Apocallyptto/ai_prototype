"""
Bracket helper

- Builds and submits Alpaca bracket (parent + TP + SL).
- Optional ATR-aware entry/TP/SL prices (env toggles).
- Exposes a legacy-friendly `submit_bracket(...)` for older imports.

Env of interest:
  ALPACA_BASE_URL (default paper)
  ALPACA_API_KEY / ALPACA_API_SECRET
  EXTENDED_HOURS (true/false)

  USE_ATR_ENTRY=1
  ATR_MULT_TP_ENTRY, ATR_MULT_SL_ENTRY
  ATR_TIMEFRAME (e.g., 5Min), ATR_LENGTH (e.g., 14)
  ALPACA_DATA_URL (default https://data.alpaca.markets)
  ALPACA_DATA_FEED (paper: iex)

  PARENT_TYPE: 'limit' (default) or 'market'
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import requests

# -------------------- Config --------------------

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1", "true", "yes", "y"}

# Parent type default can be flipped via env
DEFAULT_PARENT_TYPE = os.getenv("PARENT_TYPE", "limit").lower()  # 'limit' | 'market'

# ATR knobs for entries (independent from exits manager)
USE_ATR_ENTRY = os.getenv("USE_ATR_ENTRY", "0").lower() in {"1", "true", "yes", "y"}
ATR_MULT_TP_ENTRY = float(os.getenv("ATR_MULT_TP_ENTRY", "1.2"))
ATR_MULT_SL_ENTRY = float(os.getenv("ATR_MULT_SL_ENTRY", "1.0"))
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # free/paper: 'iex'

# request session
_sess = requests.Session()
_sess.headers.update(
    {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Content-Type": "application/json",
    }
)

def _http(method: str, url: str, **kwargs) -> requests.Response:
    # small retry for transient network/server hiccups
    for attempt in range(5):
        try:
            r = _sess.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            wait = min(2**attempt, 8)
            print(f"{datetime.utcnow().isoformat()}Z WARN bracket_helper | HTTP {e} -> retrying {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def _q(x: float, tick: Decimal = Decimal("0.01")) -> float:
    return float(Decimal(str(x)).quantize(tick, rounding=ROUND_HALF_UP))

# -------------------- (optional) ATR fetch --------------------

def _recent_bars(symbol: str, timeframe: str, lookback_days: int = 10):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (now - timedelta(days=lookback_days)).isoformat()
    end = now.isoformat()

    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 400,
        "adjustment": "split",
        "feed": ALPACA_DATA_FEED,
    }
    r = _http("GET", url, params=params)
    r.raise_for_status()
    js = r.json()
    bars = js.get("bars", [])
    if not bars:
        raise RuntimeError(f"No bars for {symbol}")
    o = [float(b["o"]) for b in bars]
    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    c = [float(b["c"]) for b in bars]
    return o, h, l, c

def _atr_from_ohlc(h, l, c, length: int) -> float:
    # Wilder ATR (EMA of True Range)
    tr = []
    prev_c = None
    for hi, lo, cls in zip(h, l, c):
        if prev_c is None:
            tr.append(hi - lo)
        else:
            tr.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
        prev_c = cls

    alpha = 1.0 / float(length)
    ema = None
    for v in tr:
        ema = v if ema is None else (alpha * v + (1 - alpha) * ema)
    return float(ema)

# -------------------- Public API --------------------

@dataclass
class BracketIntent:
    symbol: str
    side: str  # 'buy' | 'sell'
    qty: int
    parent_type: str  # 'market' | 'limit'
    limit_price: Optional[float]
    tp_price: float
    sl_price: float
    client_id: Optional[str] = None

def _infer_prices_with_atr(symbol: str, side: str):
    """Return (base, tp, sl) using ATR multiples off the latest close."""
    _o, h, l, c = _recent_bars(symbol, ATR_TIMEFRAME)
    atr = _atr_from_ohlc(h, l, c, ATR_LENGTH)
    last = float(c[-1])

    if side == "buy":
        base = last
        tp = last + ATR_MULT_TP_ENTRY * atr
        sl = last - ATR_MULT_SL_ENTRY * atr
    else:
        base = last
        tp = last - ATR_MULT_TP_ENTRY * atr
        sl = last + ATR_MULT_SL_ENTRY * atr

    return _q(base), _q(tp), _q(sl)

def submit_bracket_entry(
    symbol: str,
    side: str,
    qty: int,
    *,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    limit_price: Optional[float] = None,
    client_id: Optional[str] = None,
    parent_type: str = None,  # 'market' or 'limit'
):
    """
    Core submitter. If USE_ATR_ENTRY=1 and any of tp/sl/limit are None,
    we infer them via ATR from latest close.
    """
    if qty <= 0:
        raise ValueError("qty must be > 0")

    if parent_type is None:
        parent_type = DEFAULT_PARENT_TYPE

    base_q = None
    tp_q = tp_price
    sl_q = sl_price
    lim_q = limit_price

    if USE_ATR_ENTRY and (tp_q is None or sl_q is None or (parent_type == "limit" and lim_q is None)):
        base_q, tp_q2, sl_q2 = _infer_prices_with_atr(symbol, side)
        tp_q = tp_q if tp_q is not None else tp_q2
        sl_q = sl_q if sl_q is not None else sl_q2
        lim_q = lim_q if lim_q is not None else base_q
    else:
        if tp_q is None or sl_q is None:
            raise ValueError("tp_price and sl_price are required when USE_ATR_ENTRY=0")
        tp_q = _q(tp_q)
        sl_q = _q(sl_q)
        if parent_type == "limit":
            if lim_q is None:
                raise ValueError("limit_price required for parent_type='limit'")
            lim_q = _q(lim_q)

    payload = {
        "symbol": symbol.upper(),
        "side": side,
        "qty": str(int(qty)),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp_q:.2f}"},
        "stop_loss": {"stop_price": f"{sl_q:.2f}"},
        "extended_hours": EXTENDED_HOURS,
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
        "type": parent_type,
    }
    if parent_type == "limit":
        payload["limit_price"] = f"{lim_q:.2f}"

    print(f"{datetime.utcnow().isoformat()}Z INFO bracket_helper | submit bracket: {payload}")
    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = _http("POST", url, data=json.dumps(payload))
    if r.status_code >= 300:
        print(f"{datetime.utcnow().isoformat()}Z ERROR bracket_helper | submit failed: {r.status_code} {r.text}")
        r.raise_for_status()
    return r.json()

# ---------- Legacy-friendly shim (keep older imports working) ----------

def submit_bracket(
    symbol: str,
    side: str,
    qty: int,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    limit_price: Optional[float] = None,
    client_id: Optional[str] = None,
):
    """
    Back-compat wrapper used by older callers:
        from services.bracket_helper import submit_bracket
    """
    return submit_bracket_entry(
        symbol=symbol,
        side=side,
        qty=qty,
        tp_price=tp_price,
        sl_price=sl_price,
        limit_price=limit_price,
        client_id=client_id,
    )

def list_open_orders(symbol: Optional[str] = None):
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbol:
        url += f"&symbols={symbol}"
    r = _http("GET", url)
    r.raise_for_status()
    return r.json()
