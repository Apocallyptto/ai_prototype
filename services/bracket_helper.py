# services/bracket_helper.py
# Drop-in helper for submitting Alpaca bracket orders with optional ATR entries
# and hard clamps to satisfy Alpaca parent/TP/SL constraints.

from __future__ import annotations

import os
import json
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional, List

import requests

# -------- ENV --------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", os.getenv("ALPACA_FEED", "iex"))  # "iex" for paper
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "0").lower() in {"1", "true", "yes"}

# ATR config
USE_ATR_ENTRY = os.getenv("USE_ATR_ENTRY", "0").lower() in {"1", "true", "yes"}
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_MULT_TP_ENTRY = float(os.getenv("ATR_MULT_TP_ENTRY", "1.3"))
ATR_MULT_SL_ENTRY = float(os.getenv("ATR_MULT_SL_ENTRY", "1.0"))

# parent type: "market" (default) or "limit"
PARENT_TYPE = os.getenv("PARENT_TYPE", "market").lower()  # "market" or "limit"

# PowerShell-friendly print prefix
def _now():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

# HTTP helper
def _http(method: str, url: str, **kw) -> requests.Response:
    headers = kw.pop("headers", {})
    headers.update(
        {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
            "Content-Type": "application/json",
        }
    )
    return requests.request(method, url, headers=headers, timeout=30, **kw)

# -------- Market data helpers --------
def _latest_price(symbol: str) -> float:
    """Return a reasonable base price for market-parent constraints."""
    # Preferred: latest quote midpoint
    url_q = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = _http("GET", url_q, params={"feed": ALPACA_DATA_FEED})
    if r.status_code < 300:
        q = r.json().get("quote") or {}
        bp = q.get("bp")
        ap = q.get("ap")
        if bp and ap:
            return (float(bp) + float(ap)) / 2.0
        if ap:
            return float(ap)
        if bp:
            return float(bp)
    # Fallback: latest trade
    url_t = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    r = _http("GET", url_t, params={"feed": ALPACA_DATA_FEED})
    r.raise_for_status()
    px = r.json().get("trade", {}).get("p")
    if px is None:
        raise RuntimeError(f"No price available for {symbol}")
    return float(px)

def _recent_bars(symbol: str, timeframe: str, lookback_days: int = 5) -> List[dict]:
    """Fetch recent bars for ATR; returns list of bar dicts (o,h,l,c)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (now - timedelta(days=lookback_days)).isoformat()
    end = now.isoformat()
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "split",
        "feed": ALPACA_DATA_FEED,
    }
    r = _http("GET", url, params=params)
    r.raise_for_status()
    return r.json().get("bars", [])

def _atr_from_bars(bars: List[dict], length: int) -> float:
    """
    Wilder's ATR from list of bars (expects keys: o,h,l,c).
    Falls back to simple TR average if not enough for Wilder seed.
    """
    if len(bars) < max(2, length + 1):
        # Not enough bars; try simple average of TR
        trs = []
        prev_close = None
        for b in bars:
            h = float(b["h"]); l = float(b["l"]); c = float(b["c"])
            if prev_close is None:
                tr = h - l
            else:
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        return float(sum(trs[-length:]) / max(1, min(length, len(trs))))
    # Wilder ATR
    trs = []
    prev_close = None
    for b in bars:
        h = float(b["h"]); l = float(b["l"]); c = float(b["c"])
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    # initial ATR = simple avg of first 'length' TRs
    atr = sum(trs[:length]) / length
    # Wilder smoothing for the rest
    for tr in trs[length:]:
        atr = (atr * (length - 1) + tr) / length
    return float(atr)

# -------- Price inference --------
def _infer_prices_with_atr(symbol: str, side: str) -> Tuple[float, float, float]:
    """
    Returns (base_price, tp_suggested, sl_suggested) using ATR multipliers.
    base_price is the current market mid (market parent) or last close (as a proxy).
    """
    bars = _recent_bars(symbol, ATR_TIMEFRAME, lookback_days=5)
    if not bars:
        # No bars; use current price for both and tiny offsets
        base = _latest_price(symbol)
        return base, base + (0.10 if side == "buy" else -0.10), base - (0.10 if side == "buy" else -0.10)

    atr = _atr_from_bars(bars, ATR_LENGTH)
    base = _latest_price(symbol)  # stay consistent with parent market base

    if side == "buy":
        tp = base + ATR_MULT_TP_ENTRY * atr
        sl = base - ATR_MULT_SL_ENTRY * atr
    else:
        tp = base - ATR_MULT_TP_ENTRY * atr
        sl = base + ATR_MULT_SL_ENTRY * atr

    return float(base), float(tp), float(sl)

# -------- Formatting helpers --------
def _q(x: float, tick: float = 0.01) -> float:
    """Round to nearest valid tick."""
    return round(round(float(x) / tick) * tick, 2)

def _ensure_parent_limits(symbol: str, side: str, parent_type: str,
                          tp_q: float, sl_q: float, limit_price: Optional[float]) -> Tuple[float, float, float]:
    """
    Enforce Alpaca constraints relative to base price:
       - BUY:  SL <= base - 0.01, TP >= base + 0.01
       - SELL: SL >= base + 0.01, TP <= base - 0.01
    Returns (tp_q, sl_q, base_px_rounded)
    """
    # Determine base
    if parent_type == "market":
        base = _latest_price(symbol)
    else:
        if limit_price is None:
            raise ValueError("limit_price required for parent_type='limit'")
        base = float(limit_price)

    tick = 0.01
    # Add a tiny cushion so we never sit exactly on the boundary
    eps = 2 * tick  # 2 cents cushion

    if side == "buy":
        sl_q = min(sl_q, base - tick - eps)
        tp_q = max(tp_q, base + tick + eps)
        # As a safety, if still inverted (due to extreme prices), force a minimal band
        if sl_q >= base - tick:
            sl_q = base - tick - eps
        if tp_q <= base + tick:
            tp_q = base + tick + eps
    else:  # sell
        sl_q = max(sl_q, base + tick + eps)
        tp_q = min(tp_q, base - tick - eps)
        if sl_q <= base + tick:
            sl_q = base + tick + eps
        if tp_q >= base - tick:
            tp_q = base - tick - eps

    return _q(tp_q), _q(sl_q), _q(base)

# -------- Public API --------
def submit_bracket_entry(
    symbol: str,
    side: str,
    qty: int = 1,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    limit_price: Optional[float] = None,
    client_id: Optional[str] = None,
    parent_type: Optional[str] = None,
):
    """
    Submit a bracket order to Alpaca.
    - If USE_ATR_ENTRY=1 and either tp/sl is missing, build ATR-based targets.
    - Enforce Alpaca constraints (market vs limit parent) by clamping TP/SL.
    """
    parent_type = (parent_type or PARENT_TYPE).lower()
    side = side.lower()
    qty = int(qty)

    # 1) Suggest prices (ATR) if requested
    base_q = None
    if USE_ATR_ENTRY and (tp_price is None or sl_price is None):
        base_q, tp_s, sl_s = _infer_prices_with_atr(symbol, side)
        if tp_price is None:
            tp_price = tp_s
        if sl_price is None:
            sl_price = sl_s

    # Parent limit fallback for limit parent
    if parent_type == "limit" and limit_price is None:
        limit_price = _latest_price(symbol)

    # 2) Enforce constraints around the true base
    tp_q, sl_q, base_used = _ensure_parent_limits(
        symbol=symbol,
        side=side,
        parent_type=parent_type,
        tp_q=float(tp_price),
        sl_q=float(sl_price),
        limit_price=float(limit_price) if limit_price is not None else None,
    )

    # For limit parent, round the parent price too
    lim_q = _q(limit_price) if (parent_type == "limit" and limit_price is not None) else None

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
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

    print(f"{_now().isoformat()} INFO bracket_helper | submit bracket: {payload}")

    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = _http("POST", url, data=json.dumps(payload))
    if r.status_code >= 300:
        print(f"{_now().isoformat()} ERROR bracket_helper | submit failed: {r.status_code} {r.text}")
        r.raise_for_status()
    return r.json()

# ---- Back-compat shim (legacy import) ----
def submit_bracket(
    symbol: str,
    side: str,
    qty: int,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    limit_price: Optional[float] = None,
    client_id: Optional[str] = None,
):
    """Legacy function name preserved for older callers."""
    return submit_bracket_entry(
        symbol=symbol,
        side=side,
        qty=qty,
        tp_price=tp_price,
        sl_price=sl_price,
        limit_price=limit_price,
        client_id=client_id,
        parent_type=PARENT_TYPE,
    )
