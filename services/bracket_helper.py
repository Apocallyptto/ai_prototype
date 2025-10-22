# services/bracket_helper.py
from __future__ import annotations

import os, json, time
from typing import Optional, Literal
from datetime import datetime, timezone
import requests

from lib.atr_utils import compute_atr, atr_tp_sl, round_to_tick

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# RTH/ETH behavior
MODE = os.getenv("MODE", os.getenv("STALE_MODE", "ETH")).upper()  # RTH or ETH
TIME_IN_FORCE = os.getenv("BRACKET_TIF", "day")

# Sizing
DEFAULT_QTY = int(os.getenv("QTY_PER_TRADE", "1"))
USE_ATR_SIZING = os.getenv("USE_ATR_SIZING", "false").lower() in ("1","true","yes")

# ATR parameters
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_TICK = float(os.getenv("ATR_TICK", "0.01"))  # price rounding

# Limit/entry behavior:
# For ETH we place LIMIT parents (safer). For RTH, you can switch to MARKET or keep LIMIT.
ENTRY_TYPE_ETH = os.getenv("ENTRY_TYPE_ETH", "limit").lower()   # 'limit' or 'market'
ENTRY_TYPE_RTH = os.getenv("ENTRY_TYPE_RTH", "limit").lower()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def _entry_type() -> str:
    return ENTRY_TYPE_RTH if MODE == "RTH" else ENTRY_TYPE_ETH

def _latest_price(symbol: str) -> Optional[float]:
    # compute_atr returns (atr, last). Reuse its last component without fetching twice.
    atr, last = compute_atr(symbol, timeframe=ATR_TIMEFRAME)
    return last

def _http(method: str, url: str, **kwargs) -> requests.Response:
    for i in range(4):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            time.sleep(min(2**i, 8))
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def _determine_entry_limit(symbol: str, side: Literal["buy","sell"]) -> Optional[float]:
    """
    Simple entry limiter: use latest price as anchor and put a 0.0 tick offset (you can add spread later).
    """
    atr, last = compute_atr(symbol, timeframe=ATR_TIMEFRAME)
    if not last:
        return None
    # small nudge: for buys a bit above last; for sells a bit below last (0 ticks by default)
    off = 0.0 * ATR_TICK
    px = last + off if side == "buy" else last - off
    return round_to_tick(px, ATR_TICK)

def _size_from_atr(symbol: str, side: str, last: float, atr: float) -> int:
    """
    Optional ATR-based sizing. Default stays at DEFAULT_QTY if USE_ATR_SIZING=false.
    qty ≈ risk_per_trade_usd / atr, clamped by notional caps.
    """
    if not USE_ATR_SIZING:
        return DEFAULT_QTY
    try:
        risk = float(os.getenv("RISK_PER_TRADE_USD", "10"))
        max_notional = float(os.getenv("MAX_POSITION_USD", "500"))
        min_notional = float(os.getenv("MIN_POSITION_USD", "50"))
        raw = int(max(risk / max(atr, 1e-6), 0))
        max_q = int(max_notional // max(last, 1e-6))
        min_q = 1 if last >= min_notional else max(int(min_notional // max(last, 1e-6)), 0)
        qty = max(min(raw, max_q), min_q)
        return max(qty, 1)
    except Exception:
        return DEFAULT_QTY

def submit_bracket(symbol: str, side: Literal["buy","sell"], qty: str) -> dict:
    """
    Submit an ATR-aware bracket parent. Keeps your existing log shape & behavior.
    - Entry type: LIMIT by default (both RTH/ETH), configurable.
    - TP/SL: derived from ATR around the intended entry (limit or market anchor).
    """
    # --- get ATR + last ---
    atr, last = compute_atr(symbol, timeframe=ATR_TIMEFRAME)
    if not last:
        raise RuntimeError(f"{symbol}: cannot fetch last price")
    # size (optional ATR sizing)
    use_qty = str(_size_from_atr(symbol, side, last, atr or 0.0))

    # --- entry type & price ---
    otype = _entry_type()  # 'limit' or 'market'
    if otype == "market":
        entry_px = last
        limit_price = None
    else:
        limit_px = _determine_entry_limit(symbol, side)
        if not limit_px:
            raise RuntimeError(f"{symbol}: cannot determine limit price")
        entry_px = limit_px
        limit_price = limit_px

    # --- dynamic ATR exits ---
    # If ATR missing, fall back to 0.6% bands around entry (very rare after compute_atr)
    if atr and atr > 0:
        tp, sl = atr_tp_sl(entry_px, side, atr)
    else:
        bump = 0.006 * entry_px
        if side == "buy":
            tp, sl = (entry_px + bump, entry_px - bump)
        else:
            tp, sl = (entry_px - bump, entry_px + bump)

    tp = round_to_tick(tp, ATR_TICK)
    sl = round_to_tick(sl, ATR_TICK)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": use_qty,
        "time_in_force": TIME_IN_FORCE,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss":   {"stop_price": f"{sl:.2f}"},
        "extended_hours": (MODE != "RTH"),
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
        "type": "market" if otype == "market" else "limit",
    }
    if limit_price is not None:
        payload["limit_price"] = f"{limit_price:.2f}"

    print(f"{_now()} INFO bracket_helper | submit bracket: {payload}", flush=True)

    r = _http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        print(f"{_now()} INFO bracket_helper | submit failed: {r.status_code} {r.text}", flush=True)
        r.raise_for_status()
    return r.json()
