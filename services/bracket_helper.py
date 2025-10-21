# services/bracket_helper.py
from __future__ import annotations
import os, time, json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, List

import requests

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' for paper/free

ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.2"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
})

def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    print(f"{ts} INFO bracket_helper | {msg}")

def _http(method: str, url: str, **kwargs):
    for i in range(5):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception:
            time.sleep(min(2**i, 8))
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def _q(price: float, tick: Decimal = Decimal("0.01")) -> float:
    return float(Decimal(str(price)).quantize(tick, rounding=ROUND_HALF_UP))

def _clock_is_open() -> bool:
    r = _http("GET", f"{ALPACA_BASE_URL}/v2/clock")
    r.raise_for_status()
    return bool(r.json().get("is_open", False))

def _latest_trade_price(symbol: str) -> float:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    feeds = [DATA_FEED] + ([] if DATA_FEED.lower()=="iex" else ["iex"])
    last_err = None
    for feed in feeds:
        try:
            r = _http("GET", url, params={"feed": feed})
            if r.status_code == 403:
                last_err = RuntimeError("403 feed not allowed")
                continue
            r.raise_for_status()
            js = r.json()
            px = js.get("trade", {}).get("p")
            if px is None:
                raise RuntimeError("no trade price in response")
            return float(px)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"latest trade failed for {symbol}: {last_err}")

def _build_tp_sl(side: str, ref: float, atr: float) -> Tuple[float,float]:
    if side.lower() == "buy":
        tp = ref + ATR_MULT_TP * atr
        sl = ref - ATR_MULT_SL * atr
    else:
        tp = ref - ATR_MULT_TP * atr
        sl = ref + ATR_MULT_SL * atr
    return tp, sl

def _clamp_to_rules(side: str, tp: float, sl: float, base: float, tick: float=0.01) -> Tuple[float,float]:
    if side.lower() == "buy":
        min_tp = base + tick
        max_sl = base - tick
        tp = max(tp, min_tp)
        sl = min(sl, max_sl)
    else:
        max_tp = base - tick
        min_sl = base + tick
        tp = min(tp, max_tp)
        sl = max(sl, min_sl)
    return _q(tp), _q(sl)

def submit_bracket(symbol: str, side: str, qty: int, *, prefer_limit_when_closed: bool = True,
                   ref_atr: Optional[float] = None) -> dict:
    is_open = _clock_is_open()
    base = _latest_trade_price(symbol)
    atr = ref_atr if (ref_atr and ref_atr > 0) else max(0.0025 * base, 0.05)

    raw_tp, raw_sl = _build_tp_sl(side, base, atr)
    tp, sl = _clamp_to_rules(side, raw_tp, raw_sl, base)

    payload = {
        "symbol": symbol,
        "side": side.lower(),
        "qty": str(int(qty)),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
    }

    if is_open:
        payload["type"] = "market"
    else:
        payload["type"] = "limit" if prefer_limit_when_closed else "market"
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

# NEW: provide list_open_orders for executor_bracket compatibility
def list_open_orders(symbols: Optional[List[str]] = None) -> list[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(s.upper() for s in symbols)
    r = _http("GET", url)
    r.raise_for_status()
    return r.json()

# Back-compat alias kept:
def submit_bracket_entry(symbol: str, side: str, qty: int, prefer_limit_when_closed: bool = True) -> dict:
    return submit_bracket(symbol, side, qty, prefer_limit_when_closed=prefer_limit_when_closed)
