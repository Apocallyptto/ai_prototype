# services/bracket_helper.py
from __future__ import annotations
import os, json, time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, Any, List

import requests

from lib.atr_utils import atr_targets, get_last_price  # ATR-based targets

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
})

ENTRY_TYPE = os.getenv("BRACKET_ENTRY_TYPE", "limit").lower()  # 'market' or 'limit'
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1", "true", "yes", "y"}

# ATR controls (enabled by default)
ATR_USE = os.getenv("USE_ATR_ENTRY", "1").lower() in {"1", "true", "yes", "y"}
ATR_TF = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_LEN = int(os.getenv("ATR_LENGTH", "14"))
ATR_MULT_TP_ENTRY = float(os.getenv("ATR_MULT_TP_ENTRY", os.getenv("ATR_MULT_TP", "1.2")))
ATR_MULT_SL_ENTRY = float(os.getenv("ATR_MULT_SL_ENTRY", os.getenv("ATR_MULT_SL", "1.0")))


def _q(x: float, tick: Decimal = Decimal("0.01")) -> float:
    return float(Decimal(str(x)).quantize(tick, rounding=ROUND_HALF_UP))


def _http(method: str, url: str, **kwargs):
    for attempt in range(5):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")


def list_open_orders(symbol: Optional[str] = None) -> List[dict]:
    """Return open orders; if symbol is provided, filter on server side."""
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbol:
        url += f"&symbols={symbol}"
    r = _http("GET", url)
    r.raise_for_status()
    return r.json()


def submit_bracket_entry(symbol: str,
                         side: str,                 # 'buy' or 'sell'
                         qty: int,
                         tp_price: Optional[float], # can be None -> ATR
                         sl_price: Optional[float], # can be None -> ATR
                         limit_price: Optional[float] = None,  # parent limit if ENTRY_TYPE='limit'
                         client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    If USE_ATR_ENTRY=1, compute TP/SL from ATR w.r.t reference price (limit or last trade).
    Enforces Alpaca base_price rules to avoid 422 errors.
    """
    parent_type = "market" if ENTRY_TYPE == "market" else "limit"

    # Reference price for both ATR targets and parent limit (if 'limit' entry)
    base_price = limit_price if limit_price is not None else get_last_price(symbol)
    base_q = _q(base_price)

    if ATR_USE:
        tp_price, sl_price = atr_targets(
            symbol=symbol,
            side=side,
            ref_price=base_q,
            mult_tp=ATR_MULT_TP_ENTRY,
            mult_sl=ATR_MULT_SL_ENTRY,
            timeframe=ATR_TF,
            length=ATR_LEN,
        )

    # Directional constraints vs base price to satisfy Alpaca validations
    if side == "buy":
        tp_price = max(tp_price, base_q + 0.01)
        sl_price = min(sl_price, base_q - 0.01)
    else:
        tp_price = min(tp_price, base_q - 0.01)
        sl_price = max(sl_price, base_q + 0.01)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(int(qty)),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{_q(tp_price):.2f}"},
        "stop_loss": {"stop_price": f"{_q(sl_price):.2f}"},
        "extended_hours": EXTENDED_HOURS,
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
        "type": parent_type,
    }
    if parent_type == "limit":
        payload["limit_price"] = f"{_q(base_q):.2f}"

    print(f"{datetime.utcnow().isoformat()}Z INFO bracket_helper | submit bracket: {payload}")
    url = f"{ALPACA_BASE_URL}/v2/orders"
    r = _http("POST", url, data=json.dumps(payload))
    if r.status_code >= 300:
        print(f"{datetime.utcnow().isoformat()}Z ERROR bracket_helper | submit failed: {r.status_code} {r.text}")
        r.raise_for_status()
    return r.json()
