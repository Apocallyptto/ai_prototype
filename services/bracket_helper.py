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
DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")  # paper/free: iex

# === Exits config ===
# Static (fallback) %
BASE_TP_PCT = float(os.getenv("BASE_TP_PCT", "0.006"))  # 0.6%
BASE_SL_PCT = float(os.getenv("BASE_SL_PCT", "0.004"))  # 0.4%

# ATR toggle & params
ATR_EXITS          = os.getenv("ATR_EXITS", "0").lower() in {"1","true","yes","y"}
ATR_PERIOD         = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS  = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP        = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL        = float(os.getenv("ATR_MULT_SL", "1.0"))
# optional safety caps: keep TP/SL within reasonable % bands
MAX_TP_PCT         = float(os.getenv("MAX_TP_PCT", "0.015"))  # cap at +1.5%
MAX_SL_PCT         = float(os.getenv("MAX_SL_PCT", "0.015"))  # cap at -1.5%

# Bracket orders MUST have extended_hours = False on Alpaca
EXTENDED_HOURS = False

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

# Lazy import ATR utils only when needed
def _safe_last_atr(symbol: str) -> Optional[float]:
    try:
        from lib.atr_utils import last_atr
        return last_atr(symbol, period=ATR_PERIOD, lookback_days=ATR_LOOKBACK_DAYS)
    except Exception as e:
        _log(f"ATR disabled for {symbol}: {e}")
        return None

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} INFO bracket_helper | {msg}")

def _err(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} ERROR bracket_helper | {msg}")

def _http(method: str, url: str, **kwargs) -> requests.Response:
    r = SESSION.request(method, url, timeout=15, **kwargs)
    return r

def _clock_is_open() -> bool:
    try:
        r = _http("GET", f"{ALPACA_BASE_URL}/v2/clock")
        r.raise_for_status()
        return bool(r.json().get("is_open"))
    except Exception as e:
        _err(f"clock check failed: {e}")
        return False

def _latest_trade_price(symbol: str) -> float:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": DATA_FEED}
    r = _http("GET", url, params=params)
    try:
        r.raise_for_status()
        p = r.json().get("trade", {}).get("p")
        if p is None:
            raise ValueError("missing trade price")
        return float(p)
    except Exception as e:
        _err(f"latest trade fetch failed for {symbol}: {e}")
        raise

def _q(x: float) -> float:
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _exits_static(side: str, base: float) -> Tuple[float, float]:
    if side.lower() == "buy":
        tp = base * (1.0 + BASE_TP_PCT)
        sl = base * (1.0 - BASE_SL_PCT)
    else:
        tp = base * (1.0 - BASE_TP_PCT)
        sl = base * (1.0 + BASE_SL_PCT)
    return _q(tp), _q(sl)

def _exits_atr(side: str, base: float, atr_val: float) -> Tuple[float, float]:
    # TP/SL in absolute terms from ATR multipliers
    if side.lower() == "buy":
        tp = base + ATR_MULT_TP * atr_val
        sl = base - ATR_MULT_SL * atr_val
        # apply safety caps vs base %
        tp = min(tp, base * (1.0 + MAX_TP_PCT))
        sl = max(sl, base * (1.0 - MAX_SL_PCT))
    else:
        tp = base - ATR_MULT_TP * atr_val
        sl = base + ATR_MULT_SL * atr_val
        tp = max(tp, base * (1.0 - MAX_TP_PCT))
        sl = min(sl, base * (1.0 + MAX_SL_PCT))
    return _q(tp), _q(sl)

def _compute_exits(side: str, base: float, symbol: str) -> Tuple[float, float, str]:
    if ATR_EXITS:
        atrv = _safe_last_atr(symbol)
        if atrv and atrv > 0:
            tp, sl = _exits_atr(side, base, atrv)
            return tp, sl, f"atr(period={ATR_PERIOD},tp×{ATR_MULT_TP},sl×{ATR_MULT_SL})"
    tp, sl = _exits_static(side, base)
    return tp, sl, "static_pct"

# ---------- Public helpers (backward-compatible) ----------

def list_open_orders(symbols: Optional[List[str]] = None) -> List[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbols:
        url += "&symbols=" + ",".join(s.upper() for s in symbols)
    r = _http("GET", url); r.raise_for_status()
    return r.json()

# OLD signature used by services.executor_bracket:
# submit_bracket(symbol, side, qty, prefer_limit_when_closed=True, ref_atr=None)
def submit_bracket(symbol: str,
                   side: str,
                   qty: Optional[int],
                   *,
                   prefer_limit_when_closed: bool = True,
                   ref_atr: Optional[float] = None) -> dict:
    qty_int = int(qty or 1)
    is_open = _clock_is_open()
    base = _latest_trade_price(symbol)
    tp, sl, mode = _compute_exits(side, base, symbol)

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side.lower(),
        "qty": str(qty_int),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,  # Alpaca rule
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

    _log(f"submit bracket ({mode}): {payload}")
    r = _http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        _log(f"submit failed: {r.status_code} {r.text}")
        r.raise_for_status()
    return r.json()

# NEW-style helper (SDK client & known price)
def submit_bracket_v2(api,
                      symbol: str,
                      side: str,
                      qty: int | float,
                      last_price: float,
                      client_id: str) -> dict:
    tp, sl, mode = _compute_exits(side, float(last_price), symbol)
    payload = {
        "symbol": symbol,
        "side": side.lower(),
        "type": "limit",
        "qty": str(int(qty or 1)),
        "time_in_force": "day",
        "order_class": "bracket",
        "limit_price": f"{float(last_price):.2f}",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,
        "client_order_id": client_id,
    }
    _log(f"submit bracket (v2,{mode}): {payload}")
    try:
        return api.submit_order(**payload)
    except Exception as e:
        try:
            msg = getattr(e, "response", None)
            if msg is not None:
                _log(f"submit failed: {msg.text}")
        except Exception:
            pass
        _err(f"submit failed: {e}")
        raise

def submit_bracket_entry(symbol: str, side: str, qty: Optional[int], prefer_limit_when_closed: bool = True) -> dict:
    return submit_bracket(symbol, side, qty, prefer_limit_when_closed=prefer_limit_when_closed)
