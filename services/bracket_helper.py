# services/bracket_helper.py
from __future__ import annotations

import os
from math import floor
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

import requests

# -------------------------------
# Alpaca REST (account/quotes/bars)
# -------------------------------
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
if API_KEY and API_SECRET:
    S.headers.update({
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
    })

def _get_account_json() -> dict:
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------------------
# yfinance fallbacks
# -------------------------------
def _yf_last_price(symbol: str) -> float:
    import yfinance as yf
    # fast path: 1d 1m
    df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=False, prepost=True)
    if df is not None and len(df):
        return float(df["Close"].iloc[-1])
    # slower but robust
    df = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=False, prepost=True)
    if df is not None and len(df):
        return float(df["Close"].iloc[-1])
    raise RuntimeError(f"yfinance: unable to fetch last price for {symbol}")

def _get_last_price(symbol: str) -> float:
    # Try Alpaca quotes -> trades -> yfinance
    try:
        r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest", timeout=20)
        if r.status_code == 200:
            j = r.json()
            if "quote" in j and j["quote"]:
                q = j["quote"]
                ap = float(q.get("ap") or 0)
                bp = float(q.get("bp") or 0)
                if ap > 0:
                    return ap
                if bp > 0:
                    return bp
        # trades fallback (still Alpaca)
        r2 = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest", timeout=20)
        if r2.status_code == 200:
            return float(r2.json()["trade"]["p"])
        # else fall through to yfinance
    except Exception:
        pass
    # yfinance fallback
    return _yf_last_price(symbol)

# -------------------------------
# Bars / ATR calculation
# -------------------------------
def _fetch_bars_alpaca(symbol: str, start: datetime, end: datetime, timeframe: str):
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
    }
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars?timeframe={timeframe}"
    r = S.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("bars", [])

def _fetch_bars_yf(symbol: str, lookback_days: int, timeframe: str):
    import yfinance as yf
    # map timeframe
    tf_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "60m", "1Day": "1d"}
    yf_int = tf_map.get(timeframe, "5m")
    period = f"{max(lookback_days, 2)}d"
    df = yf.download(symbol, period=period, interval=yf_int, progress=False, auto_adjust=False, prepost=True)
    if df is None or len(df) == 0:
        return []
    # adapt to Alpaca-like dicts
    out = []
    for ts, row in df.iterrows():
        out.append({
            "t": ts.to_pydatetime().astimezone(timezone.utc).isoformat(),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row.get("Volume", 0)),
        })
    return out

# feed selection: auto (default), alpaca, yf
ATR_FEED = os.getenv("ATR_FEED", "auto").lower()

def _fetch_bars(symbol: str, lookback_days: int, timeframe: str = "5Min"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    # Respect explicit feed selection
    if ATR_FEED == "alpaca":
        return _fetch_bars_alpaca(symbol, start, end, timeframe)
    if ATR_FEED == "yf":
        return _fetch_bars_yf(symbol, lookback_days, timeframe)

    # auto: try alpaca, fall back to yfinance
    try:
        return _fetch_bars_alpaca(symbol, start, end, timeframe)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403, 404, 429, 500, 503):
            return _fetch_bars_yf(symbol, lookback_days, timeframe)
        raise
    except Exception:
        return _fetch_bars_yf(symbol, lookback_days, timeframe)

# small ATR cache to avoid hammering APIs during loops
_ATR_CACHE: Dict[Tuple[str,int,str,int], Tuple[float, float]] = {}
# key: (symbol, period, timeframe, lookback_days) -> (atr, cached_epoch_secs)
_ATR_TTL = int(os.getenv("ATR_CACHE_TTL_SEC", "60"))

def _atr(symbol: str, period: int, lookback_days: int, timeframe: str = "5Min") -> float:
    import time, pandas as pd
    now = time.time()
    key = (symbol, period, timeframe, lookback_days)
    if key in _ATR_CACHE:
        val, ts = _ATR_CACHE[key]
        if now - ts <= _ATR_TTL:
            return val

    bars = _fetch_bars(symbol, lookback_days, timeframe)
    if not bars:
        _ATR_CACHE[key] = (0.0, now)
        return 0.0

    df = pd.DataFrame(bars)
    # expected columns: t,o,h,l,c,v
    df["h_l"]  = df["h"] - df["l"]
    df["h_pc"] = (df["h"] - df["c"].shift(1)).abs()
    df["l_pc"] = (df["l"] - df["c"].shift(1)).abs()
    tr = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    out = float(atr) if atr and atr > 0 else 0.0
    _ATR_CACHE[key] = (out, now)
    return out

# -------------------------------
# Env config for exits & sizing
# -------------------------------
ATR_ON         = os.getenv("ATR_EXITS", "0") == "1"
ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK   = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP    = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL    = float(os.getenv("ATR_MULT_SL", "1.0"))

MAX_TP_PCT     = float(os.getenv("MAX_TP_PCT", "0.015"))
MAX_SL_PCT     = float(os.getenv("MAX_SL_PCT", "0.015"))

USE_DYNAMIC_SIZE      = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE    = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))
MIN_QTY               = int(os.getenv("MIN_QTY", "1"))
MAX_QTY               = int(os.getenv("MAX_QTY", "5"))
MIN_SL_DOLLARS        = float(os.getenv("MIN_SL_DOLLARS", "0.05"))

def _cap_pct(base: float, pct: float, up: bool) -> float:
    return base * (1 + pct) if up else base * (1 - pct)

def _atr_exits(symbol: str, side: str, last_price: float) -> Tuple[float, float]:
    if ATR_ON:
        atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK)
        if atr > 0:
            if side == "buy":
                tp = last_price + ATR_MULT_TP * atr
                sl = last_price - max(ATR_MULT_SL * atr, MIN_SL_DOLLARS)
            else:
                tp = last_price - ATR_MULT_TP * atr
                sl = last_price + max(ATR_MULT_SL * atr, MIN_SL_DOLLARS)

            tp_cap_up   = _cap_pct(last_price, MAX_TP_PCT, up=True)
            tp_cap_down = _cap_pct(last_price, MAX_TP_PCT, up=False)
            sl_cap_up   = _cap_pct(last_price, MAX_SL_PCT, up=True)
            sl_cap_down = _cap_pct(last_price, MAX_SL_PCT, up=False)

            if side == "buy":
                tp = min(tp, tp_cap_up)
                sl = max(sl, sl_cap_down)
            else:
                tp = max(tp, tp_cap_down)
                sl = min(sl, sl_cap_up)
            return (round(tp, 2), round(sl, 2))

    # fallback only-percentage
    if side == "buy":
        return (round(_cap_pct(last_price, MAX_TP_PCT, True), 2),
                round(_cap_pct(last_price, MAX_SL_PCT, False), 2))
    else:
        return (round(_cap_pct(last_price, MAX_TP_PCT, False), 2),
                round(_cap_pct(last_price, MAX_SL_PCT, True), 2))

def _risk_per_share(symbol: str, side: str, last_price: float) -> float:
    tp, sl = _atr_exits(symbol, side, last_price)
    return max(abs(last_price - sl), MIN_SL_DOLLARS)

def _compute_dynamic_qty(symbol: str, side: str, last_price: float) -> int:
    acct = _get_account_json()
    equity = float(acct.get("equity", 0))
    buying_power = float(acct.get("buying_power", 0))

    risk_budget = equity * RISK_PCT_PER_TRADE
    rps = _risk_per_share(symbol, side, last_price)
    if rps <= 0:
        return MIN_QTY

    qty = int(floor(risk_budget / rps))
    if qty < MIN_QTY:
        qty = MIN_QTY
    if qty > MAX_QTY:
        qty = MAX_QTY

    max_by_bp = int(floor(buying_power / max(last_price, 0.01)))
    if max_by_bp < 1:
        return 0
    qty = min(qty, max_by_bp)
    return qty

def submit_bracket(
    symbol: str,
    side: str,                 # "buy" | "sell"
    strength: float,
    qty: Optional[int] = None,
    api: Optional[object] = None,      # kept for compatibility
    last_price: Optional[float] = None,
    client_id: Optional[str] = None,
    time_in_force: str = "day",
    order_type: str = "market",
    extended_hours: bool = False,
) -> Dict:
    if last_price is None:
        last_price = _get_last_price(symbol)

    if USE_DYNAMIC_SIZE:
        dyn_qty = _compute_dynamic_qty(symbol, side, last_price)
        if dyn_qty <= 0:
            raise RuntimeError(f"dynamic sizing -> qty=0 for {symbol}, check buying power / params")
        use_qty = dyn_qty
    else:
        use_qty = int(qty or 1)

    tp, sl = _atr_exits(symbol, side, last_price)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(use_qty),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss":   {"stop_price": f"{sl:.2f}"},
        "extended_hours": bool(extended_hours),
        "type": order_type,
    }
    if client_id:
        payload["client_order_id"] = client_id

    print(
        f"submit bracket {'(atr(period=%s,tp×%s,sl×%s))' % (ATR_PERIOD, ATR_MULT_TP, ATR_MULT_SL) if ATR_ON else ''}: "
        f"{payload}"
    )

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=20)
    if r.status_code >= 400:
        try:
            print("submit failed:", r.text)
        finally:
            r.raise_for_status()
    return r.json()
