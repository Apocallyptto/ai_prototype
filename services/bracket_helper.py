# services/bracket_helper.py
from __future__ import annotations
import os, json, time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, Any, Tuple, List

import requests
import pandas as pd

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

# Strategy knobs (align with your jobs)
ATR_LENGTH     = int(os.getenv("ATR_LENGTH", "14"))
ATR_TIMEFRAME  = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_MULT_TP    = float(os.getenv("ATR_MULT_TP", "1.2"))
ATR_MULT_SL    = float(os.getenv("ATR_MULT_SL", "1.0"))
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1","true","yes","y"}
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO").upper()

# Sizing knobs (optional)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.05"))  # fraction of 1-share ATR risk fallback sizing
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", "10"))

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} bracket_helper | {msg}", flush=True)

def q2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def http(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            wait = min(2 ** attempt, 8)
            log(f"HTTP error {e} -> retry {wait}s", level="WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

# ---------- Venue/Data ----------

def get_clock() -> dict:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/clock")
    r.raise_for_status()
    return r.json()

def is_open_now() -> bool:
    try:
        return bool(get_clock().get("is_open"))
    except Exception as e:
        log(f"clock check failed: {e}", level="WARN")
        return True

def _get_with_feed(url: str, params: Dict[str, Any]) -> requests.Response:
    p = dict(params)
    p["feed"] = ALPACA_DATA_FEED
    r = http("GET", url, params=p)
    if r.status_code == 403 and ALPACA_DATA_FEED.lower() != "iex":
        p["feed"] = "iex"
        r = http("GET", url, params=p)
    return r

def latest_quote(symbol: str) -> Optional[dict]:
    r = _get_with_feed(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest", {})
    if r.status_code >= 300:
        log(f"quote fetch {symbol} -> {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("quote") or js
    except Exception as e:
        log(f"quote parse {symbol}: {e}", level="WARN")
        return None

def latest_trade(symbol: str) -> Optional[dict]:
    r = _get_with_feed(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest", {})
    if r.status_code >= 300:
        return None
    try:
        js = r.json()
        return js.get("trade") or js
    except Exception:
        return None

def latest_bar_close(symbol: str, timeframe: str = ATR_TIMEFRAME) -> Optional[float]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5)
    r = _get_with_feed(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", {
        "timeframe": timeframe, "start": start.isoformat(), "end": now.isoformat(),
        "limit": 1, "adjustment": "split",
    })
    if r.status_code >= 300:
        return None
    try:
        js = r.json()
        bars = js.get("bars") or []
        if not bars: return None
        return float(bars[-1].get("c") or bars[-1].get("close") or 0)
    except Exception:
        return None

def bars_df(symbol: str, timeframe: str = ATR_TIMEFRAME, lookback: int = 400) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=10)
    feeds = [ALPACA_DATA_FEED] + ([] if ALPACA_DATA_FEED.lower() == "iex" else ["iex"])
    last_err = None
    for feed in feeds:
        try:
            r = http("GET", f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars", params={
                "timeframe": timeframe, "start": start.isoformat(), "end": now.isoformat(),
                "limit": lookback, "adjustment": "split", "feed": feed,
            })
            if r.status_code == 403:
                last_err = RuntimeError("403 feed")
                continue
            r.raise_for_status()
            js = r.json(); bars = js.get("bars", [])
            if not bars: raise RuntimeError("no bars")
            df = pd.DataFrame(bars).rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True); df.set_index("ts", inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            last_err = e
            log(f"bars fetch fail {symbol} feed={feed}: {e}", level="WARN")
    raise RuntimeError(f"bars failed for {symbol}: {last_err}")

def compute_atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> float:
    hi, lo, cl = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_close = cl.shift(1)
    tr = pd.concat([(hi-lo), (hi-prev_close).abs(), (lo-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return float(atr.dropna().iloc[-1])

def infer_entry_ref(symbol: str, side: str, is_open: bool) -> float:
    q = latest_quote(symbol) or {}
    last = float(q.get("lp") or q.get("last") or 0)
    ask  = float(q.get("ap") or q.get("ask_price") or 0)
    bid  = float(q.get("bp") or q.get("bid_price") or 0)

    ref = last
    if ref <= 0:
        ref = (ask if side == "buy" else bid) or (bid if side == "sell" else ask)
    if ref <= 0:
        t = latest_trade(symbol) or {}
        ref = float(t.get("p") or t.get("price") or 0)
    if ref <= 0:
        bc = latest_bar_close(symbol) or 0
        ref = float(bc)
    if ref <= 0:
        raise RuntimeError("cannot infer entry reference price")
    return q2(ref)

# ---------- Risk/Sizing ----------

def account_cash() -> float:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/account")
    r.raise_for_status()
    js = r.json()
    try:
        return float(js.get("cash") or js.get("portfolio_value") or 0)
    except Exception:
        return 0.0

def compute_qty_from_risk(symbol: str, side: str, entry_ref: float, atr: float) -> int:
    """
    Very simple ATR-based sizing:
      risk_per_share = ATR_MULT_SL * atr
      target_risk    = RISK_PER_TRADE * 1 * ATR (fallback notion) -> keep small on paper
      qty            = max(1, floor(target_risk / risk_per_share))
    You can replace with your own equity-based sizing.
    """
    risk_per_share = max(0.01, ATR_MULT_SL * atr)
    target_risk = max(0.01, RISK_PER_TRADE * atr)   # small for paper mode
    qty = int(max(1, target_risk // risk_per_share))
    return qty if qty >= 1 else 1

# ---------- Dedupe / Wash-trade checks (Alpaca-side only) ----------

def list_open_orders(symbol: Optional[str] = None) -> List[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true"
    if symbol: url += f"&symbols={symbol}"
    r = http("GET", url); r.raise_for_status()
    return r.json()

def washed_or_duplicate(symbol: str, side: str, lookback_sec: int = 600) -> bool:
    """
    Lightweight guard: if there is an open order for the same symbol
    on the opposite side created within 'lookback_sec', skip placing new.
    Also skip if identical side open exists (basic dedupe).
    """
    orders = list_open_orders(symbol)
    now = datetime.now(timezone.utc)
    for o in orders:
        if o.get("symbol") != symbol: continue
        o_side = o.get("side", "")
        created = o.get("created_at") or o.get("submitted_at") or ""
        try:
            ct = datetime.fromisoformat(created.replace("Z","+00:00"))
        except Exception:
            ct = now
        if (now - ct).total_seconds() <= lookback_sec:
            if o_side != side:  # opposite side within window
                return True
            if o_side == side:  # same side already open
                return True
    return False

# ---------- Submit bracket ----------

def submit_bracket_entry(symbol: str, side: str, qty: Optional[int] = None) -> dict:
    """
    Submit ATR-aware bracket (market if RTH; limit if ETH).
    If qty is None, auto-size with a tiny ATR-risk model.
    """
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    if washed_or_duplicate(symbol, side):
        raise RuntimeError(f"dedupe/wash guard tripped for {symbol} {side}")

    is_open = is_open_now()
    df = bars_df(symbol)
    atr = compute_atr(df)
    entry_ref = infer_entry_ref(symbol, side, is_open)

    # decide qty
    q = int(qty) if qty else compute_qty_from_risk(symbol, side, entry_ref, atr)

    # build TP/SL around entry_ref
    if side == "buy":
        tp = q2(entry_ref + ATR_MULT_TP * atr)
        sl = q2(entry_ref - ATR_MULT_SL * atr)
    else:
        tp = q2(entry_ref - ATR_MULT_TP * atr)
        sl = q2(entry_ref + ATR_MULT_SL * atr)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(q),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": EXTENDED_HOURS,
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
        "type": "market" if is_open else "limit",
    }
    if payload["type"] == "limit":
        payload["limit_price"] = f"{q2(entry_ref):.2f}"

    log(f"submit bracket: {payload}")
    r = http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"submit failed: {r.status_code} {r.text}", level="ERROR")
        r.raise_for_status()
    js = r.json()
    print(json.dumps(js, indent=2))
    return js
