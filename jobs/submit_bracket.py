# jobs/submit_bracket.py
from __future__ import annotations
import os, sys, json, time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List

import requests
import pandas as pd

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in {"1","true","yes","y"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ATR knobs (aligns with manage_exits)
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.2"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))

# Optional pad to nudge ETH limit price a tiny bit
LIMIT_PAD_CENTS = int(os.getenv("LIMIT_PAD_CENTS", "0"))  # default 0; set 1-3 if needed

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
        print(f"{ts} {level} submit_bracket | {msg}", flush=True)

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

def q2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# -------- Market clock --------

def get_clock() -> dict:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/clock")
    r.raise_for_status()
    return r.json()

def is_open_now() -> bool:
    try:
        clk = get_clock()
        return bool(clk.get("is_open"))
    except Exception as e:
        log(f"clock check failed: {e}", level="WARN")
        # If uncertain, assume open so we can still place market orders
        return True

# -------- Data helpers (with feed fallback) --------

def _get_with_feed(url: str, params: dict) -> requests.Response:
    # try configured feed, then iex if 403
    p = dict(params)
    p["feed"] = ALPACA_DATA_FEED
    r = http("GET", url, params=p)
    if r.status_code == 403 and ALPACA_DATA_FEED.lower() != "iex":
        p["feed"] = "iex"
        r = http("GET", url, params=p)
    return r

def latest_quote(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = _get_with_feed(url, {})
    if r.status_code >= 300:
        log(f"quote fetch failed {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("quote") or js
    except Exception as e:
        log(f"quote parse failed {symbol}: {e}", level="WARN")
        return None

def latest_trade(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    r = _get_with_feed(url, {})
    if r.status_code >= 300:
        log(f"trade fetch failed {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("trade") or js
    except Exception as e:
        log(f"trade parse failed {symbol}: {e}", level="WARN")
        return None

def latest_bar(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5)
    r = _get_with_feed(url, {
        "timeframe": ATR_TIMEFRAME,
        "start": start.isoformat(),
        "end": now.isoformat(),
        "limit": 1,
        "adjustment": "split",
    })
    if r.status_code >= 300:
        log(f"bar fetch failed {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        bars = js.get("bars") or []
        return bars[-1] if bars else None
    except Exception as e:
        log(f"bar parse failed {symbol}: {e}", level="WARN")
        return None

def bars_df(symbol: str, timeframe: str = ATR_TIMEFRAME, lookback: int = 400) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=10)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    feeds = [ALPACA_DATA_FEED] + ([] if ALPACA_DATA_FEED.lower() == "iex" else ["iex"])
    last_err = None
    for feed in feeds:
        try:
            r = http("GET", url, params={
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": now.isoformat(),
                "limit": lookback,
                "adjustment": "split",
                "feed": feed,
            })
            if r.status_code == 403:
                last_err = RuntimeError(f"403 on feed {feed}")
                continue
            r.raise_for_status()
            js = r.json()
            bars = js.get("bars", [])
            if not bars:
                raise RuntimeError(f"No bars for {symbol} (feed={feed})")
            df = pd.DataFrame(bars).rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df.set_index("ts", inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception as e:
            last_err = e
            log(f"bars fetch fail {symbol} feed={feed}: {e}", level="WARN")
    raise RuntimeError(f"bars failed for {symbol}: {last_err}")

def compute_atr(df: pd.DataFrame, length: int = ATR_LENGTH) -> float:
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return float(atr.dropna().iloc[-1])

# -------- Entry price inference --------

def infer_entry_reference(symbol: str, side: str, is_open: bool) -> float:
    """
    Robust reference price for entry:
      1) latest quote: lp (last) then ap/bp
      2) latest trade: p
      3) latest bar: close
    Must be > 0 or raises.
    """
    q = latest_quote(symbol) or {}
    last = float(q.get("lp") or q.get("last") or 0)
    ask  = float(q.get("ap") or q.get("ask_price") or 0)
    bid  = float(q.get("bp") or q.get("bid_price") or 0)

    # Prefer last if present
    ref = last

    # During ETH last may be missing; fall back to side-relevant quote
    if ref <= 0:
        if side == "buy":
            ref = ask if ask > 0 else bid
        else:
            ref = bid if bid > 0 else ask

    # Try latest trade
    if ref <= 0:
        t = latest_trade(symbol) or {}
        ref = float(t.get("p") or t.get("price") or 0)

    # Try latest bar close
    if ref <= 0:
        b = latest_bar(symbol) or {}
        ref = float(b.get("c") or b.get("close") or 0)

    if ref <= 0:
        raise RuntimeError("cannot infer entry reference price")

    # Optional tiny pad in ETH so a limit is more likely to rest logically
    if not is_open and LIMIT_PAD_CENTS != 0:
        pad = LIMIT_PAD_CENTS / 100.0
        ref = ref + (pad if side == "buy" else -pad)

    return q2(ref)

# -------- Bracket submit --------

def submit_bracket(symbol: str, side: str, qty: str, prefer_limit_when_closed: bool = True) -> dict:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    is_open = is_open_now()
    log(f"clock is_open={is_open}")

    # Compute ATR + reference
    df = bars_df(symbol)
    atr = compute_atr(df)

    entry_ref = infer_entry_reference(symbol, side, is_open)

    # Decide entry type
    if is_open:
        order_type = "market"
        limit_price = None
    else:
        order_type = "limit" if prefer_limit_when_closed else "market"
        limit_price = entry_ref

    # Build TP/SL from entry_ref
    if side == "buy":
        tp = q2(entry_ref + ATR_MULT_TP * atr)
        sl = q2(entry_ref - ATR_MULT_SL * atr)
    else:
        tp = q2(entry_ref - ATR_MULT_TP * atr)
        sl = q2(entry_ref + ATR_MULT_SL * atr)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": EXTENDED_HOURS,
        "client_order_id": f"BRK-{symbol}-{int(time.time())}",
    }

    if order_type == "market":
        payload["type"] = "market"
    else:
        payload["type"] = "limit"
        payload["limit_price"] = f"{q2(limit_price):.2f}"

    log(f"submit bracket: {payload}")
    r = http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"bracket submit failed: {r.status_code} {r.text}", level="ERROR")
        r.raise_for_status()
    js = r.json()
    print(json.dumps(js, indent=2))
    return js

# -------- CLI --------

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit ATR-aware bracket entry (market RTH, limit ETH).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy","sell"])
    ap.add_argument("--qty", default="1")
    ap.add_argument("--prefer_limit_when_closed", action="store_true",
                    help="Prefer limit order when market is closed (default True).")
    return ap.parse_args(argv)

if __name__ == "__main__":
    args = parse_cli(sys.argv[1:])
    submit_bracket(args.symbol, args.side, args.qty, prefer_limit_when_closed=True or args.prefer_limit_when_closed)
