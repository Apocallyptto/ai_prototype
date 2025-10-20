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

# ATR knobs
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_TIMEFRAME = os.getenv("ATR_TIMEFRAME", "5Min")
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.2"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))

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

def get_clock() -> dict:
    r = http("GET", f"{ALPACA_BASE_URL}/v2/clock")
    r.raise_for_status()
    return r.json()

def latest_quote(symbol: str) -> Optional[dict]:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = http("GET", url, params={"feed": ALPACA_DATA_FEED})
    if r.status_code == 403 and ALPACA_DATA_FEED.lower() != "iex":
        r = http("GET", url, params={"feed": "iex"})
    if r.status_code >= 300:
        log(f"quote fetch failed {symbol}: {r.status_code} {r.text}", level="WARN")
        return None
    try:
        js = r.json()
        return js.get("quote") or js
    except Exception as e:
        log(f"quote parse failed {symbol}: {e}", level="WARN")
        return None

def bars_df(symbol: str, timeframe: str = ATR_TIMEFRAME, lookback: int = 400) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=10)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

    feeds_to_try = [ALPACA_DATA_FEED] + ([] if ALPACA_DATA_FEED.lower() == "iex" else ["iex"])
    last_err = None
    for feed in feeds_to_try:
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

def q2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def submit_bracket(symbol: str, side: str, qty: str, prefer_limit_when_closed: bool = True) -> dict:
    # Determine session status
    is_open = True
    try:
        clk = get_clock()
        is_open = bool(clk.get("is_open"))
        log(f"clock is_open={is_open} next_open={clk.get('next_open')}")
    except Exception as e:
        log(f"clock check failed: {e}", level="WARN")

    # Get ATR and latest ref price
    df = bars_df(symbol)
    atr = compute_atr(df)
    q = latest_quote(symbol) or {}
    bid = float(q.get("bp") or q.get("bid_price") or 0)
    ask = float(q.get("ap") or q.get("ask_price") or 0)
    last = float(q.get("lp") or q.get("last") or 0)

    # Entry type
    order_type = "market"
    limit_price = None
    if not is_open and prefer_limit_when_closed:
        order_type = "limit"
        limit_price = q2(ask if side == "buy" else bid) if (ask if side=="buy" else bid) > 0 else q2(last)

    # Use entry_ref for TP/SL deltas (fall back to limit/last)
    entry_ref = float(limit_price or last or (ask if side=="buy" else bid))
    if entry_ref <= 0:
        raise RuntimeError("cannot infer entry reference price")

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
    }

    if order_type == "market":
        payload["type"] = "market"
    else:
        payload["type"] = "limit"
        payload["limit_price"] = f"{q2(limit_price):.2f}"

    # Optional helpful tag
    payload["client_order_id"] = f"BRK-{symbol}-{int(time.time())}"

    log(f"submit bracket: {payload}")
    r = http("POST", f"{ALPACA_BASE_URL}/v2/orders", data=json.dumps(payload))
    if r.status_code >= 300:
        log(f"bracket submit failed: {r.status_code} {r.text}", level="ERROR")
        r.raise_for_status()
    js = r.json()
    print(json.dumps(js, indent=2))
    return js

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit ATR-aware bracket entry.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["buy","sell"])
    ap.add_argument("--qty", default="1")
    ap.add_argument("--prefer_limit_when_closed", action="store_true", help="Use limit when market is closed (default True).")
    args = ap.parse_args(argv)
    return args

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")
    args = parse_cli(sys.argv[1:])
    submit_bracket(args.symbol, args.side, args.qty, prefer_limit_when_closed=True or args.prefer_limit_when_closed)
