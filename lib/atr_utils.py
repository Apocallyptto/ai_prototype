# lib/atr_utils.py
from __future__ import annotations
import os, time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, Optional, Dict

import requests
import pandas as pd

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' for paper/free

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
})

_cache: Dict[str, Tuple[float, float]] = {}  # key -> (atr_value, expires_epoch)


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


def _quantize(price: float, tick: Decimal = Decimal("0.01")) -> float:
    return float(Decimal(str(price)).quantize(tick, rounding=ROUND_HALF_UP))


def get_last_price(symbol: str) -> float:
    """Return latest trade price; fallback to last 1-min bar close if feed is restricted."""
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    r = _http("GET", url, params={"feed": ALPACA_DATA_FEED})
    if r.status_code == 403:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=2)
        bars_url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
        rb = _http("GET", bars_url, params={
            "timeframe": "1Min",
            "start": start.isoformat(),
            "end": now.isoformat(),
            "limit": 500,
            "adjustment": "split",
            "feed": "iex",
        })
        rb.raise_for_status()
        bars = rb.json().get("bars", [])
        if not bars:
            raise RuntimeError(f"No bars for {symbol}")
        return float(bars[-1]["c"])
    r.raise_for_status()
    return float(r.json()["trade"]["p"])


def _recent_bars(symbol: str, timeframe: str, lookback_days: int = 10, limit: int = 400) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

    def _get(feed: str):
        return _http("GET", url, params={
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": now.isoformat(),
            "limit": limit,
            "adjustment": "split",
            "feed": feed,
        })

    last_err = None
    feeds = [ALPACA_DATA_FEED] + ([] if ALPACA_DATA_FEED == "iex" else ["iex"])
    for feed in feeds:
        try:
            r = _get(feed)
            if r.status_code == 403:
                last_err = RuntimeError(f"403 on feed '{feed}'")
                continue
            r.raise_for_status()
            bars = r.json().get("bars", [])
            if not bars:
                last_err = RuntimeError("no bars")
                continue
            df = pd.DataFrame(bars)
            df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df.set_index("ts", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"bar fetch failed for {symbol}: {last_err}")


def _atr_from_df(df: pd.DataFrame, length: int) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean().dropna()
    return float(atr.iloc[-1])


def get_atr(symbol: str,
            timeframe: str = os.getenv("ATR_TIMEFRAME", "5Min"),
            length: int = int(os.getenv("ATR_LENGTH", "14")),
            cache_sec: int = 60) -> float:
    """Fetch ATR with a tiny TTL cache to avoid API spam."""
    key = f"{symbol}|{timeframe}|{length}"
    now = time.time()
    hit = _cache.get(key)
    if hit and hit[1] > now:
        return hit[0]
    df = _recent_bars(symbol, timeframe=timeframe)
    val = _atr_from_df(df, length)
    _cache[key] = (val, now + cache_sec)
    return val


def atr_targets(symbol: str,
                side: str,
                ref_price: Optional[float] = None,
                mult_tp: float = float(os.getenv("ATR_MULT_TP_ENTRY", os.getenv("ATR_MULT_TP", "1.2"))),
                mult_sl: float = float(os.getenv("ATR_MULT_SL_ENTRY", os.getenv("ATR_MULT_SL", "1.0"))),
                timeframe: str = os.getenv("ATR_TIMEFRAME", "5Min"),
                length: int = int(os.getenv("ATR_LENGTH", "14"))) -> Tuple[float, float]:
    """
    Compute (tp, sl) off a reference price using ATR multiples.
    Returns tick-quantized prices (0.01).
    """
    if ref_price is None:
        ref_price = get_last_price(symbol)
    atr = get_atr(symbol, timeframe=timeframe, length=length)
    if side == "buy":
        tp = ref_price + mult_tp * atr
        sl = ref_price - mult_sl * atr
    else:  # 'sell' short
        tp = ref_price - mult_tp * atr
        sl = ref_price + mult_sl * atr
    return _quantize(tp), _quantize(sl)
