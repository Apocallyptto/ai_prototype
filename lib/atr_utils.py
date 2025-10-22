# lib/atr_utils.py
from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional, Literal
from datetime import datetime, timezone
import requests

ALPACA_MARKET_BASE = "https://data.alpaca.markets"
FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # 'iex' (default) or 'sip'

# Risk knobs (can be overridden by env)
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "1.2"))  # take-profit = entry +- ATR*mult
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.0"))  # stop-loss  = entry +- ATR*mult

SESSION = requests.Session()
SESSION.headers.update({
    "Content-Type": "application/json",
    # market data is public for free feed, but set keys if you have them to avoid rate limits
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET", "")
})

def _bars(symbol: str, timeframe: str = "5Min", limit: int = 200) -> List[dict]:
    """
    Fetch recent bars. timeframe one of: 1Min, 5Min, 15Min, 1Hour, 1Day.
    Returns list of bars (each has t, o, h, l, c, v).
    """
    url = f"{ALPACA_MARKET_BASE}/v2/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": str(limit), "feed": FEED, "adjustment": "split"}
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    return js.get("bars", []) or []

def _last_trade_price(symbol: str) -> Optional[float]:
    try:
        url = f"{ALPACA_MARKET_BASE}/v2/stocks/{symbol}/trades/latest"
        r = SESSION.get(url, params={"feed": FEED}, timeout=10)
        r.raise_for_status()
        t = r.json().get("trade")
        if not t:
            return None
        return float(t.get("p", 0) or 0)
    except Exception:
        return None

def _true_ranges(bars: List[dict]) -> List[float]:
    tr: List[float] = []
    prev_close: Optional[float] = None
    for b in bars:
        h = float(b["h"]); l = float(b["l"]); c = float(b["c"])
        if prev_close is None:
            tr.append(h - l)
        else:
            tr.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
        prev_close = c
    return tr

def compute_atr(symbol: str, timeframe: str = "5Min", period: int = ATR_PERIOD) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (atr, last_price). If not enough bars, atr is None.
    """
    bars = _bars(symbol, timeframe=timeframe, limit=max(2*period+2, 60))
    if len(bars) < period + 1:
        return (None, _last_trade_price(symbol))
    tr = _true_ranges(bars)
    if len(tr) < period:
        return (None, _last_trade_price(symbol))
    # Wilder's ATR (simple moving average over TR works fine for live usage)
    atr = sum(tr[-period:]) / float(period)
    last = float(bars[-1]["c"])
    return (atr, last)

def atr_tp_sl(entry: float, side: Literal["buy","sell"], atr: float,
              tp_mult: float = ATR_TP_MULT, sl_mult: float = ATR_SL_MULT) -> Tuple[float, float]:
    """
    Compute (take_profit, stop_loss) from entry and ATR.
    For BUY: TP=entry+atr*tp_mult, SL=entry-atr*sl_mult
    For SELL: TP=entry-atr*tp_mult, SL=entry+atr*sl_mult
    """
    if side == "buy":
        return (entry + atr*tp_mult, entry - atr*sl_mult)
    else:
        return (entry - atr*tp_mult, entry + atr*sl_mult)

def round_to_tick(px: float, tick: float = 0.01) -> float:
    if tick <= 0:
        return px
    return round(px / tick) * tick
