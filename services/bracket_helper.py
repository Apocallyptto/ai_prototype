# services/bracket_helper.py
from __future__ import annotations

import os
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

import requests

# Optional yfinance fallback for ATR/last price
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# ---- ATR / exits configuration (env) ----
ATR_EXITS = os.getenv("ATR_EXITS", "1") == "1"
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT = float(os.getenv("MAX_TP_PCT", "0.015"))  # cap 1.5%
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "0.015"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()  # 'iex' fixes SIP 403 on paper keys
ATR_DATA_SOURCE = os.getenv("ATR_DATA_SOURCE", "alpaca,yf").lower().split(",")

# ---- Dynamic sizing ----
USE_DYNAMIC_SIZE = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25% default
MIN_QTY = int(os.getenv("MIN_QTY", "1"))
MAX_QTY = int(os.getenv("MAX_QTY", "10"))

S = requests.Session()
if API_KEY and API_SECRET:
    S.headers.update({
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
        "Content-Type": "application/json",
    })

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# ------------------------ Data helpers ------------------------

def get_last_price(symbol: str) -> float:
    """
    Public helper: last trade price. Tries Alpaca (feed=iex), falls back to yfinance.
    """
    # 1) Alpaca latest trade (IEX)
    try:
        r = S.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest",
            params={"feed": ALPACA_DATA_FEED},
            timeout=7,
        )
        r.raise_for_status()
        j = r.json()
        px = float(j["trade"]["p"])
        return px
    except Exception:
        pass

    # 2) yfinance fallback
    if yf is not None:
        try:
            hist = yf.Ticker(symbol).history(period="1d", interval="1m")
            if len(hist) > 0:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass

    raise RuntimeError(f"get_last_price: unable to fetch price for {symbol}")

def _fetch_bars_alpaca(symbol: str, lookback_days: int, timeframe: str = "5Min"):
    start = (_utcnow() - timedelta(days=lookback_days)).isoformat()
    end = _utcnow().isoformat()
    r = S.get(
        f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
        params={
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": 10000,
            "adjustment": "all",
            "feed": ALPACA_DATA_FEED,  # <- critical to avoid SIP 403 on paper
        },
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    bars = j.get("bars", [])
    # Normalize to list of dicts with o,h,l,c
    out = []
    for b in bars:
        out.append(
            {"o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"])}
        )
    return out

def _fetch_bars_yf(symbol: str, lookback_days: int, interval: str = "5m"):
    if yf is None:
        raise RuntimeError("yfinance not installed")
    import pandas as pd
    df = yf.download(symbol, period=f"{lookback_days}d", interval=interval, progress=False)
    if df is None or len(df) == 0:
        return []
    # Ensure single row access uses .iloc[...]
    out = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        out.append(
            {
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
            }
        )
    return out

def _fetch_bars(symbol: str, lookback_days: int, timeframe: str = "5Min"):
    """
    Tries sources in ATR_DATA_SOURCE (e.g., 'alpaca,yf').
    """
    first_err = None
    for src in ATR_DATA_SOURCE:
        src = src.strip()
        try:
            if src == "alpaca":
                return _fetch_bars_alpaca(symbol, lookback_days, timeframe)
            elif src in ("yf", "yfinance"):
                interval = "5m" if timeframe.lower().startswith("5") else "1h"
                return _fetch_bars_yf(symbol, lookback_days, interval)
        except Exception as e:
            first_err = first_err or e
            continue
    if first_err:
        raise first_err
    return []

# ------------------------ ATR & exits ------------------------

def _atr_from_bars(bars: List[Dict], period: int) -> float:
    """
    Wilder ATR over provided bar list (expects o,h,l,c). Returns last ATR value.
    """
    if len(bars) < period + 1:
        raise RuntimeError(f"Not enough bars for ATR: have={len(bars)} need>={period+1}")

    trs: List[float] = []
    prev_close = bars[0]["c"]
    for b in bars[1:]:
        h, l, c = b["h"], b["l"], b["c"]
        tr = max(
            h - l,
            abs(h - prev_close),
            abs(l - prev_close),
        )
        trs.append(tr)
        prev_close = c

    # Wilder smoothing
    atr = sum(trs[:period]) / period
    for t in trs[period:]:
        atr = (atr * (period - 1) + t) / period
    return float(atr)

def _atr(symbol: str, period: int, lookback_days: int, timeframe: str = "5Min") -> float:
    bars = _fetch_bars(symbol, lookback_days, timeframe)
    return _atr_from_bars(bars, period)

def _cap_by_pct(base: float, target: float, cap_pct: float, up: bool) -> float:
    """
    Cap distance from base by +/- cap_pct.
    up=True caps to base * (1+cap_pct), else base * (1-cap_pct).
    """
    if cap_pct <= 0:
        return target
    if up:
        return min(target, base * (1.0 + cap_pct))
    return max(target, base * (1.0 - cap_pct))

def _atr_exits(symbol: str, side: str, base_price: float) -> Tuple[float, float]:
    """
    Return (tp_price, sl_price) using ATR. Ensures stop is at least $0.01 away
    to satisfy Alpaca bracket validation.
    """
    atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK_DAYS)
    if side == "buy":
        tp = base_price + ATR_MULT_TP * atr
        sl = base_price - ATR_MULT_SL * atr
        # safety caps
        tp = _cap_by_pct(base_price, tp, MAX_TP_PCT, up=True)
        sl = _cap_by_pct(base_price, sl, MAX_SL_PCT, up=False)
        # bracket rule: stop must be <= base - 0.01
        sl = min(sl, base_price - 0.01)
    else:
        tp = base_price - ATR_MULT_TP * atr
        sl = base_price + ATR_MULT_SL * atr
        tp = _cap_by_pct(base_price, tp, MAX_TP_PCT, up=False)
        sl = _cap_by_pct(base_price, sl, MAX_SL_PCT, up=True)
        # bracket rule: stop must be >= base + 0.01
        sl = max(sl, base_price + 0.01)
    return (float(round(tp, 2)), float(round(sl, 2)))

# ------------------------ Dynamic sizing ------------------------

def _account_equity_fallback() -> float:
    """
    Try Alpaca account equity; fallback to env or 100k.
    """
    try:
        r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=7)
        r.raise_for_status()
        j = r.json()
        return float(j.get("equity") or j.get("last_equity") or j.get("cash") or 100000.0)
    except Exception:
        return float(os.getenv("EQUITY_FALLBACK", "100000"))

def _risk_per_share(symbol: str, side: str, base_price: float) -> float:
    """
    Estimated $ risk per share = distance between base and stop.
    """
    if ATR_EXITS:
        tp, sl = _atr_exits(symbol, side, base_price)
        return abs(base_price - sl)
    # fallback fixed 1% stop
    return base_price * 0.01

def _compute_dynamic_qty(symbol: str, side: str, base_price: float) -> int:
    """
    Position size by fixed-fraction risk budgeting.
    """
    equity = _account_equity_fallback()
    risk_budget = equity * RISK_PCT_PER_TRADE
    rps = _risk_per_share(symbol, side, base_price)
    if rps <= 0:
        return max(MIN_QTY, 1)
    qty = int(max(math.floor(risk_budget / rps), 0))
    qty = max(MIN_QTY, min(qty, MAX_QTY))
    return qty

# ------------------------ Submit ------------------------

def _market_is_open() -> bool:
    """
    Quick read of clock. If clock says trading_on=True we consider open.
    """
    try:
        r = S.get(f"{ALPACA_BASE_URL}/v2/clock", timeout=5)
        r.raise_for_status()
        j = r.json()
        return bool(j.get("is_open") or j.get("trading_on"))
    except Exception:
        # assume open; the worst case we submit DAY orders (brackets only support RTH anyway)
        return True

def submit_bracket(
    symbol: str,
    side: str,
    qty: Optional[int] = None,
    strength: Optional[float] = None,
    client_id: Optional[str] = None,
    time_in_force: str = "day",
    order_type: str = "market",
    extended_hours: bool = False,  # brackets do not support ETH on Alpaca
) -> Dict:
    """
    Place an ATR-aware bracket. If qty is None and USE_DYNAMIC_SIZE=1, compute qty dynamically.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    base_price = get_last_price(symbol)

    # ATR exits if enabled
    if ATR_EXITS:
        tp, sl = _atr_exits(symbol, side, base_price)
        atr_tag = f"atr(period={ATR_PERIOD},tp×{ATR_MULT_TP},sl×{ATR_MULT_SL}), dyn={USE_DYNAMIC_SIZE}"
    else:
        # fixed 1% exits fallback
        if side == "buy":
            tp = round(base_price * (1 + 0.01), 2)
            sl = round(base_price * (1 - 0.01), 2)
        else:
            tp = round(base_price * (1 - 0.01), 2)
            sl = round(base_price * (1 + 0.01), 2)
        atr_tag = "fixed1%, dyn={USE_DYNAMIC_SIZE}"

    # qty
    if qty is None and USE_DYNAMIC_SIZE:
        qty = _compute_dynamic_qty(symbol, side, base_price)
    qty = int(qty or 1)

    # Entry order type — DAY only (brackets require RTH)
    is_open = _market_is_open()
    entry_type = order_type
    entry_limit = None
    if not is_open and entry_type == "market":
        # place a tight limit near last price (±$0.02) to avoid post-close market entry attempt
        entry_type = "limit"
        entry_limit = round(base_price + (0.02 if side == "buy" else -0.02), 2)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp:.2f}"},
        "stop_loss": {"stop_price": f"{sl:.2f}"},
        "extended_hours": False,  # brackets: must be RTH
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
    }

    if entry_type == "market":
        payload["type"] = "market"
    else:
        payload["type"] = "limit"
        payload["limit_price"] = f"{entry_limit:.2f}"

    print(f"INFO bracket_helper | submit bracket ({atr_tag}) -> {payload}")

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=10)
    try:
        r.raise_for_status()
    except requests.HTTPError as he:
        print(f"INFO bracket_helper | submit failed: {r.text}")
        raise
    return r.json()
