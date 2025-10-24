# services/bracket_helper.py
from __future__ import annotations

import os
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# Optional yfinance fallback for bars/last price
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

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
        "Content-Type": "application/json",
    })

# ---- ATR + caps ----
USE_ATR_EXITS   = os.getenv("ATR_EXITS", "1") == "1"
ATR_PERIOD      = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK    = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP     = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL     = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT      = float(os.getenv("MAX_TP_PCT", "0.015"))  # 1.5%
MAX_SL_PCT      = float(os.getenv("MAX_SL_PCT", "0.015"))  # 1.5%

# ---- Dynamic sizing ----
USE_DYNAMIC_SIZE    = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE  = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25% of equity
MIN_QTY             = int(os.getenv("MIN_QTY", "1"))
MAX_QTY             = int(os.getenv("MAX_QTY", "5"))

# ---- Misc ----
TIMEFRAME   = os.getenv("NN_TIMEFRAME", "5Min")
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))  # passed through to signal table elsewhere

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# --------- Market data helpers ---------

def _yf_bars(symbol: str, days: int, interval: str = "5m") -> pd.DataFrame:
    if not HAVE_YF:
        raise RuntimeError("yfinance is not installed; cannot fetch bars fallback.")
    df = yf.download(symbol, period=f"{days}d", interval=interval, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"yfinance returned empty df for {symbol}")
    # Normalize columns to title case (Open/High/Low/Close/Volume)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    return df

def _last_price_yf(symbol: str) -> float:
    if not HAVE_YF:
        raise RuntimeError("yfinance is not installed; cannot fetch last price fallback.")
    tkr = yf.Ticker(symbol)
    p = tkr.fast_info.get("last_price") or tkr.fast_info.get("lastPrice")
    if p:
        return float(p)
    # fallback to last close from 1d 1m
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return float(df["Close"].iloc[-1])
    raise RuntimeError(f"unable to get last price for {symbol} via yfinance")

def _alpaca_last_price(symbol: str) -> Optional[float]:
    try:
        r = S.get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest", timeout=10)
        r.raise_for_status()
        q = r.json().get("quote") or {}
        # use ask if present else bid/last-like
        for k in ("ap","bp","lp"):
            if k in q and q[k] is not None:
                return float(q[k])
    except Exception:
        return None
    return None

def _get_last_price(symbol: str) -> float:
    p = _alpaca_last_price(symbol)
    if p is not None:
        return p
    return _last_price_yf(symbol)

def _fetch_bars(symbol: str, lookback_days: int, timeframe: str) -> pd.DataFrame:
    # Prefer Alpaca data; if blocked/403, fall back to yfinance
    start = (_utcnow() - timedelta(days=lookback_days)).isoformat()
    end   = _utcnow().isoformat()
    try:
        r = S.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
            params={"timeframe": timeframe, "start": start, "end": end, "limit": 10000},
            timeout=15,
        )
        r.raise_for_status()
        js = r.json()
        bars = js.get("bars") or []
        if bars:
            # Build a DataFrame in the same shape we use for ATR
            df = pd.DataFrame([{
                "Open":  float(b["o"]),
                "High":  float(b["h"]),
                "Low":   float(b["l"]),
                "Close": float(b["c"]),
                "Volume": float(b.get("v", 0.0)),
                "ts": b.get("t"),
            } for b in bars])
            df.index = pd.to_datetime(df["ts"])
            df = df[["Open","High","Low","Close","Volume"]]
            return df
    except Exception:
        pass
    # Fallback
    return _yf_bars(symbol, lookback_days, interval="5m")

# --------- ATR / exits ---------

def _atr(symbol: str, period: int, lookback_days: int) -> float:
    df = _fetch_bars(symbol, lookback_days, TIMEFRAME)
    if df.shape[0] < period + 1:
        raise RuntimeError(f"not enough bars for ATR({period}) on {symbol}")

    high = df["High"].astype("float64")
    low  = df["Low"].astype("float64")
    close_prev = df["Close"].shift(1).astype("float64")

    tr1 = (high - low).abs()
    tr2 = (high - close_prev).abs()
    tr3 = (low  - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    return float(atr)

def _apply_caps(entry: float, proposed_tp: float, proposed_sl: float) -> Tuple[float, float]:
    max_tp = entry * (1 + MAX_TP_PCT)
    min_tp = entry * (1 - MAX_TP_PCT)
    max_sl = entry * (1 + MAX_SL_PCT)
    min_sl = entry * (1 - MAX_SL_PCT)
    tp = proposed_tp
    sl = proposed_sl
    # clamp
    tp = max(min(tp, max_tp), min_tp)
    sl = max(min(sl, max_sl), min_sl)
    return tp, sl

def _atr_exits(symbol: str, side: str, last_price: float) -> Tuple[float, float]:
    """Return (tp, sl) using ATR. If ATR disabled, make small symmetric exits."""
    if not USE_ATR_EXITS:
        # tiny static rails (still clamped)
        raw = last_price * 0.002  # 0.2%
        if side == "buy":
            tp, sl = last_price + raw, last_price - raw
        else:
            tp, sl = last_price - raw, last_price + raw
        return _apply_caps(last_price, tp, sl)

    atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK)
    if side == "buy":
        tp = last_price + ATR_MULT_TP * atr
        sl = last_price - ATR_MULT_SL * atr
    else:
        tp = last_price - ATR_MULT_TP * atr
        sl = last_price + ATR_MULT_SL * atr
    return _apply_caps(last_price, tp, sl)

# --------- Equity / sizing ---------

def _account_equity() -> float:
    r = S.get(f"{ALPACA_BASE_URL}/v2/account", timeout=10)
    r.raise_for_status()
    js = r.json()
    # use cash + market_value if available; fallback to 'equity'
    eq = js.get("equity") or js.get("portfolio_value") or js.get("cash")
    return float(eq)

def _risk_per_share(symbol: str, side: str, last_price: float) -> float:
    tp, sl = _atr_exits(symbol, side, last_price)
    return abs(last_price - sl)

def _compute_dynamic_qty(symbol: str, side: str, last_price: float) -> int:
    if not USE_DYNAMIC_SIZE:
        return max(MIN_QTY, 1)
    equity = _account_equity()
    risk_budget = equity * RISK_PCT_PER_TRADE
    rps = max(_risk_per_share(symbol, side, last_price), 1e-6)
    qty = int(math.floor(risk_budget / rps))
    qty = max(MIN_QTY, min(qty, MAX_QTY))
    return qty

# --------- Submit ---------

def submit_bracket(
    symbol: str,
    side: str,
    strength: float,
    qty: Optional[int] = None,
    client_id: Optional[str] = None,
    time_in_force: str = "day",
    order_type: str = "market",
    extended_hours: bool = False,
) -> Dict:
    """
    Build and send a bracket order to Alpaca with ATR exits and optional dynamic sizing.
    - If qty is None and USE_DYNAMIC_SIZE=1, size = Kelly-like risk budget / ATR distance.
    - order_type: "market" (recommended for reliability) or "limit" (then add limit_price).
    """
    side = side.lower()
    assert side in ("buy", "sell"), f"invalid side {side}"

    # robust last price source
    last_price = _get_last_price(symbol)

    # dynamic size if needed
    if qty is None:
        qty = _compute_dynamic_qty(symbol, side, last_price)
    qty = int(max(qty or 0, 1))

    # ATR exits
    tp_px, sl_px = _atr_exits(symbol, side, last_price)

    payload = {
        "symbol": symbol.upper(),
        "side": side,
        "qty": str(qty),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp_px:.2f}"},
        "stop_loss":   {"stop_price":  f"{sl_px:.2f}"},
        "extended_hours": bool(extended_hours),
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
        "type": order_type,
    }

    # Only attach limit_price for limit orders
    if order_type == "limit":
        payload["limit_price"] = f"{last_price:.2f}"

    print(f"INFO bracket_helper | submit bracket (atr(period={ATR_PERIOD},tp×{ATR_MULT_TP},sl×{ATR_MULT_SL}), dyn={USE_DYNAMIC_SIZE}) -> {payload}")

    r = S.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=15)
    try:
        r.raise_for_status()
    except requests.HTTPError as he:
        print(f"INFO bracket_helper | submit failed: {r.text}")
        raise
    return r.json()
