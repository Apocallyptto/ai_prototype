# services/bracket_helper.py
from __future__ import annotations

import os
import time
import math
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional

import requests
import pandas as pd

LOG = logging.getLogger("bracket_helper")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s %(name)s | %(message)s"))
if not LOG.handlers:
    LOG.addHandler(ch)

# -----------------------
# Environment / constants
# -----------------------
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()  # paper | live

TIMEFRAME = os.getenv("TIMEFRAME", "5Min")

# ATR exits
USE_ATR = os.getenv("ATR_EXITS", "1") == "1"
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_LOOKBACK_DAYS = int(os.getenv("ATR_LOOKBACK_DAYS", "30"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.5"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))
MAX_TP_PCT = float(os.getenv("MAX_TP_PCT", "0.015"))  # 1.5%
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "0.015"))  # 1.5%

# Dynamic sizing (risk-based)
USE_DYNAMIC_SIZE = os.getenv("USE_DYNAMIC_SIZE", "0") == "1"
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "0.0025"))  # 0.25%
MIN_QTY = int(os.getenv("MIN_QTY", "1"))
MAX_QTY = int(os.getenv("MAX_QTY", "10"))

# Confidence-weighted sizing
USE_SIZE_BY_STRENGTH = os.getenv("USE_SIZE_BY_STRENGTH", "0") == "1"
SIZE_MIN_STRENGTH = float(os.getenv("SIZE_MIN_STRENGTH", "0.58"))
SIZE_MAX_STRENGTH = float(os.getenv("SIZE_MAX_STRENGTH", "0.75"))
SIZE_EXP = float(os.getenv("SIZE_EXP", "2.0"))
SIZE_MIN_MULT = float(os.getenv("SIZE_MIN_MULT", "0.50"))  # 50% of base qty at lower bound
SIZE_MAX_MULT = float(os.getenv("SIZE_MAX_MULT", "1.00"))  # 100% at/above upper bound

# Data source fallback for ATR (Alpaca first, yfinance fallback)
ATR_DATA_SOURCE = os.getenv("ATR_DATA_SOURCE", "auto").lower()  # auto | iex | yf

# -----------------------
# Helpers (HTTP)
# -----------------------
def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def _alpaca_get(path: str, params: Optional[dict] = None) -> requests.Response:
    url = f"{ALPACA_BASE_URL}{path}"
    return requests.get(url, headers=_alpaca_headers(), params=params or {}, timeout=15)

def _alpaca_post(path: str, payload: dict) -> requests.Response:
    url = f"{ALPACA_BASE_URL}{path}"
    return requests.post(url, headers=_alpaca_headers(), data=json.dumps(payload), timeout=15)

def _data_get(path: str, params: Optional[dict] = None) -> requests.Response:
    url = f"{ALPACA_DATA_URL}{path}"
    return requests.get(url, headers=_alpaca_headers(), params=params or {}, timeout=20)

# -----------------------
# Market utilities
# -----------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _is_market_open() -> bool:
    try:
        r = _alpaca_get("/v2/clock")
        r.raise_for_status()
        j = r.json()
        return bool(j.get("is_open"))
    except Exception:
        # If unsure, treat as closed to be conservative
        return False

def _get_last_price(symbol: str) -> float:
    """
    Fetch last trade price (prefer Alpaca /last/stocks, fallback to yfinance).
    """
    # Try Alpaca last trade (works on paper/live with proper data entitlement)
    try:
        r = _data_get(f"/v2/stocks/{symbol}/trades/latest")
        if r.status_code == 403:
            raise requests.HTTPError("Forbidden", response=r)
        r.raise_for_status()
        px = float(r.json()["trade"]["p"])
        return px
    except Exception:
        # Fallback to yfinance
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.fast_info
        last = info.get("last_price")
        if last is None:
            # final fallback: close
            hist = t.history(period="1d", interval="1m")
            if len(hist) == 0:
                raise RuntimeError(f"Cannot fetch price for {symbol}")
            last = float(hist["Close"].iloc[-1])
        return float(last)

# -----------------------
# ATR computation
# -----------------------
def _fetch_bars(symbol: str, lookback_days: int, timeframe: str) -> pd.DataFrame:
    start = (_now_utc() - timedelta(days=lookback_days)).isoformat()
    end = _now_utc().isoformat()
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "all",
    }
    # Alpaca data (IEX feed)
    if ATR_DATA_SOURCE in ("auto", "iex"):
        try:
            r = _data_get(f"/v2/stocks/{symbol}/bars", params)
            if r.status_code == 403:
                raise requests.HTTPError("Forbidden", response=r)
            r.raise_for_status()
            bars = r.json().get("bars", [])
            if bars:
                df = pd.DataFrame(bars)
                # normalize naming
                df.rename(
                    columns={"t": "time", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
                    inplace=True,
                )
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df.set_index("time", inplace=True)
                return df
        except Exception:
            if ATR_DATA_SOURCE == "iex":  # do not fallback if explicitly forced
                raise

    # yfinance fallback
    import yfinance as yf
    t = yf.Ticker(symbol)
    hist = t.history(period=f"{lookback_days}d", interval="5m", auto_adjust=False)
    if hist is None or len(hist) == 0:
        raise RuntimeError(f"No bars for {symbol} from yfinance")
    # Standardize columns
    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def _atr(symbol: str, period: int, lookback_days: int) -> float:
    df = _fetch_bars(symbol, lookback_days, TIMEFRAME)
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr)

def _atr_exits(symbol: str, side: str, base_price: float) -> Tuple[float, float]:
    if not USE_ATR:
        # fallback to small static brackets if ATR disabled
        tp = base_price * (1.0 + (0.004 if side == "buy" else -0.004))
        sl = base_price * (1.0 - (0.004 if side == "buy" else -0.004))
        return (tp, sl)

    atr = _atr(symbol, ATR_PERIOD, ATR_LOOKBACK_DAYS)
    # clamp ATR-based distances with safety % caps
    tp_off = min(ATR_MULT_TP * atr, MAX_TP_PCT * base_price)
    sl_off = min(ATR_MULT_SL * atr, MAX_SL_PCT * base_price)

    if side == "buy":
        tp = base_price + tp_off
        sl = base_price - sl_off
    else:
        tp = base_price - tp_off
        sl = base_price + sl_off

    # Alpaca requires stop_price <= base - 0.01 for buys, >= base + 0.01 for sells
    penny = 0.01
    if side == "buy":
        sl = min(sl, base_price - penny)
    else:
        sl = max(sl, base_price + penny)

    return (float(tp), float(sl))

# -----------------------
# Sizing
# -----------------------
def _account_equity() -> float:
    try:
        r = _alpaca_get("/v2/account")
        r.raise_for_status()
        eq = float(r.json().get("equity", 0.0))
        return eq
    except Exception:
        # last resort: use 100k default
        return 100000.0

def _risk_per_share(symbol: str, side: str, base_price: float) -> float:
    """
    Use ATR stop distance as risk-per-share.
    """
    tp, sl = _atr_exits(symbol, side, base_price)
    if side == "buy":
        rps = base_price - sl
    else:
        rps = sl - base_price
    return max(float(rps), 0.01)  # avoid division by zero

def _strength_scale(strength: Optional[float]) -> float:
    """
    Convert signal strength -> [SIZE_MIN_MULT, SIZE_MAX_MULT] using
    a curved mapping with exponent SIZE_EXP.
    """
    if not USE_SIZE_BY_STRENGTH or strength is None:
        return 1.0

    s = max(min(float(strength), 1.0), 0.0)
    lo, hi = SIZE_MIN_STRENGTH, SIZE_MAX_STRENGTH
    if hi <= lo:
        return 1.0
    # normalize and curve
    t = (s - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    curved = t ** SIZE_EXP
    mult = SIZE_MIN_MULT + (SIZE_MAX_MULT - SIZE_MIN_MULT) * curved
    return float(max(0.0, mult))

def _compute_dynamic_qty(symbol: str, side: str, base_price: float, strength: Optional[float] = None) -> int:
    if not USE_DYNAMIC_SIZE:
        return max(MIN_QTY, 1)

    equity = _account_equity()
    per_trade_risk = max(equity * RISK_PCT_PER_TRADE, 1.0)
    rps = _risk_per_share(symbol, side, base_price)
    if rps <= 0.0:
        return MIN_QTY

    raw_qty = per_trade_risk / rps
    # strength weighting
    mult = _strength_scale(strength)
    raw_qty *= mult

    # clamp + round
    qty = int(max(MIN_QTY, min(MAX_QTY, math.floor(raw_qty))))
    return max(qty, 1)

# -----------------------
# Public: submit_bracket
# -----------------------
def submit_bracket(
    symbol: str,
    side: str,
    qty: Optional[int] = None,
    time_in_force: str = "day",
    order_type: str = "market",  # "market" | "limit"
    client_id: Optional[str] = None,
    strength: Optional[float] = None,
) -> Dict:
    """
    Centralized bracket submitter. Applies:
      - ATR exits (if enabled)
      - Dynamic sizing (if enabled)
      - Confidence-weighted sizing (if enabled and strength provided)
      - Auto RTH/ETH rules for Alpaca brackets (brackets are RTH-only -> use 'limit' if market closed)
    """
    symbol = symbol.upper()
    side = side.lower()
    assert side in ("buy", "sell")

    is_open = _is_market_open()
    last_price = _get_last_price(symbol)

    # Choose entry type:
    # - If market is open and user asked "market", use market
    # - Else force a tight limit around last_price to emulate "marketable" after-hours
    use_limit = (order_type == "limit") or (not is_open)
    entry_type = "limit" if use_limit else "market"

    base_price = float(last_price)
    take_profit, stop_loss = _atr_exits(symbol, side, base_price)

    # quantity
    if qty is None:
        qty = _compute_dynamic_qty(symbol, side, base_price, strength=strength)

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": str(int(qty)),
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{take_profit:.2f}"},
        "stop_loss": {"stop_price": f"{stop_loss:.2f}"},
        "extended_hours": False,  # Alpaca restriction for brackets
        "client_order_id": client_id or f"BRK-{symbol}-{int(time.time())}",
        "type": entry_type,
    }

    if entry_type == "limit":
        payload["limit_price"] = f"{base_price:.2f}"

    LOG.info(
        "submit bracket (atr(period=%s,tp×%s,sl×%s), dyn=%s) -> %s",
        ATR_PERIOD, ATR_MULT_TP, ATR_MULT_SL, str(USE_DYNAMIC_SIZE),
        payload,
    )

    r = _alpaca_post("/v2/orders", payload)
    if r.status_code >= 400:
        try:
            LOG.info("submit failed: %s", r.text)
        finally:
            r.raise_for_status()
    return r.json()
