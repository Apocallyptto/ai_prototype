# services/bracket_public.py
"""
Public, stable wrapper over services.bracket_helper with robust fallbacks
and an adaptive submit_bracket that maps price kwargs to whatever your
real function expects.

Public API:
- get_last_price(symbol) -> float
- atr_value(symbol, period:int, lookback_days:int) -> float
- risk_per_share(symbol, side:str, last_price:float) -> float
- compute_dynamic_qty(symbol, side:str, last_price:float) -> int
- submit_bracket(symbol, side, qty, **kwargs) -> broker response
"""

from __future__ import annotations
from typing import Callable, Optional
import os
import math
import inspect

# Your existing implementation (whatever names it uses)
from services import bracket_helper as _bh  # type: ignore


# ---------- small utils ----------
def _first_callable(names: list[str]) -> Optional[Callable]:
    for n in names:
        fn = getattr(_bh, n, None)
        if callable(fn):
            return fn
    return None

def _envfloat(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _envint(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


# ---------- Try to locate existing helpers ----------
_get_last_price_fn = _first_callable([
    "get_last_price", "_get_last_price",
    "last_price", "fetch_last_price", "get_price", "last",
])

_atr_fn = _first_callable([
    "atr_value", "_atr", "atr", "compute_atr", "average_true_range",
])

_risk_per_share_fn = _first_callable([
    "risk_per_share", "_risk_per_share", "compute_risk_per_share",
    "calc_risk_per_share", "risk_share",
])

_compute_dynamic_qty_fn = _first_callable([
    "compute_dynamic_qty", "_compute_dynamic_qty", "position_size",
    "size_for_order", "compute_qty", "dynamic_qty",
])

_submit_bracket_fn = _first_callable([
    "submit_bracket", "place_bracket", "submit_bracket_order",
])


# ---------- Fallbacks (yfinance-based) ----------
def _yf_last_price(symbol: str) -> float:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(
            "get_last_price fallback needs yfinance. Install `yfinance` "
            "or expose get_last_price in services.bracket_helper."
        ) from e
    t = yf.Ticker(symbol)
    info = getattr(t, "fast_info", None)
    if info:
        v = getattr(info, "last_price", None) or getattr(info, "last", None)
        if v is not None:
            return float(v)
        if isinstance(info, dict):
            v = info.get("last_price")
            if v is not None:
                return float(v)
    hist = t.history(period="5d", interval="1m")
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance could not fetch prices for {symbol}")
    return float(hist["Close"].dropna().iloc[-1])

def _yf_atr(symbol: str, period: int, lookback_days: int) -> float:
    try:
        import yfinance as yf
        import pandas as pd  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "atr_value fallback needs yfinance (and pandas)."
        ) from e

    t = yf.Ticker(symbol)
    bars = t.history(period=f"{max(lookback_days, period*3)}d", interval="1d")
    if bars is None or bars.empty:
        raise RuntimeError(f"yfinance could not fetch daily bars for {symbol}")

    high = bars["High"]
    low = bars["Low"]
    close = bars["Close"].shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = (tr1.combine(tr2, max)).combine(tr3, max)

    atr = tr.rolling(window=period, min_periods=period).mean().dropna()
    if atr.empty:
        raise RuntimeError(f"ATR could not be computed for {symbol}")
    return float(atr.iloc[-1])

def _fallback_risk_per_share(symbol: str, side: str, last_price: float) -> float:
    period = _envint("ATR_PERIOD", 14)
    lookback = _envint("ATR_LOOKBACK_DAYS", 30)
    atr_mult_sl = _envfloat("ATR_MULT_SL", 1.0)
    atr = atr_value(symbol, period, lookback)
    return max(1e-6, atr * atr_mult_sl)

def _fallback_compute_dynamic_qty(symbol: str, side: str, last_price: float) -> int:
    equity = _envfloat("EQUITY", 100000.0)
    risk_pct = _envfloat("RISK_PCT_PER_TRADE", 0.0025)  # 0.25%
    min_qty = _envint("MIN_QTY", 1)
    max_qty = _envint("MAX_QTY", 10)

    rps = risk_per_share(symbol, side, last_price)
    risk_budget = equity * risk_pct
    raw = 0 if rps <= 0 else math.floor(risk_budget / rps)

    # Optional strength scaling (if provided)
    try:
        strength = float(os.getenv("STRENGTH", "1.0"))
    except Exception:
        strength = 1.0
    scaled = int(max(0, round(raw * max(0.0, min(1.0, strength)))))

    return int(max(min_qty, min(max_qty, scaled)))


# ---------- Public API (resolve to impl or fallback) ----------
def get_last_price(symbol: str) -> float:
    if _get_last_price_fn:
        return float(_get_last_price_fn(symbol))
    return _yf_last_price(symbol)

def atr_value(symbol: str, period: int, lookback_days: int) -> float:
    if _atr_fn:
        try:
            return float(_atr_fn(symbol, period, lookback_days))
        except TypeError:
            try:
                return float(_atr_fn(symbol, period))
            except TypeError:
                return float(_atr_fn(symbol))
    return _yf_atr(symbol, period, lookback_days)

def risk_per_share(symbol: str, side: str, last_price: float) -> float:
    if _risk_per_share_fn:
        return float(_risk_per_share_fn(symbol, side, last_price))
    return _fallback_risk_per_share(symbol, side, last_price)

def compute_dynamic_qty(symbol: str, side: str, last_price: float) -> int:
    if _compute_dynamic_qty_fn:
        return int(_compute_dynamic_qty_fn(symbol, side, last_price))
    return _fallback_compute_dynamic_qty(symbol, side, last_price)


# ---------- Adaptive submit_bracket wrapper ----------
_PRICE_SYNONYMS = ("limit", "limit_price", "price", "entry", "entry_price")

def submit_bracket(*, symbol: str, side: str, qty: int, last_price: Optional[float] = None, **kwargs):
    """
    Adaptive wrapper:
    - Accepts last_price from caller.
    - Maps it to whichever name the real helper expects (limit/price/entry...).
    - Strips unknown kwargs so we never crash on unexpected names.
    """
    if not _submit_bracket_fn:
        raise NotImplementedError(
            "submit_bracket is not exposed in services.bracket_helper and no fallback is provided."
        )

    sig = inspect.signature(_submit_bracket_fn)
    params = sig.parameters.keys()

    call_kwargs = dict(kwargs)

    # Map last_price → whichever price-like param exists
    if last_price is not None:
        mapped = False
        for name in _PRICE_SYNONYMS:
            if name in params:
                call_kwargs[name] = last_price
                mapped = True
                break
        # If helper doesn't take any price-like param, just ignore last_price.

    # Keep only kwargs the target function accepts
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}

    # Build positional/keyword call safely
    # We always pass symbol, side, qty as keywords if available
    base_kwargs = {}
    if "symbol" in params:
        base_kwargs["symbol"] = symbol
    if "side" in params:
        base_kwargs["side"] = side
    if "qty" in params or "quantity" in params:
        if "qty" in params:
            base_kwargs["qty"] = qty
        else:
            base_kwargs["quantity"] = qty

    return _submit_bracket_fn(**base_kwargs, **call_kwargs)


__all__ = [
    "get_last_price",
    "atr_value",
    "risk_per_share",
    "compute_dynamic_qty",
    "submit_bracket",
]
