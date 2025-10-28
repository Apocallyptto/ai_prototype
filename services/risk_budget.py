# services/risk_budget.py
from __future__ import annotations
import os
import math
from typing import Optional, Literal, Tuple, List

Side = Literal["buy", "sell"]

# Reuse the stable shim (with yfinance fallbacks) you already have
from services.bracket_public import (
    get_last_price,
    risk_per_share,
)

# ------------- env helpers -------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

EQUITY_DEFAULT = 100_000.0
MAX_RISK_PCT_DEFAULT = 0.01  # 1% of equity across the whole portfolio

def equity() -> float:
    return _env_float("EQUITY", EQUITY_DEFAULT)

def max_portfolio_risk_pct() -> float:
    return _env_float("MAX_PORTFOLIO_RISK_PCT", MAX_RISK_PCT_DEFAULT)

# ------------- optional Alpaca client -------------
def _alpaca_client():
    key = os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        return None
    base = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    try:
        from alpaca_trade_api.rest import REST
    except Exception:
        return None
    return REST(key, sec, base_url=base)

# ------------- core math -------------
def estimate_risk(symbol: str, side: Side, qty: int, last_price: Optional[float] = None) -> float:
    """
    Risk per trade ≈ risk_per_share * qty.
    risk_per_share comes from your ATR SL sizing (via bracket_public).
    """
    if qty <= 0:
        return 0.0
    lp = last_price if last_price is not None else get_last_price(symbol)
    rps = max(1e-8, risk_per_share(symbol, side, lp))
    return float(rps * qty)

def _side_from_qty(q: float) -> Side:
    return "buy" if q >= 0 else "sell"

def _absint(x) -> int:
    try:
        return int(abs(float(x)))
    except Exception:
        return 0

# ------------- measuring active risk -------------
def active_risk_positions() -> Tuple[float, List[str]]:
    """
    Sum estimated risk across open positions (uses current price and ATR SL distance).
    Returns (risk_sum, details[])
    """
    api = _alpaca_client()
    if api is None:
        return 0.0, ["(alpaca not configured → positions not included)"]

    details = []
    total = 0.0
    try:
        positions = api.list_positions()
    except Exception:
        return 0.0, ["(no positions or API error)"]

    for p in positions or []:
        sym = getattr(p, "symbol", "")
        qty = _absint(getattr(p, "qty", 0))
        side = _side_from_qty(getattr(p, "qty", 0))
        if qty <= 0 or not sym:
            continue
        r = estimate_risk(sym, side, qty)
        total += r
        details.append(f"pos {sym} {side} qty={qty} -> risk≈{r:.2f}")
    return float(total), details

def active_risk_open_orders() -> Tuple[float, List[str]]:
    """
    Sum estimated risk for OPEN parent orders (market/limit entries).
    Approximates using order qty and current risk_per_share.
    """
    api = _alpaca_client()
    if api is None:
        return 0.0, ["(alpaca not configured → open orders not included)"]

    details = []
    total = 0.0
    try:
        # status='open' includes new/accepted/partially_filled
        orders = api.list_orders(status="open", nested=True)  # nested for bracket legs
    except Exception:
        return 0.0, ["(no open orders or API error)"]

    for o in orders or []:
        sym = getattr(o, "symbol", "")
        side_raw = str(getattr(o, "side", "buy")).lower()
        side: Side = "buy" if side_raw == "buy" else "sell"
        qty = _absint(getattr(o, "qty", 0) or getattr(o, "quantity", 0))
        if qty <= 0 or not sym:
            continue

        # Only count parent orders (ignore attached TP/SL) — heuristic:
        parent = getattr(o, "order_class", "") in ("bracket", "oco", "otoco") or getattr(o, "parent_order_id", None) is None
        if not parent:
            continue

        r = estimate_risk(sym, side, qty)
        total += r
        details.append(f"ord {sym} {side} qty={qty} -> risk≈{r:.2f}")
    return float(total), details

def active_portfolio_risk() -> Tuple[float, List[str]]:
    """
    Returns current active risk (positions + open parent orders) and detail lines.
    """
    pos_risk, pos_lines = active_risk_positions()
    ord_risk, ord_lines = active_risk_open_orders()
    total = pos_risk + ord_risk
    lines = []
    if pos_lines:
        lines.extend(pos_lines)
    if ord_lines:
        lines.extend(ord_lines)
    return float(total), lines

# ------------- gate -------------
def can_open(symbol: str, side: Side, qty: int) -> Tuple[bool, str]:
    """
    Check if a NEW trade (symbol, side, qty) fits under MAX_PORTFOLIO_RISK_PCT of EQUITY.
    Returns (ok, reason)
    """
    port_equity = equity()
    cap_pct = max(0.0, min(1.0, max_portfolio_risk_pct()))
    cap_abs = port_equity * cap_pct

    active, lines = active_portfolio_risk()
    proposed = estimate_risk(symbol, side, qty)
    after = active + proposed

    if after <= cap_abs:
        return True, (
            f"OK: active_risk≈{active:.2f} + proposed≈{proposed:.2f} "
            f"= {after:.2f} ≤ cap≈{cap_abs:.2f} ({cap_pct*100:.2f}% of equity {port_equity:.2f})"
        )

    detail = " | ".join(lines[:6])  # don’t spam
    return False, (
        f"BLOCK: active_risk≈{active:.2f} + proposed≈{proposed:.2f} "
        f"= {after:.2f} > cap≈{cap_abs:.2f} ({cap_pct*100:.2f}% of equity {port_equity:.2f})"
        + (f" | details: {detail}" if detail else "")
    )
