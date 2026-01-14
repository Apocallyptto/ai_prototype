# services/alpaca_exit_guard.py
from __future__ import annotations

import os
import time
import logging
from typing import List, Optional, Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderClass,
    OrderSide,
    TimeInForce,
)

log = logging.getLogger(__name__)


# -------------------------
# Helper: Trading client
# -------------------------
def get_trading_client() -> TradingClient:
    """
    Centralized Alpaca TradingClient creator (paper by default).
    Env:
      ALPACA_API_KEY
      ALPACA_API_SECRET
      ALPACA_PAPER=true/false (default true)
    """
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")

    paper_raw = (os.getenv("ALPACA_PAPER") or "true").strip().lower()
    paper = paper_raw not in ("0", "false", "no", "off")

    return TradingClient(key, secret, paper=paper)


# -------------------------
# Internal helpers
# -------------------------
def _get_open_orders(tc: TradingClient) -> list:
    """
    Return OPEN orders. Try nested=True if supported by alpaca-py version.
    """
    try:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)) or []
    except TypeError:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)) or []


def _iter_orders_and_legs(orders: list) -> Iterable:
    """
    Yield each order and its legs (if present) as flat stream.
    """
    for o in orders or []:
        yield o
        for leg in (getattr(o, "legs", None) or []):
            yield leg


def _is_exit_order(o) -> bool:
    """
    Our exits are OCO orders (parent or leg). So: order_class == OCO.
    """
    oc = getattr(o, "order_class", None)
    return oc == OrderClass.OCO


def _close_side_from_qty(qty: float) -> OrderSide:
    """
    Long qty>0 -> close via SELL
    Short qty<0 -> close via BUY
    """
    return OrderSide.SELL if qty > 0 else OrderSide.BUY


# -------------------------
# Public API used by oco_exit_monitor
# -------------------------
def has_exit_orders(tc: TradingClient, symbol: str, qty: Optional[float] = None) -> bool:
    """
    MUST be compatible with calls:
      has_exit_orders(tc, symbol)
      has_exit_orders(tc, symbol, qty)

    We detect any OPEN OCO order on that symbol (parent or leg).
    qty is optional (kept for backwards-compatibility / future logic).
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return False

    open_orders = _get_open_orders(tc)

    for o in _iter_orders_and_legs(open_orders):
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue
        if _is_exit_order(o):
            return True

    return False


def list_non_exit_closing_orders(tc: TradingClient, symbol: str, qty: Optional[float] = None) -> List:
    """
    Return OPEN orders that would CLOSE/consume the position qty but are NOT our exit OCO.

    MUST be compatible with calls:
      list_non_exit_closing_orders(tc, symbol, qty)
      list_non_exit_closing_orders(tc, symbol)

    If qty is None, we conservatively return any non-exit OPEN orders on the symbol.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return []

    open_orders = _get_open_orders(tc)

    # Determine which side would close the position.
    closing_side: Optional[OrderSide] = None
    if qty is not None:
        try:
            closing_side = _close_side_from_qty(float(qty))
        except Exception:
            closing_side = None

    res: List = []
    for o in _iter_orders_and_legs(open_orders):
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue

        # ignore our exits
        if _is_exit_order(o):
            continue

        # if qty known: only orders that close the position side
        if closing_side is not None:
            if getattr(o, "side", None) != closing_side:
                continue

        res.append(o)

    return res


def place_exit_oco(tc: TradingClient, symbol: str, qty: float, tp: float, sl: float) -> str:
    """
    Create EXIT OCO for the position:
      - take_profit.limit_price = tp
      - stop_loss.stop_price = sl

    IMPORTANT:
    Alpaca requires for OCO:
      - it must be a LIMIT parent order
      - take_profit.limit_price must be present
    We therefore submit LimitOrderRequest with BOTH:
      limit_price=tp AND take_profit=TakeProfitRequest(limit_price=tp)
    (redundant but it’s what Alpaca accepts reliably)

    Returns submitted order id (string).
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        raise ValueError("symbol empty")

    q = float(qty)
    if q == 0:
        raise ValueError("qty is 0")

    qty_abs = int(abs(q))
    if qty_abs <= 0:
        raise ValueError(f"qty_abs invalid: {qty}")

    exit_side = _close_side_from_qty(q)

    tp = round(float(tp), 2)
    sl = round(float(sl), 2)

    req = LimitOrderRequest(
        symbol=sym,
        qty=qty_abs,
        side=exit_side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.OCO,
        limit_price=tp,  # parent limit (Alpaca wants limit)
        take_profit=TakeProfitRequest(limit_price=tp),  # REQUIRED by Alpaca for OCO
        stop_loss=StopLossRequest(stop_price=sl),
    )

    o = tc.submit_order(req)
    oid = str(getattr(o, "id", "") or "")
    if not oid:
        raise RuntimeError("submit_order returned empty id")
    return oid


# -------------------------
# Optional: small utility (not required by monitor, but handy)
# -------------------------
def wait_until_no_open_exit_orders(tc: TradingClient, symbol: str, timeout_s: int = 20) -> bool:
    """
    Wait a bit until there are no OPEN OCO orders for symbol.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not has_exit_orders(tc, symbol):
            return True
        time.sleep(0.5)
    return False
