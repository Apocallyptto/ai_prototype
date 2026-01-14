# services/alpaca_exit_guard.py
import os
import time
import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    TimeInForce,
    OrderClass,
)

log = logging.getLogger("alpaca_exit_guard")


# ----------------------------
# Helper: Trading client
# ----------------------------
def get_trading_client(*, paper: Optional[bool] = None) -> TradingClient:
    """
    Central factory for TradingClient so services can share the same logic.
    Defaults to PAPER unless explicitly overridden.

    Env fallback:
      ALPACA_PAPER=true/false (optional)
    """
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in environment.")

    if paper is None:
        paper_env = (os.getenv("ALPACA_PAPER") or "true").strip().lower()
        paper = paper_env in ("1", "true", "yes", "y", "on")

    return TradingClient(key, sec, paper=paper)


# ----------------------------
# Helper: orders fetch (nested compatible)
# ----------------------------
def _get_orders(tc: TradingClient, *, status: QueryOrderStatus, limit: int = 500, nested: bool = True):
    """
    Some alpaca-py versions support nested=True, some don't.
    This helper tries nested first, then falls back.
    """
    if nested:
        try:
            return tc.get_orders(GetOrdersRequest(status=status, limit=limit, nested=True)) or []
        except TypeError:
            return tc.get_orders(GetOrdersRequest(status=status, limit=limit)) or []
    return tc.get_orders(GetOrdersRequest(status=status, limit=limit)) or []


def _exit_side_for_qty(qty: float) -> OrderSide:
    # qty > 0 => long => exit is SELL
    # qty < 0 => short => exit is BUY
    return OrderSide.SELL if qty > 0 else OrderSide.BUY


def _is_exit_oco_order(o, exit_side: OrderSide) -> bool:
    """
    Treat any OPEN OCO on the correct side as "exit order exists".
    With nested=True, parent contains legs; with nested=False, we may only see parent.
    """
    if getattr(o, "order_class", None) != OrderClass.OCO:
        return False
    if getattr(o, "side", None) != exit_side:
        return False
    # Parent LIMIT with TP price is typical; stop leg is HELD.
    return True


# ----------------------------
# REQUIRED by oco_exit_monitor.py
# ----------------------------
def has_exit_orders(tc: TradingClient, symbol: str, qty: float) -> bool:
    """
    True if there is already an OPEN OCO exit order for this symbol and position side.
    Signature matches oco_exit_monitor: (tc, symbol, qty)
    """
    sym = (symbol or "").upper()
    exit_side = _exit_side_for_qty(float(qty))

    open_orders = _get_orders(tc, status=QueryOrderStatus.OPEN, limit=500, nested=True)

    for o in open_orders:
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue

        # If nested=True, OCO should appear as parent with legs (or as OCO items depending on API response).
        if _is_exit_oco_order(o, exit_side):
            return True

        # Defensive: sometimes legs can appear; if so, also accept them as "exit exists"
        for leg in (getattr(o, "legs", None) or []):
            if str(getattr(leg, "symbol", "")).upper() != sym:
                continue
            if _is_exit_oco_order(leg, exit_side):
                return True

    return False


def list_non_exit_closing_orders(tc: TradingClient, symbol: str, qty: float) -> List[str]:
    """
    Returns IDs of OPEN orders that are "closing" this position (same exit side)
    BUT are NOT the OCO exit orders we want to keep.

    Signature matches oco_exit_monitor: (tc, symbol, qty)
    """
    sym = (symbol or "").upper()
    exit_side = _exit_side_for_qty(float(qty))

    open_orders = _get_orders(tc, status=QueryOrderStatus.OPEN, limit=500, nested=True)

    ids: List[str] = []
    for o in open_orders:
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue

        side = getattr(o, "side", None)
        if side != exit_side:
            # not a closing order for this position
            continue

        # KEEP any OCO (parent or leg) for this symbol/side
        if getattr(o, "order_class", None) == OrderClass.OCO:
            continue

        # Sometimes legs might appear separately; if they are OCO, keep them too
        parent_id = getattr(o, "parent_order_id", None)
        if parent_id and getattr(o, "order_class", None) == OrderClass.OCO:
            continue

        oid = getattr(o, "id", None)
        if oid:
            ids.append(str(oid))

    return ids


def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    qty: float,
    tp_price: float,
    sl_price: float,
    tif: str = "day",
):
    """
    Submit an OCO exit:
      - Parent LIMIT (take profit)
      - stop_loss leg

    IMPORTANT: Alpaca requires take_profit.limit_price for OCO.
    We provide BOTH limit_price and take_profit.limit_price (works with their API behavior you saw).
    """
    sym = (symbol or "").upper()
    q = float(qty)
    qty_abs = int(abs(q))
    if qty_abs <= 0:
        raise ValueError(f"place_exit_oco: qty_abs <= 0 for {sym}: qty={qty}")

    exit_side = _exit_side_for_qty(q)

    tif_enum = TimeInForce.DAY if str(tif).lower() == "day" else TimeInForce.GTC

    req = LimitOrderRequest(
        symbol=sym,
        qty=qty_abs,
        side=exit_side,
        time_in_force=tif_enum,
        order_class=OrderClass.OCO,
        limit_price=float(tp_price),
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_price)),
    )
    return tc.submit_order(req)


# (Optional helpers – not required by oco_exit_monitor, but handy)
def cancel_open_orders_for_symbol(tc: TradingClient, symbol: str) -> int:
    sym = (symbol or "").upper()
    open_orders = _get_orders(tc, status=QueryOrderStatus.OPEN, limit=500, nested=False)
    n = 0
    for o in open_orders:
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue
        oid = getattr(o, "id", None)
        if oid:
            tc.cancel_order_by_id(oid)
            n += 1
    return n


def wait_until_no_open_orders(tc: TradingClient, symbol: str, timeout_sec: float = 15.0, poll_sec: float = 0.5) -> bool:
    sym = (symbol or "").upper()
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        open_orders = _get_orders(tc, status=QueryOrderStatus.OPEN, limit=500, nested=False)
        if not any(str(getattr(o, "symbol", "")).upper() == sym for o in open_orders):
            return True
        time.sleep(float(poll_sec))
    return False
