import os
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

log = logging.getLogger(__name__)

__all__ = [
    "get_trading_client",
    "has_exit_orders",
    "list_non_exit_closing_orders",
    "place_exit_oco",
]


def get_trading_client(paper: Optional[bool] = None) -> TradingClient:
    """
    Single place to build Alpaca TradingClient (used by oco_exit_monitor).
    Defaults to paper trading unless explicitly overridden.
    """
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars")

    if paper is None:
        # default paper unless user explicitly sets ALPACA_PAPER=false/0/no
        v = (os.getenv("ALPACA_PAPER", "true") or "true").strip().lower()
        paper = v not in ("0", "false", "no")

    return TradingClient(key, secret, paper=paper)


def _get_open_orders(tc: TradingClient, symbol: Optional[str] = None):
    """
    Fetch OPEN orders; tries nested=True (legs) when supported by SDK.
    """
    try:
        orders = tc.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
        ) or []
    except TypeError:
        orders = tc.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        ) or []

    if symbol:
        sym = str(symbol).upper()
        orders = [o for o in orders if str(getattr(o, "symbol", "")).upper() == sym]
    return orders


def _is_exit_oco_like(order) -> bool:
    """
    True if order looks like an exit bracket/OCO (has order_class OCO/BRACKET or legs).
    """
    oc = getattr(order, "order_class", None)
    if oc in (OrderClass.OCO, OrderClass.BRACKET):
        return True
    legs = getattr(order, "legs", None) or []
    return len(legs) > 0


def has_exit_orders(tc: TradingClient, symbol: str) -> bool:
    """
    Used by oco_exit_monitor.py as: has_exit_orders(tc, symbol)
    Returns True if there is any OPEN OCO/BRACKET (or legged) exit order for symbol.
    """
    for o in _get_open_orders(tc, symbol):
        if _is_exit_oco_like(o):
            return True
    return False


def list_non_exit_closing_orders(tc: TradingClient, symbol: str, qty: float):
    """
    Used by oco_exit_monitor.py as: list_non_exit_closing_orders(tc, symbol, qty)

    Returns OPEN orders that would close the position but are NOT exit OCO/BRACKET.
    (Example: manual SELL limit to close long, but not OCO.)
    """
    if qty == 0:
        return []

    exit_side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    out = []
    for o in _get_open_orders(tc, symbol):
        # must match side that would close the position
        if getattr(o, "side", None) != exit_side:
            continue
        # exclude true exit OCO/BRACKET orders
        if _is_exit_oco_like(o):
            continue
        out.append(o)
    return out


def _qty_abs(qty: float):
    """
    Alpaca accepts int for whole shares; for safety keep float if fractional.
    """
    q = abs(float(qty))
    if q.is_integer():
        return int(q)
    return q


def place_exit_oco(tc: TradingClient, symbol: str, qty: float, tp: float, sl: float):
    """
    Used by oco_exit_monitor.py as: place_exit_oco(tc, symbol, qty, tp, sl)
    Places OCO exit: take-profit LIMIT + stop-loss STOP.
    tp/sl are absolute prices (already computed by monitor).
    """
    if qty == 0:
        raise ValueError("place_exit_oco: qty=0 (nothing to protect)")

    symbol = str(symbol).upper()
    q = _qty_abs(qty)
    exit_side = OrderSide.SELL if qty > 0 else OrderSide.BUY

    tp = float(tp)
    sl = float(sl)
    if tp <= 0 or sl <= 0:
        raise ValueError(f"place_exit_oco: invalid tp/sl: tp={tp} sl={sl}")

    # basic sanity
    if qty > 0 and not (tp > sl):
        raise ValueError(f"place_exit_oco: long expects tp>sl, got tp={tp} sl={sl}")
    if qty < 0 and not (tp < sl):
        raise ValueError(f"place_exit_oco: short expects tp<sl, got tp={tp} sl={sl}")

    req = LimitOrderRequest(
        symbol=symbol,
        qty=q,
        side=exit_side,
        time_in_force=TimeInForce.GTC,  # <-- CHANGE: DAY -> GTC
        order_class=OrderClass.OCO,
        # Alpaca requires take_profit.limit_price for OCO; keep both to satisfy SDK/API
        limit_price=tp,
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
    )

    o = tc.submit_order(req)
    log.info("exit_oco_submitted | %s qty=%s side=%s tp=%s sl=%s id=%s",
             symbol, q, exit_side, tp, sl, getattr(o, "id", None))
    return o
