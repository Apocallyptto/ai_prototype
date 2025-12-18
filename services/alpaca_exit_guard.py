# services/alpaca_exit_guard.py
import os
import time
import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderClass

log = logging.getLogger("alpaca_exit_guard")

EXIT_ORDER_PREFIX = os.getenv("EXIT_ORDER_PREFIX", "EXIT-OCO-").strip()


def _paper() -> bool:
    return os.getenv("TRADING_MODE", "paper").strip().lower() == "paper"


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")
    return TradingClient(key, secret, paper=_paper())


def _tif() -> TimeInForce:
    # default GTC, aby exit ostal aj cez noc (ako u teba v legs: expires 2026-03-18)
    raw = os.getenv("EXIT_TIF", "GTC").strip().upper()
    return TimeInForce(raw)


def get_position_qty(tc: TradingClient, symbol: str) -> int:
    symbol = symbol.upper().strip()
    try:
        p = tc.get_open_position(symbol)
        # qty je string, pri short môže byť záporné
        q = float(p.qty)
        return int(round(q))
    except Exception:
        return 0


def _list_open_orders(tc: TradingClient, symbol: str):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=500)
    orders = tc.get_orders(req)
    return list(orders) if orders is not None else []


def _is_exit_order(order, symbol: Optional[str] = None) -> bool:
    cid = getattr(order, "client_order_id", None) or ""
    if not cid.startswith(EXIT_ORDER_PREFIX):
        return False
    if symbol is None:
        return True
    return cid.startswith(f"{EXIT_ORDER_PREFIX}{symbol.upper()}-")


def list_open_exit_orders(tc: TradingClient, symbol: str) -> List:
    symbol = symbol.upper().strip()
    return [o for o in _list_open_orders(tc, symbol) if _is_exit_order(o, symbol)]


def cancel_exit_orders(tc: TradingClient, symbol: str) -> int:
    """
    Cancel OPEN EXIT-OCO orders for given symbol.
    Returns count of cancel requests.
    """
    symbol = symbol.upper().strip()
    orders = list_open_exit_orders(tc, symbol)
    n = 0
    for o in orders:
        try:
            tc.cancel_order_by_id(str(o.id))
            n += 1
        except Exception as e:
            log.warning("cancel_exit_orders: failed to cancel %s %s: %s", symbol, getattr(o, "id", None), e)
    return n


def wait_exit_orders_cleared(tc: TradingClient, symbol: str, timeout: float = 6.0, poll: float = 0.25) -> bool:
    """
    After cancel request, Alpaca may still show the order as OPEN for a moment.
    Wait until there are no OPEN exit orders, or timeout.
    """
    symbol = symbol.upper().strip()
    deadline = time.time() + timeout

    while time.time() < deadline:
        remaining = list_open_exit_orders(tc, symbol)
        if not remaining:
            return True
        time.sleep(poll)

    # still there
    remaining = list_open_exit_orders(tc, symbol)
    log.warning("wait_exit_orders_cleared: still OPEN after %.1fs for %s: %s",
                timeout, symbol, [str(o.id) for o in remaining])
    return False


def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    qty: int,
    tp_price: float,
    sl_stop_price: float,
    client_order_id: Optional[str] = None,
) -> str:
    """
    Create OCO exit:
      - take profit limit
      - stop loss stop (market on trigger)
    """
    symbol = symbol.upper().strip()
    if qty <= 0:
        raise ValueError("qty must be > 0 for SELL_TO_CLOSE OCO")

    cid = client_order_id or f"{EXIT_ORDER_PREFIX}{symbol}-{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=_tif(),
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_stop_price)),
        client_order_id=cid,
    )
    o = tc.submit_order(req)
    return str(o.id)
