import os
import time
import json
import logging
from typing import Optional, List, Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderClass
from alpaca.common.exceptions import APIError

log = logging.getLogger("alpaca_exit_guard")


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("TRADING_MODE", "paper") == "paper"
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def _prefix() -> str:
    # used in client_order_id:  f"{PREFIX}-{SYMBOL}-{epoch}"
    return os.getenv("EXIT_OCO_PREFIX", "EXIT-OCO").strip() or "EXIT-OCO"


def _is_our_exit_order(order, symbol: str) -> bool:
    cid = getattr(order, "client_order_id", None) or ""
    return cid.startswith(f"{_prefix()}-{symbol.upper()}-")


def list_open_orders(tc: TradingClient, symbol: str) -> List:
    r = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol.upper()],
        limit=500,
    )
    return list(tc.get_orders(r))


def get_position_qty(tc: TradingClient, symbol: str) -> float:
    """
    Returns:
      >0  long qty
      <0  short qty
       0  no position
    """
    symbol = symbol.upper()
    try:
        positions = tc.get_all_positions()
    except Exception:
        return 0.0

    for p in positions:
        if str(p.symbol).upper() != symbol:
            continue
        try:
            return float(p.qty)
        except Exception:
            return 0.0
    return 0.0


def cancel_exit_orders(tc: TradingClient, symbol: str, side_to_cancel: Optional[OrderSide] = None) -> int:
    """
    Cancel ONLY orders that have client_order_id prefix: {EXIT_OCO_PREFIX}-{symbol}-...
    If side_to_cancel is None, cancels both sides.
    Returns count requested to cancel (not necessarily already canceled).
    """
    orders = list_open_orders(tc, symbol)
    to_cancel = []
    for o in orders:
        if not _is_our_exit_order(o, symbol):
            continue
        if side_to_cancel is not None and o.side != side_to_cancel:
            continue
        to_cancel.append(o)

    for o in to_cancel:
        try:
            tc.cancel_order_by_id(str(o.id))
        except Exception as e:
            log.warning("Failed to cancel order %s for %s: %s", o.id, symbol, e)

    return len(to_cancel)


def has_exit_orders(tc: TradingClient, symbol: str) -> bool:
    orders = list_open_orders(tc, symbol)
    return any(_is_our_exit_order(o, symbol) for o in orders)


def wait_exit_orders_cleared(
    tc: TradingClient,
    symbol: str,
    timeout_s: float = 10.0,
    poll_s: float = 0.25,
) -> bool:
    """
    Wait until no EXIT-OCO orders exist for symbol. Returns True if cleared, False on timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if not has_exit_orders(tc, symbol):
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def _extract_related_order_ids_from_apierror(e: Exception) -> List[str]:
    """
    Alpaca-py APIError often stringifies to JSON like:
      {"code":40310000, "message":"...", "related_orders":[...]}
    Try to parse and extract related_orders.
    """
    if not isinstance(e, APIError):
        return []

    msg = str(e).strip()
    if not (msg.startswith("{") and msg.endswith("}")):
        return []

    try:
        data = json.loads(msg)
    except Exception:
        return []

    ids = data.get("related_orders") or []
    out = []
    for x in ids:
        if x is None:
            continue
        out.append(str(x))
    return out


def cancel_orders_by_ids(tc: TradingClient, order_ids: Iterable[str]) -> int:
    n = 0
    for oid in order_ids:
        try:
            tc.cancel_order_by_id(str(oid))
            n += 1
        except Exception as ex:
            log.warning("Failed to cancel related order %s: %s", oid, ex)
    return n


def cancel_related_orders_from_exception(tc: TradingClient, e: Exception) -> int:
    ids = _extract_related_order_ids_from_apierror(e)
    if not ids:
        return 0
    return cancel_orders_by_ids(tc, ids)


def _tif() -> TimeInForce:
    tif = os.getenv("OCO_TIF", "gtc").strip().lower()
    return TimeInForce.GTC if tif == "gtc" else TimeInForce.DAY


def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    position_qty: float,
    tp_price: float,
    sl_stop_price: float,
) -> str:
    """
    Create OCO exit for an existing position.
    LONG qty>0 -> SELL OCO
    SHORT qty<0 -> BUY OCO (buy-to-close)
    """
    symbol = symbol.upper()
    qty = abs(int(round(position_qty)))
    if qty <= 0:
        raise ValueError("position_qty must be non-zero")

    side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
    cid = f"{_prefix()}-{symbol}-{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=_tif(),
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_stop_price)),
        client_order_id=cid,
    )

    o = tc.submit_order(req)
    return str(o.id)
