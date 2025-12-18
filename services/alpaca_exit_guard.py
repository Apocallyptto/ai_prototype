# services/alpaca_exit_guard.py
import os
import time
import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderClass

log = logging.getLogger("alpaca_exit_guard")


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars")

    paper = os.getenv("TRADING_MODE", "paper").strip().lower() == "paper"
    return TradingClient(key, secret, paper=paper)


def _prefix() -> str:
    # prefix WITHOUT symbol, e.g. "EXIT-OCO-"
    return os.getenv("EXIT_ORDER_PREFIX", "EXIT-OCO-")


def _is_our_exit_order(order, symbol: str) -> bool:
    cid = (getattr(order, "client_order_id", None) or "")
    return cid.startswith(f"{_prefix()}{symbol}-")


def list_open_orders(tc: TradingClient, symbol: Optional[str] = None, limit: int = 500):
    if symbol:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=limit)
    else:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=limit)
    return tc.get_orders(req)


def get_position_qty(tc: TradingClient, symbol: str) -> float:
    try:
        p = tc.get_open_position(symbol)
        return float(p.qty)
    except Exception:
        return 0.0


def cancel_exit_orders(tc: TradingClient, symbol: str, side_to_cancel: Optional[OrderSide] = None) -> int:
    """
    Zruší iba orders s client_order_id prefixom: EXIT-OCO-{symbol}-*
    Pozn.: OPEN list typicky vráti parent OCO. Zrušenie parenta zruší aj HELD stop leg.
    """
    orders = list_open_orders(tc, symbol)
    to_cancel = []
    for o in orders:
        if not _is_our_exit_order(o, symbol):
            continue
        if side_to_cancel is not None and getattr(o, "side", None) != side_to_cancel:
            continue
        to_cancel.append(o)

    for o in to_cancel:
        try:
            tc.cancel_order_by_id(str(o.id))
        except Exception as e:
            log.warning("Failed to cancel order %s for %s: %s", getattr(o, "id", None), symbol, e)

    return len(to_cancel)


def has_exit_orders(tc: TradingClient, symbol: str) -> bool:
    orders = list_open_orders(tc, symbol)
    return any(_is_our_exit_order(o, symbol) for o in orders)


def _tif() -> TimeInForce:
    tif = os.getenv("OCO_TIF", "gtc").strip().lower()
    return TimeInForce.GTC if tif == "gtc" else TimeInForce.DAY


def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    position_qty_signed: float,
    tp_price: float,
    sl_stop_price: float,
) -> str:
    """
    Vytvorí OCO exit pre existujúcu pozíciu.
    - LONG  -> SELL OCO
    - SHORT -> BUY OCO (buy-to-close)

    Kritické: Alpaca vyžaduje take_profit.limit_price (inak: "oco orders require take_profit.limit_price")
    """
    if position_qty_signed == 0:
        raise ValueError("position_qty_signed is 0 (nothing to protect)")

    side = OrderSide.SELL if position_qty_signed > 0 else OrderSide.BUY
    qty = abs(float(position_qty_signed))
    cid = f"{_prefix()}{symbol}-{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=_tif(),

        # Parent = TP limit (Alpaca to takto reprezentuje)
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,

        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_stop_price)),

        client_order_id=cid,
    )

    o = tc.submit_order(req)
    return str(o.id)
