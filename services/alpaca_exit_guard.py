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
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    TimeInForce,
    OrderClass,
)

log = logging.getLogger("alpaca_exit_guard")

FINAL_STATUSES = {"CANCELED", "FILLED", "REJECTED", "EXPIRED"}


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("TRADING_MODE", "paper").lower() == "paper"
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars.")
    return TradingClient(key, secret, paper=paper)


def _norm_status(st) -> str:
    s = str(st).upper()
    return s.split(".")[-1]


def _exit_prefix(symbol: str) -> str:
    return f"EXIT-OCO-{symbol.upper()}-"


def _is_exit_order(symbol: str, order) -> bool:
    cid = (getattr(order, "client_order_id", "") or "")
    return cid.startswith(_exit_prefix(symbol))


def get_orders(
    tc: TradingClient,
    symbols: list[str],
    limit: int = 200,
    nested: bool = True,
    status: QueryOrderStatus = QueryOrderStatus.ALL,
):
    try:
        return tc.get_orders(
            GetOrdersRequest(status=status, symbols=symbols, limit=limit, nested=nested)
        )
    except TypeError:
        return tc.get_orders(GetOrdersRequest(status=status, symbols=symbols, limit=limit))


def has_exit_orders(tc: TradingClient, symbol: str) -> bool:
    orders = get_orders(tc, [symbol], limit=200, nested=True, status=QueryOrderStatus.ALL)
    for o in orders:
        st = _norm_status(getattr(o, "status", ""))
        if st in FINAL_STATUSES:
            continue
        if _is_exit_order(symbol, o):
            return True
    return False


def list_non_exit_closing_orders(
    tc: TradingClient,
    symbol: str,
    position_qty: float,
    limit: int = 200,
) -> list:
    """
    Vráti OPEN orders, ktoré by zatvárali pozíciu, ale NIE sú naše EXIT-OCO.

    LONG (qty > 0) -> hľadáme OPEN SELL orders
    SHORT (qty < 0) -> hľadáme OPEN BUY orders
    """
    closing_side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY

    open_orders = get_orders(tc, [symbol], limit=limit, nested=True, status=QueryOrderStatus.OPEN)
    out = []

    for o in open_orders:
        st = _norm_status(getattr(o, "status", ""))
        if st in FINAL_STATUSES:
            continue

        # Ak je to OCO order_class, nechceme to brať ako manuálne zatváranie.
        # (V praxi je to takmer vždy náš EXIT parent/leg.)
        if getattr(o, "order_class", None) == OrderClass.OCO:
            continue

        if _is_exit_order(symbol, o):
            continue

        if getattr(o, "side", None) == closing_side:
            out.append(o)

    return out


def cancel_exit_orders(tc: TradingClient, symbol: str, min_age_sec: int = 0) -> int:
    now = time.time()
    orders = get_orders(tc, [symbol], limit=300, nested=True, status=QueryOrderStatus.ALL)

    canceled = 0
    for o in orders:
        st = _norm_status(getattr(o, "status", ""))
        if st in FINAL_STATUSES:
            continue
        if not _is_exit_order(symbol, o):
            continue

        created_at = getattr(o, "created_at", None)
        age = None
        if created_at is not None:
            try:
                age = now - created_at.timestamp()
            except Exception:
                age = None

        if min_age_sec and age is not None and age < min_age_sec:
            continue

        try:
            tc.cancel_order_by_id(o.id)
            canceled += 1
        except Exception as e:
            log.warning(
                "Failed to cancel EXIT order %s (%s): %s",
                getattr(o, "id", None),
                getattr(o, "client_order_id", None),
                e,
            )

    return canceled


def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    position_qty: float,
    tp_price: float,
    sl_price: float,
    client_prefix: Optional[str] = None,
) -> str:
    """
    Vytvor EXIT OCO pre existujúcu pozíciu.

    Poznámka (Alpaca):
    - pre OCO je potrebné poslať take_profit.limit_price (inak 40010001)
    - parent je LIMIT (TP), SL je leg typu STOP (často HELD)
    """
    qty = int(abs(float(position_qty)))
    if qty <= 0:
        raise ValueError("position_qty must be non-zero")

    side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
    prefix = client_prefix or _exit_prefix(symbol)
    client_id = f"{prefix}{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol.upper(),
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_price)),
        client_order_id=client_id,
    )

    created = tc.submit_order(req)
    return str(created.id)


def wait_exit_orders_cleared(tc, symbol: str, timeout_sec: int = 15, poll_sec: float = 0.5, cancel_first: bool = True, **_):
    """Cancel our EXIT orders for symbol and wait until they are cleared from OPEN orders.
    Returns True if cleared, False on timeout.
    """
    if cancel_first:
        cancel_exit_orders(tc, symbol)

    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if not has_exit_orders(tc, symbol):
            return True
        time.sleep(float(poll_sec))
    return False
