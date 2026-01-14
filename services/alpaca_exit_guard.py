import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    TimeInForce,
    OrderClass,
)

log = logging.getLogger("alpaca_exit_guard")


FINAL_STATUSES = {"CANCELED", "FILLED", "REJECTED", "EXPIRED", "STOPPED"}

# --- basic helpers -------------------------------------------------------------

def get_tc() -> TradingClient:
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_API_SECRET"),
        paper=True,
    )


def _exit_prefix(symbol: str) -> str:
    return f"EXIT-{(symbol or '').upper()}-"


def _norm_status(st) -> str:
    if st is None:
        return ""
    s = str(st).strip().upper()
    # "OrderStatus.NEW" -> "NEW"
    if "." in s:
        s = s.split(".")[-1]
    return s


def _is_exit_order(symbol: str, order) -> bool:
    """Exit order created by OUR bot via client_order_id prefix EXIT-<SYM>-..."""
    sym = (symbol or "").upper()
    cid = str(getattr(order, "client_order_id", "") or "")
    return cid.startswith(_exit_prefix(sym))


def _norm_enum_name(v) -> str:
    """
    Normalize Alpaca enum-ish fields to a stable uppercase string.
    Works for values like OrderClass.OCO, "oco", None, etc.
    """
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # "OrderClass.OCO" -> "OCO"
    if "." in s:
        s = s.split(".")[-1]
    return s.strip().upper()


def _is_oco_exit_order(symbol: str, o, closing_side=None) -> bool:
    """
    True if the order looks like our exit protection:
    - order_class == OCO (or bracket/oto variants), OR
    - nested parent has legs (typical for OCO in Alpaca),
    and it matches symbol (+ optional closing_side).
    """
    sym = (symbol or "").upper()
    if not sym:
        return False

    osym = str(getattr(o, "symbol", "") or "").upper()
    if osym != sym:
        return False

    if closing_side is not None:
        if _norm_enum_name(getattr(o, "side", None)) != _norm_enum_name(closing_side):
            return False

    oc = _norm_enum_name(getattr(o, "order_class", None))
    if oc in {"OCO", "BRACKET", "OTO"}:
        return True

    # nested=True often returns OCO parent with legs (even if order_class is missing/odd)
    legs = getattr(o, "legs", None) or []
    if len(legs) > 0:
        return True

    return False


# --- order listing -------------------------------------------------------------

def _list_open_orders(tc: TradingClient, symbol: str):
    sym = (symbol or "").upper()
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[sym],
        limit=500,
        nested=True,
    )
    try:
        return tc.get_orders(req) or []
    except TypeError:
        # older alpaca-py may not support nested
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[sym],
            limit=500,
        )
        return tc.get_orders(req) or []


# --- FIX #1 -------------------------------------------------------------------

def list_non_exit_closing_orders(tc: TradingClient, symbol: str, position_qty: float):
    """Return open *closing* orders that are NOT our managed exits (client_id EXIT-* or OCO exits).

    Used to detect "someone/something is already trying to close this position", so the exit monitor
    shouldn't fight it.
    """
    sym = (symbol or "").upper()
    if not sym:
        return []

    closing_side = OrderSide.SELL if float(position_qty) > 0 else OrderSide.BUY

    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[sym],
        limit=500,
        nested=True,
    )

    try:
        orders = tc.get_orders(req) or []
    except TypeError:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[sym],
            limit=500,
        )
        orders = tc.get_orders(req) or []

    out = []
    for o in orders:
        # skip anything that is already our exit protection
        if _is_exit_order(sym, o) or _is_oco_exit_order(sym, o, closing_side=closing_side):
            continue

        # if nested parent contains legs, also treat it as managed exit protection
        legs = getattr(o, "legs", None) or []
        if legs:
            continue

        if getattr(o, "side", None) != closing_side:
            continue

        st = _norm_status(getattr(o, "status", None))
        if st in FINAL_STATUSES:
            continue

        out.append(o)

    return out


# --- exit placement ------------------------------------------------------------

def place_exit_oco(
    tc: TradingClient,
    symbol: str,
    qty_abs: int,
    exit_side: OrderSide,
    tp: float,
    sl: float,
) -> str:
    """Submit OCO exits (TP limit + SL stop). Returns order id."""
    sym = (symbol or "").upper()
    req = LimitOrderRequest(
        symbol=sym,
        qty=qty_abs,
        side=exit_side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.OCO,
        limit_price=tp,
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
        client_order_id=f"{_exit_prefix(sym)}{int(time.time())}",
    )
    o = tc.submit_order(req)
    return str(getattr(o, "id", ""))


def cancel_open_orders(tc: TradingClient, symbol: str, *, side: Optional[OrderSide] = None) -> int:
    sym = (symbol or "").upper()
    orders = _list_open_orders(tc, sym)
    n = 0
    for o in orders:
        if str(getattr(o, "symbol", "")).upper() != sym:
            continue
        if side is not None and getattr(o, "side", None) != side:
            continue
        try:
            tc.cancel_order_by_id(o.id)
            n += 1
        except Exception as e:
            log.warning("cancel_fail %s %s %s", sym, getattr(o, "id", None), e)
    return n


# --- FIX #2 -------------------------------------------------------------------

def has_exit_orders(tc: TradingClient, symbol: str, position_qty: float | None = None) -> bool:
    """True if the symbol already has exit protection orders (EXIT-* client id OR OCO exits)."""
    sym = (symbol or "").upper()
    if not sym:
        return False

    # infer closing side if possible (helps avoid misclassifying non-closing OCOs)
    closing_side = None
    if position_qty is not None:
        closing_side = OrderSide.SELL if float(position_qty) > 0 else OrderSide.BUY
    else:
        try:
            for p in (tc.get_all_positions() or []):
                if str(getattr(p, "symbol", "")).upper() == sym:
                    q = float(getattr(p, "qty", 0) or 0)
                    if q != 0:
                        closing_side = OrderSide.SELL if q > 0 else OrderSide.BUY
                    break
        except Exception:
            closing_side = None

    orders = _list_open_orders(tc, sym)

    for o in orders:
        st = _norm_status(getattr(o, "status", None))
        if st in FINAL_STATUSES:
            continue

        if _is_exit_order(sym, o) or _is_oco_exit_order(sym, o, closing_side=closing_side):
            return True

        # nested=True: also check legs (stop leg is often HELD but still part of the exit)
        for leg in (getattr(o, "legs", None) or []):
            st2 = _norm_status(getattr(leg, "status", None))
            if st2 in FINAL_STATUSES:
                continue
            if _is_exit_order(sym, leg) or _is_oco_exit_order(sym, leg, closing_side=closing_side):
                return True

    return False
