# services/alpaca_exit_guard.py
import os
import time
import logging
import datetime as dt
from typing import List, Optional, Tuple

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


def _norm_status(st) -> str:
    s = str(st).upper()
    return s.split(".")[-1]  # OrderStatus.CANCELED -> CANCELED


def _csv_env(name: str, default: str) -> List[str]:
    raw = (os.getenv(name) or default).strip()
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _get_orders_all(tc: TradingClient, symbols: List[str], limit: int = 200, nested: bool = True):
    """
    nested=True niekedy pomôže, ale nie všetky verzie alpaca-py ho majú.
    """
    try:
        return tc.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                symbols=symbols,
                limit=limit,
                nested=nested,
            )
        )
    except TypeError:
        return tc.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                symbols=symbols,
                limit=limit,
            )
        )


def _get_orders_open(tc: TradingClient, symbols: List[str], limit: int = 200):
    return tc.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=symbols,
            limit=limit,
        )
    )


def _tif() -> TimeInForce:
    tif = (os.getenv("OCO_TIF") or "gtc").strip().lower()
    if tif == "day":
        return TimeInForce.DAY
    return TimeInForce.GTC


def _exit_prefix() -> str:
    return (os.getenv("EXIT_PREFIX") or "EXIT-OCO-").strip()


def _price_step() -> float:
    try:
        return float(os.getenv("PRICE_STEP") or "0.01")
    except Exception:
        return 0.01


def round_to_step(price: float, step: float) -> float:
    if step <= 0:
        return float(price)
    return round(round(float(price) / step) * step, 8)


def list_active_exit_orders(
    tc: TradingClient,
    symbol: str,
    limit: int = 300,
) -> List[object]:
    """
    Nájde všetky ACTIVE (non-final) orders, ktoré vyzerajú ako EXIT:
    - order_class == OCO  (parent aj niektoré legy)
    - alebo client_order_id začína na EXIT prefix + SYMBOL-
    """
    prefix = _exit_prefix()
    sym_u = symbol.upper()

    orders = _get_orders_all(tc, [sym_u], limit=limit, nested=True)

    active = []
    for o in orders:
        st = _norm_status(getattr(o, "status", ""))
        if st in FINAL_STATUSES:
            continue

        cid = (getattr(o, "client_order_id", "") or "")
        oc = getattr(o, "order_class", None)

        is_oco = str(oc).upper().endswith("OCO")
        is_ours = cid.startswith(f"{prefix}{sym_u}-")

        if is_oco or is_ours:
            active.append(o)

    return active


def has_exit_orders(tc: TradingClient, symbol: str) -> bool:
    """
    Rýchly check: OPEN parent OCO sa zvyčajne objaví v OPEN.
    Pre istotu doplníme aj ALL(active) filter.
    """
    sym_u = symbol.upper()

    # 1) OPEN parenty (najlacnejšie)
    oo = _get_orders_open(tc, [sym_u], limit=100)
    for o in oo:
        oc = getattr(o, "order_class", None)
        cid = (getattr(o, "client_order_id", "") or "")
        if str(oc).upper().endswith("OCO") or cid.startswith(f"{_exit_prefix()}{sym_u}-"):
            return True

    # 2) fallback: ALL(active)
    return len(list_active_exit_orders(tc, sym_u, limit=200)) > 0


def cancel_exit_orders(tc: TradingClient, symbol: str, dry_run: bool = False) -> int:
    """
    Zruší active EXITy pre symbol.
    Preferujeme zrušiť parent OCO (tým sa zrušia aj legs).
    """
    sym_u = symbol.upper()
    active = list_active_exit_orders(tc, sym_u, limit=400)

    # parenty prvé
    parents = []
    others = []
    for o in active:
        oc = getattr(o, "order_class", None)
        if str(oc).upper().endswith("OCO"):
            parents.append(o)
        else:
            others.append(o)

    canceled = 0
    for bucket in (parents, others):
        for o in bucket:
            oid = str(getattr(o, "id", ""))
            try:
                if dry_run:
                    log.info("DRY_RUN cancel_exit_orders | would_cancel=%s", oid)
                else:
                    tc.cancel_order_by_id(oid)
                canceled += 1
            except Exception as e:
                log.warning("cancel_exit_orders failed | id=%s | err=%s", oid, e)

    return canceled


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

    sym_u = symbol.upper()
    qty = abs(float(position_qty_signed))

    side = OrderSide.SELL if position_qty_signed > 0 else OrderSide.BUY

    step = _price_step()
    tp_price = round_to_step(tp_price, step)
    sl_stop_price = round_to_step(sl_stop_price, step)

    prefix = _exit_prefix()
    cid = f"{prefix}{sym_u}-{int(time.time())}"

    req = LimitOrderRequest(
        symbol=sym_u,
        qty=qty,
        side=side,
        time_in_force=_tif(),

        # Parent = TP limit (Alpaca to takto reprezentuje)
        limit_price=float(tp_price),
        order_class=OrderClass.OCO,

        # !!! toto je povinné pre OCO:
        take_profit=TakeProfitRequest(limit_price=float(tp_price)),
        stop_loss=StopLossRequest(stop_price=float(sl_stop_price)),

        client_order_id=cid,
    )

    o = tc.submit_order(req)

    # bonus robustnosť: over legs (niekedy sa objavia po chvíľke)
    try:
        time.sleep(0.5)
        full = tc.get_order_by_id(str(o.id))
        legs = getattr(full, "legs", None)
        if not legs:
            # malá šanca race condition -> nechaj, ale zaloguj
            log.warning("OCO submitted but legs missing (yet) | parent_id=%s | client=%s", o.id, cid)
    except Exception:
        pass

    return str(o.id)
