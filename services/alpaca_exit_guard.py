# services/alpaca_exit_guard.py
from __future__ import annotations

import os
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Any

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


# ---------- helpers ----------

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _round_price(x: float) -> float:
    # equities -> 0.01 tick (2 decimals)
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _get_open_orders(tc: TradingClient, limit: int = 500):
    # nested=True je super (vracia legs). Niektoré verzie alpaca-py to nemusia podporovať.
    try:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=limit, nested=True)) or []
    except TypeError:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=limit)) or []


def _get_position_qty(tc: TradingClient, symbol: str) -> Optional[float]:
    # oco_exit_monitor volá has_exit_orders(tc, symbol) bez qty -> zistíme qty z pozícií
    try:
        positions = tc.get_all_positions() or []
    except Exception:
        return None

    sym_u = symbol.upper()
    for p in positions:
        if str(getattr(p, "symbol", "")).upper() == sym_u:
            return _to_float(getattr(p, "qty", None), default=None)  # type: ignore
    return None


def _cancel_open_orders(tc: TradingClient, symbol: str, side: Optional[OrderSide] = None) -> int:
    sym_u = symbol.upper()
    canceled = 0
    for o in _get_open_orders(tc):
        if str(getattr(o, "symbol", "")).upper() != sym_u:
            continue
        if side is not None and getattr(o, "side", None) != side:
            continue
        try:
            tc.cancel_order_by_id(o.id)
            canceled += 1
        except Exception:
            # nechceme tu padať; monitor beží ďalej
            pass
    return canceled


# ---------- public API (used by oco_exit_monitor) ----------

def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")

    # default paper=True (keď nemáš explictne ALPACA_PAPER=false)
    paper = _env_bool("ALPACA_PAPER", default=True)
    return TradingClient(key, secret, paper=paper)


def has_exit_orders(tc: TradingClient, symbol: str, qty: Optional[int] = None) -> bool:
    """
    True ak už existujú OPEN closing/exit orders pre aktuálnu pozíciu na symbole.

    Pozn:
    - oco_exit_monitor volá len (tc, symbol), takže si zistíme pozíciu a z nej exit side.
    - Ak existuje OCO/BRACKET na closing side -> berieme to ako exit.
    - Pre SIMPLE orders kontrolujeme, či closing qty >= position qty (ak vieme).
    """
    sym_u = symbol.upper()

    pos_qty = _get_position_qty(tc, sym_u)
    if pos_qty is None:
        # ak nevieme zistiť pozíciu, buď bezpečný: povedz "nemám exit"
        return False

    if pos_qty == 0:
        return False

    exit_side = OrderSide.SELL if pos_qty > 0 else OrderSide.BUY
    target_qty = abs(int(pos_qty)) if qty is None else int(abs(qty))

    open_orders = _get_open_orders(tc)

    closing_qty_sum = 0.0

    for o in open_orders:
        if str(getattr(o, "symbol", "")).upper() != sym_u:
            continue
        if getattr(o, "side", None) != exit_side:
            continue

        oc = getattr(o, "order_class", None)
        # ak je to OCO/BRACKET na closing side, považuj to za exit (aj keď nevieme qty)
        if oc in (OrderClass.OCO, OrderClass.BRACKET):
            return True

        # SIMPLE closing (napr. manuálny TP limit alebo stop)
        oqty = _to_float(getattr(o, "qty", 0), 0.0)
        if oqty > 0:
            closing_qty_sum += oqty

    if target_qty <= 0:
        # fallback
        return closing_qty_sum > 0

    return closing_qty_sum >= float(target_qty)


def submit_oco_exits(
    tc: TradingClient,
    symbol: str,
    qty: int,
    exit_side: OrderSide,
    avg_entry_price: float,
    tp_pct: float,
    sl_pct: float,
    tif: TimeInForce = TimeInForce.DAY,
):
    """
    Submitne OCO exit:
      - take_profit limit (ako parent LIMIT)
      - stop_loss stop (ako leg)
    Alpaca v praxi vyžaduje take_profit.limit_price (aj keď parent je limit).
    """
    sym_u = symbol.upper()
    qty_abs = int(abs(qty))
    if qty_abs <= 0:
        raise ValueError(f"qty must be > 0 (got {qty})")
    if avg_entry_price <= 0:
        raise ValueError(f"avg_entry_price must be > 0 (got {avg_entry_price})")

    # Pre long: TP vyššie, SL nižšie
    # Pre short: TP nižšie, SL vyššie
    if exit_side == OrderSide.SELL:
        tp = _round_price(avg_entry_price * (1.0 + float(tp_pct)))
        sl = _round_price(avg_entry_price * (1.0 - float(sl_pct)))
    else:
        tp = _round_price(avg_entry_price * (1.0 - float(tp_pct)))
        sl = _round_price(avg_entry_price * (1.0 + float(sl_pct)))

    # pred submitom zruš staré closing orders na tej istej side (aby sme nemali held qty)
    _cancel_open_orders(tc, sym_u, side=exit_side)

    req = LimitOrderRequest(
        symbol=sym_u,
        qty=qty_abs,
        side=exit_side,
        time_in_force=tif,
        order_class=OrderClass.OCO,

        # parent limit = take profit
        limit_price=tp,

        # alpaca chce explicitne aj take_profit.limit_price
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
    )

    return tc.submit_order(req)
