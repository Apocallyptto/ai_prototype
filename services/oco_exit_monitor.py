import time
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderClass


FINAL = {"CANCELED", "FILLED", "REJECTED", "EXPIRED"}

def norm_status(st) -> str:
    s = str(st).upper()
    return s.split(".")[-1]  # OrderStatus.CANCELED -> CANCELED

def get_orders(tc, symbols, limit=300, nested=True):
    try:
        return tc.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=symbols, limit=limit, nested=nested)
        )
    except TypeError:
        return tc.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=symbols, limit=limit)
        )

def place_exit_oco(tc, symbol: str, qty: int, tp_price: float, sl_price: float, prefix: str):
    """
    Vytvorí OCO: TP limit + SL stop.
    - zruší existujúce aktívne EXIT-OCO parenty s rovnakým prefixom
    - po submit-e overí legs (SL stop by mal byť HELD)
    """
    # 1) cancel existing active EXIT-OCO parents
    orders = get_orders(tc, [symbol], limit=300, nested=True)
    for o in orders:
        st = norm_status(getattr(o, "status", ""))
        cid = (getattr(o, "client_order_id", "") or "")
        if cid.startswith(prefix) and st not in FINAL:
            tc.cancel_order_by_id(o.id)

    time.sleep(1.5)

    # 2) submit new OCO (kritické: take_profit.limit_price)
    client_id = f"{prefix}{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,

        # Alpaca často reprezentuje OCO tak, že parent je TP limit
        limit_price=tp_price,
        order_class=OrderClass.OCO,

        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),

        client_order_id=client_id,
    )

    created = tc.submit_order(req)

    # 3) verify legs (ak legs chýbajú, hneď zruš parent – nech nezostane “len TP” bez SL)
    full = tc.get_order_by_id(created.id)
    legs = getattr(full, "legs", None)

    if not legs:
        tc.cancel_order_by_id(full.id)
        raise RuntimeError(f"OCO legs missing for {symbol} (canceled parent {full.id})")

    return full  # parent order (s legs)
