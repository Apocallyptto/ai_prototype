from alpaca.trading import GetOrdersRequest
from alpaca.trading.enums import OrderClass, QueryOrderStatus


def _is_exit_order(symbol: str, o) -> bool:
    sym = str(getattr(o, "symbol", "")).upper()
    if sym != symbol.upper():
        return False

    # 1) tvoja pôvodná logika (napr. client_order_id prefix) nech ostane
    coid = (getattr(o, "client_order_id", None) or "")
    if coid.startswith("EXIT_") or coid.startswith("OCO_EXIT_"):
        return True

    # 2) FALLBACK: ak je to OCO, ber to ako exit (aj keď je manuálne vytvorené)
    if getattr(o, "order_class", None) == OrderClass.OCO:
        return True

    return False


def has_exit_orders(tc, symbol: str) -> bool:
    # ideálne kontroluj OPEN, nie ALL
    try:
        orders = tc.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol], limit=200, nested=True
        )) or []
    except TypeError:
        orders = tc.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol], limit=200
        )) or []

    for o in orders:
        if _is_exit_order(symbol, o):
            return True

    return False
