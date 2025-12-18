# tools/cancel_exit_oco_parents.py
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

SYMBOL = "AAPL"
PREFIX = f"EXIT-OCO-{SYMBOL}-"

def norm_status(st):
    s = str(st).upper()
    return s.split(".")[-1]  # OrderStatus.CANCELED -> CANCELED

orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[SYMBOL], limit=300))

active = []
for o in orders:
    cid = (getattr(o, "client_order_id", "") or "")
    st = norm_status(getattr(o, "status", ""))
    if cid.startswith(PREFIX) and st not in ("CANCELED", "FILLED", "REJECTED", "EXPIRED"):
        active.append(o)

print("Active EXIT-OCO parent orders:", len(active))
for o in active:
    print(o.id, norm_status(o.status), o.side, o.type, "qty", getattr(o, "qty", None), "client", o.client_order_id)
    tc.cancel_order_by_id(o.id)

print("Done.")
