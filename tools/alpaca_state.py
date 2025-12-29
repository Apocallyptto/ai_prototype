import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

print("=== ACCOUNT ===")
a = tc.get_account()
print("cash=", a.cash)
print("buying_power=", a.buying_power)
print("equity=", a.equity)
print("shorting_enabled=", getattr(a,"shorting_enabled",None))
print("trading_blocked=", getattr(a,"trading_blocked",None))

print("\n=== OPEN ORDERS ===")
orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
print("OPEN ORDERS:", len(orders))
for o in orders:
    print(o.symbol, o.side, "qty=", o.qty, "type=", getattr(o,"order_type",None),
          "limit=", getattr(o,"limit_price",None), "stop=", getattr(o,"stop_price",None),
          "status=", o.status, "id=", o.id, "submitted_at=", getattr(o,"submitted_at",None))

print("\n=== POSITIONS ===")
pos = tc.get_all_positions()
print("POSITIONS:", len(pos))
for p in pos:
    print(p.symbol, p.side, p.qty, "avg=", getattr(p,"avg_entry_price",None))
