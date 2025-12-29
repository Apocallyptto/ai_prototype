import os, time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce

tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

print("=== CANCEL OPEN ORDERS ===")
orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
print("Open orders:", len(orders))
for o in orders:
    try:
        tc.cancel_order_by_id(o.id)
        print(" canceled", o.symbol, o.id)
    except Exception as e:
        print(" cancel_failed", o.symbol, o.id, e)

time.sleep(2)

print("\n=== CLOSE POSITIONS (market) ===")
pos = tc.get_all_positions()
print("Positions:", len(pos))
for p in pos:
    sym = p.symbol
    qty = abs(float(p.qty))
    side = str(p.side).upper()
    close_side = OrderSide.BUY if "SHORT" in side else OrderSide.SELL
    try:
        o = tc.submit_order(MarketOrderRequest(
            symbol=sym, qty=qty, side=close_side, time_in_force=TimeInForce.DAY
        ))
        print(" close_submitted", sym, "qty", qty, "side", close_side, "order_id", getattr(o,"id",None))
    except Exception as e:
        print(" close_failed", sym, e)

print("\nDONE")
