import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide

def main():
    symbol = os.getenv("SYMBOL", (sys.argv[1] if len(sys.argv) > 1 else "AAPL")).upper()
    side_str = os.getenv("SIDE", (sys.argv[2] if len(sys.argv) > 2 else "buy")).lower()
    side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

    tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=os.getenv("ALPACA_PAPER","1")!="0")
    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol]))

    cnt = 0
    for o in od:
        try:
            if getattr(o, "side", None) == side:
                tc.cancel_order_by_id(o.id)
                print("cancelled", o.id, o.client_order_id, o.side, o.type)
                cnt += 1
        except Exception as e:
            print("cancel fail", getattr(o, "id", "?"), e)
    print(f"done, cancelled={cnt}")

if __name__ == "__main__":
    main()
