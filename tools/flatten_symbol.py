import os, sys, time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderType

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"
SYMBOL = os.getenv("FLAT_SYMBOL", "AAPL")

def cancel_all_sells(tc, sym):
    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym]))
    for o in od:
        if str(getattr(o, "side", "")).lower().endswith("sell"):
            try:
                tc.cancel_order_by_id(o.id)
                print(f"cancelled {o.client_order_id}")
            except Exception as e:
                print(f"cancel fail {o.client_order_id}: {e}")

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    pos = {p.symbol: p for p in tc.get_all_positions()}
    if SYMBOL not in pos:
        print("no position")
        return
    qty = abs(float(pos[SYMBOL].qty))
    side = OrderSide.SELL if float(pos[SYMBOL].qty) > 0 else OrderSide.BUY
    print(f"flatten {SYMBOL} qty={qty} via MARKET {side}")

    cancel_all_sells(tc, SYMBOL)

    req = MarketOrderRequest(symbol=SYMBOL, side=side, type=OrderType.MARKET, time_in_force=TimeInForce.DAY, qty=qty)
    tc.submit_order(req)
    print("submitted flatten")

if __name__ == "__main__":
    main()
