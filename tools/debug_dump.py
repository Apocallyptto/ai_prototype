import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    clk = tc.get_clock()
    print(f"market is_open={getattr(clk,'is_open',None)} next_open={getattr(clk,'next_open',None)} next_close={getattr(clk,'next_close',None)}")

    positions = tc.get_all_positions()
    print(f"positions: {len(positions)}")
    for p in positions:
        print(f"  {p.symbol} qty={p.qty} avg_entry={p.avg_entry_price}")

    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True))
    print(f"open orders: {len(od)}")
    for o in od:
        print(f"  {o.symbol} {o.side} {o.type} {o.status} lim={getattr(o,'limit_price',None)} coid={o.client_order_id}")

if __name__ == "__main__":
    main()
