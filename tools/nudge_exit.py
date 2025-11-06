import os, sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"
SYMBOL = os.getenv("NUDGE_SYMBOL", "AAPL")

def qt(x, p=2): return round(float(x)+1e-9, p)

def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # find our synthetic TP limit (client_order_id endswith -tp)
    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[SYMBOL]))
    tps = [o for o in od if (o.client_order_id or "").endswith("-tp")]
    if not tps:
        print("no TP leg found to nudge"); sys.exit(0)
    tp = tps[0]

    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=SYMBOL))[SYMBOL]
    bid = float(q.bid_price) if q.bid_price is not None else None
    if bid is None:
        print("no bid available"); sys.exit(0)

    new_px = qt(bid, 2)
    print(f"nudge {SYMBOL} TP: {tp.limit_price} -> {new_px}")
    tc.replace_order_by_id(tp.id, order_data=ReplaceOrderRequest(limit_price=new_px, client_order_id=tp.client_order_id))
    print("done")

if __name__ == "__main__":
    main()
