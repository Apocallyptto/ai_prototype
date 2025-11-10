import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

def _qt(x, p=2): return round(float(x)+1e-9, p)
def _qq(q): return max(0.0, int(float(q)*1000+1e-9)/1000.0)

def main():
    sym = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("SYMBOL", "AAPL")).upper()
    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("ALPACA_PAPER","1") != "0"
    min_notional = float(os.getenv("MIN_NOTIONAL","1.00"))

    tc = TradingClient(k, s, paper=paper)
    dc = StockHistoricalDataClient(k, s)

    pos = [p for p in tc.get_all_positions() if p.symbol.upper() == sym]
    if not pos:
        print("no position")
        return
    qty = float(pos[0].qty)

    od = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym]))
    sell_held = sum(float(o.qty or 0) for o in od if o.side == OrderSide.SELL)
    free = _qq(qty - sell_held)
    if free <= 0:
        print("exits already present")
        return

    # rough anchor
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
    mid = ((q.bid_price or 0)+(q.ask_price or 0))/2 if q.bid_price and q.ask_price else float(pos[0].avg_entry_price)

    # simple 0.5 ATR proxy
    tp = _qt(mid + 0.5, 2)
    sl = _qt(max(0.01, mid - 0.5), 2)

    # ensure notional
    if tp*free < min_notional:
        tp = _qt(min_notional/free, 2)
    if sl*free < min_notional:
        sl = _qt(min_notional/free, 2)

    # place TP (AH allowed)
    t = tc.submit_order(LimitOrderRequest(symbol=sym, side=OrderSide.SELL, qty=free,
                                          time_in_force=TimeInForce.DAY, type=OrderType.LIMIT,
                                          limit_price=tp, extended_hours=True))
    print("TP submitted", t.id, tp)

    # place SL (no extended hours!)
    u = tc.submit_order(StopOrderRequest(symbol=sym, side=OrderSide.SELL, qty=free,
                                         time_in_force=TimeInForce.DAY, type=OrderType.STOP,
                                         stop_price=sl, extended_hours=False))
    print("SL submitted", u.id, sl)

if __name__ == "__main__":
    main()
