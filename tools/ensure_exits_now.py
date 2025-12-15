# tools/ensure_exits_now.py
import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, TimeInForce, OrderClass
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

def _qt(x, p=2):  # price quantize
    return round(float(x) + 1e-9, p)

def _qq(q):       # qty quantize
    return max(0.0, int(float(q) * 1000 + 1e-9) / 1000.0)

def mid_price(dc: StockHistoricalDataClient, sym: str, fallback: float) -> float:
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
    if q.bid_price and q.ask_price:
        return float((q.bid_price + q.ask_price) / 2)
    return float(fallback)

def main():
    sym = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("SYMBOL", "AAPL")).upper()

    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("TRADING_MODE", "paper").lower() == "paper"

    atr_pct = float(os.getenv("ATR_PCT", "0.01"))
    tp_mult = float(os.getenv("TP_ATR_MULT", "1.0"))
    sl_mult = float(os.getenv("SL_ATR_MULT", "1.0"))
    min_notional = float(os.getenv("MIN_NOTIONAL", "1.00"))

    tc = TradingClient(k, s, paper=paper)
    dc = StockHistoricalDataClient(k, s)

    # find position
    pos = [p for p in tc.get_all_positions() if p.symbol.upper() == sym]
    if not pos:
        print("no position")
        return

    p = pos[0]
    qty = float(p.qty)             # can be negative for short
    abs_qty = abs(qty)
    side = "LONG" if qty > 0 else "SHORT"

    exit_side = OrderSide.SELL if qty > 0 else OrderSide.BUY  # sell-to-close for long, buy-to-close for short

    # how much is already held by OPEN exit orders
    od = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym]))
    held = sum(float(o.qty or 0) for o in od if o.side == exit_side)  # parent qty counts for OCO
    free = _qq(abs_qty - held)

    print(f"[ensure_exits_now] {sym} pos={side} qty={qty} abs_qty={abs_qty} held={held} free={free}")
    if free <= 0:
        print("exits already present")
        return

    mid = mid_price(dc, sym, fallback=float(p.avg_entry_price))
    atr = mid * atr_pct

    if qty > 0:
        # LONG: TP above, SL below
        tp = _qt(mid + atr * tp_mult, 2)
        sl = _qt(max(0.01, mid - atr * sl_mult), 2)
    else:
        # SHORT: TP below (buy cheaper), SL above
        tp = _qt(max(0.01, mid - atr * tp_mult), 2)
        sl = _qt(mid + atr * sl_mult, 2)

    # ensure minimum notional (avoid tiny orders)
    if tp * free < min_notional:
        tp = _qt(min_notional / free, 2)
    if sl * free < min_notional:
        sl = _qt(min_notional / free, 2)

    print(f"[ensure_exits_now] mid={mid:.4f} atr_pct={atr_pct:.4f} tp={tp} sl={sl} exit_side={exit_side.value} tif=day paper={paper}")

    # Submit OCO: parent is LIMIT, and we attach stop_loss leg
    req = LimitOrderRequest(
        symbol=sym,
        qty=free,
        side=exit_side,
        time_in_force=TimeInForce.DAY,
        type=OrderType.LIMIT,
        limit_price=tp,
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
    )

    o = tc.submit_order(req)
    print("OCO submitted:", o.id, "status:", o.status, "order_class:", o.order_class)

if __name__ == "__main__":
    main()
