# tools/ensure_exits_now.py

import os
import sys

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def _qt(x, p=2):
    return round(float(x) + 1e-9, p)


def _qq(q):
    # keep 0.001 precision like before, avoid negative
    return max(0.0, int(float(q) * 1000 + 1e-9) / 1000.0)


def _is_paper_mode() -> bool:
    """
    Paper mode detection:
    - prefer TRADING_MODE (paper/live)
    - ALPACA_PAPER (1/0) overrides if set
    """
    trading_mode = (os.getenv("TRADING_MODE", "paper") or "paper").lower()
    paper = trading_mode != "live"

    alpaca_paper = os.getenv("ALPACA_PAPER")
    if alpaca_paper is not None:
        paper = alpaca_paper != "0"
    return paper


def main(argv=None):
    argv = argv or sys.argv[1:]
    sym = (argv[0] if len(argv) > 0 else os.getenv("SYMBOL", "AAPL")).upper()

    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_API_SECRET")
    if not k or not s:
        print("ERROR: missing ALPACA_API_KEY / ALPACA_API_SECRET")
        sys.exit(2)

    paper = _is_paper_mode()

    # minimum notional sanity (usually irrelevant for stocks, but keep)
    min_notional = float(os.getenv("MIN_NOTIONAL", "1.00"))

    # simple offset in USD (proxy instead of real ATR)
    # can override via EXIT_OFFSET_USD env
    exit_offset = float(os.getenv("EXIT_OFFSET_USD", "0.50"))

    tc = TradingClient(k, s, paper=paper)
    dc = StockHistoricalDataClient(k, s)

    # Find position
    pos = [p for p in tc.get_all_positions() if p.symbol.upper() == sym]
    if not pos:
        print("no position")
        return

    qty = float(pos[0].qty)
    is_long = qty > 0
    abs_qty = abs(qty)

    # Exit side depends on position direction
    # LONG exits -> SELL
    # SHORT exits -> BUY (buy to cover)
    exit_side = OrderSide.SELL if is_long else OrderSide.BUY

    # How many exit shares are already covered by OPEN orders?
    od = tc.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
            symbols=[sym],
        )
    )
    held = sum(float(o.qty or 0) for o in od if o.side == exit_side)

    free = _qq(abs_qty - held)
    if free <= 0:
        print("exits already present")
        return

    # Anchor price: mid quote if possible, else avg entry
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
    mid = (
        ((q.bid_price or 0) + (q.ask_price or 0)) / 2
        if q.bid_price and q.ask_price
        else float(pos[0].avg_entry_price)
    )

    # Compute TP/SL levels (proxy "ATR" by exit_offset USD)
    if is_long:
        tp = _qt(mid + exit_offset, 2)
        sl = _qt(max(0.01, mid - exit_offset), 2)
    else:
        # SHORT: profit is DOWN, stop is UP
        tp = _qt(max(0.01, mid - exit_offset), 2)
        sl = _qt(mid + exit_offset, 2)

    # Ensure notional sanity (should basically never trigger with real stock prices)
    if tp * free < min_notional:
        tp = _qt(max(0.01, min_notional / free), 2)
    if sl * free < min_notional:
        sl = _qt(max(0.01, min_notional / free), 2)

    side_label = "LONG" if is_long else "SHORT"
    print(f"[ensure_exits_now] {sym} pos={side_label} qty={qty} abs_qty={abs_qty} held={held} free={free}")
    print(f"[ensure_exits_now] mid={mid:.4f} tp={tp} sl={sl} exit_side={exit_side.value} paper={paper}")

    # Place TP (limit) — extended hours allowed
    t = tc.submit_order(
        LimitOrderRequest(
            symbol=sym,
            side=exit_side,
            qty=free,
            time_in_force=TimeInForce.DAY,
            type=OrderType.LIMIT,
            limit_price=tp,
            extended_hours=True,
        )
    )
    print("TP submitted", t.id, tp)

    # Place SL (stop) — extended hours NOT allowed
    u = tc.submit_order(
        StopOrderRequest(
            symbol=sym,
            side=exit_side,
            qty=free,
            time_in_force=TimeInForce.DAY,
            type=OrderType.STOP,
            stop_price=sl,
            extended_hours=False,
        )
    )
    print("SL submitted", u.id, sl)


if __name__ == "__main__":
    main()
