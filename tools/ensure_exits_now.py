# tools/ensure_exits_now.py
import os
import sys
import time
import random
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def _qt(x, p=2) -> float:
    return round(float(x) + 1e-9, p)


def _qq(q) -> float:
    # keep up to 3 decimals, never negative
    return max(0.0, int(float(q) * 1000 + 1e-9) / 1000.0)


def _mid_quote(dc: StockHistoricalDataClient, sym: str, fallback: float) -> float:
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
    bid = q.bid_price
    ask = q.ask_price
    if bid and ask and bid > 0 and ask > 0:
        return float(bid + ask) / 2.0
    return float(fallback)


def _build_tp_sl(mid: float, atr_pct: float, tp_mult: float, sl_mult: float, is_long: bool):
    # ATR proxy = atr_pct * mid
    atr = max(0.01, float(atr_pct) * float(mid))

    if is_long:
        tp = _qt(mid + atr * tp_mult, 2)
        sl = _qt(max(0.01, mid - atr * sl_mult), 2)
    else:
        tp = _qt(max(0.01, mid - atr * tp_mult), 2)  # take profit below
        sl = _qt(mid + atr * sl_mult, 2)             # stop above

    return tp, sl


def _submit_oco_exit(
    base_url: str,
    api_key: str,
    api_secret: str,
    symbol: str,
    qty: float,
    exit_side: OrderSide,
    tp: float,
    sl: float,
    tif: str = "day",
    stop_limit: bool = False,
):
    """
    Submit ONE OCO exit order:
      - take_profit: limit_price
      - stop_loss: stop_price (optionally also limit_price for stop-limit)
    This avoids "held_for_orders" problems from placing 2 separate orders.
    """
    url = base_url.rstrip("/") + "/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json",
    }

    payload = {
        "symbol": symbol,
        "qty": str(_qq(qty)),
        "side": "sell" if exit_side == OrderSide.SELL else "buy",
        "type": "limit",
        "time_in_force": tif,
        "order_class": "oco",
        "take_profit": {"limit_price": str(_qt(tp, 2))},
        "stop_loss": {"stop_price": str(_qt(sl, 2))},
        "client_order_id": f"EXIT-OCO-{symbol}-{int(time.time())}-{random.randrange(1000,9999)}",
    }

    # optional stop-limit (more controlled fill than stop-market)
    if stop_limit:
        payload["stop_loss"]["limit_price"] = str(_qt(sl, 2))

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def main():
    sym = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("SYMBOL", "AAPL")).upper()

    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_API_SECRET")
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper = os.getenv("ALPACA_PAPER", "1") != "0"

    if not k or not s:
        print("ERROR: ALPACA_API_KEY / ALPACA_API_SECRET not set")
        sys.exit(2)

    # config knobs (use your env defaults)
    atr_pct = float(os.getenv("ATR_PCT", "0.0100"))
    tp_mult = float(os.getenv("TP_ATR_MULT", "1.0"))
    sl_mult = float(os.getenv("SL_ATR_MULT", "1.0"))
    tif = os.getenv("EXIT_TIF", "day").lower()  # day | gtc
    stop_limit = os.getenv("EXIT_STOP_LIMIT", "0") == "1"

    tc = TradingClient(k, s, paper=paper)
    dc = StockHistoricalDataClient(k, s)

    # find position
    pos = [p for p in tc.get_all_positions() if p.symbol.upper() == sym]
    if not pos:
        print("no position")
        return

    p = pos[0]
    qty = float(p.qty)
    abs_qty = abs(qty)
    is_long = qty > 0
    exit_side = OrderSide.SELL if is_long else OrderSide.BUY

    # detect if exits already exist (any OPEN OCO/BRACKET with our EXIT-OCO cid, or any open order holding full qty)
    od = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym]))
    # "held_for_orders" problem happens when we already have exit order(s) for full qty
    held_same_side = sum(float(o.qty or 0) for o in od if getattr(o, "side", None) == exit_side)

    free = _qq(abs_qty - held_same_side)

    print(f"[ensure_exits_now] {sym} pos={'LONG' if is_long else 'SHORT'} qty={qty} abs_qty={abs_qty} held={held_same_side} free={free}")

    if free <= 0:
        print("exits already present")
        return

    # anchor mid
    mid = _mid_quote(dc, sym, fallback=float(p.avg_entry_price))
    tp, sl = _build_tp_sl(mid, atr_pct, tp_mult, sl_mult, is_long)

    print(f"[ensure_exits_now] mid={mid:.4f} atr_pct={atr_pct:.4f} tp={tp:.2f} sl={sl:.2f} exit_side={exit_side.value} tif={tif} paper={paper}")

    # submit OCO exit (single request -> no 403 held_for_orders)
    out = _submit_oco_exit(
        base_url=base,
        api_key=k,
        api_secret=s,
        symbol=sym,
        qty=free,
        exit_side=exit_side,
        tp=tp,
        sl=sl,
        tif=tif,
        stop_limit=stop_limit,
    )

    print("OCO submitted:", out.get("id"), "status:", out.get("status"), "order_class:", out.get("order_class"))


if __name__ == "__main__":
    main()
