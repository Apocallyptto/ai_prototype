import os
import time
import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.common.exceptions import APIError

from tools.util import market_is_open
from tools.atr import get_atr
from tools.quotes import get_bid_ask_mid

log = logging.getLogger("oco_exit_monitor")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))

POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def _qt(x: float, places: int = 2) -> float:
    return round(float(x) + 1e-9, places)

def _tp_sl_for_position(symbol: str, side: str, entry_ref: float) -> tuple[float, float]:
    # Reuse your ATR source and fallback the same way you already do.
    try:
        atr_val, last_close = get_atr(symbol)
        ref = entry_ref or last_close
    except Exception:
        # modest default if ATR not available
        atr_val = 0.50
        ref = entry_ref

    if side == "long":
        tp = _qt(ref + TP_ATR_MULT * atr_val, 2)
        sl = _qt(max(0.01, ref - SL_ATR_MULT * atr_val), 2)
    else:
        tp = _qt(ref - TP_ATR_MULT * atr_val, 2)
        sl = _qt(ref + SL_ATR_MULT * atr_val, 2)
    return tp, sl

def _has_attached_children(client: TradingClient, symbol: str) -> bool:
    # If any open OCO child exists for the symbol, consider exits present.
    open_orders = client.get_orders(status="open", nested=True, symbols=[symbol])
    for o in open_orders:
        if getattr(o, "order_class", None) == "bracket" or getattr(o, "order_class", None) == "oco":
            return True
        # Some accounts return children separately; nested=True helps, but fallback:
        if getattr(o, "trail_percent", None) or getattr(o, "take_profit", None) or getattr(o, "stop_loss", None):
            return True
    return False

def _submit_oco(client: TradingClient, symbol: str, side: str, qty: float, limit_ref: float, tp: float, sl: float):
    # Build a synthetic parent + OCO children (Alpaca bracket is an easy way to express OCO)
    # We set parent as a tiny “marker” limit around current price to host the OCO. A safer pattern:
    # submit two separate orders: take-profit (limit) and stop-loss (stop). However, bracket
    # creates atomic OCO and is simpler.
    parent_side = OrderSide.SELL if side == "long" else OrderSide.BUY

    req = LimitOrderRequest(
        symbol=symbol,
        side=parent_side,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=_qt(limit_ref, 2),
        qty=qty,
        order_class="bracket",
        take_profit=TakeProfitRequest(limit_price=_qt(tp, 2)),
        stop_loss=StopLossRequest(stop_price=_qt(sl, 2)),
        extended_hours=False,
    )
    if DRY_RUN:
        log.info("[DRY_RUN] would submit OCO bracket for %s qty=%s TP=%.2f SL=%.2f", symbol, qty, tp, sl)
        return
    o = client.submit_order(req)
    log.info("attached OCO exits -> %s parent_id=%s", symbol, o.id)

def run_once(client: TradingClient):
    positions = client.get_all_positions()
    if not positions:
        log.info("no positions")
        return

    for p in positions:
        symbol = p.symbol
        side = "long" if float(p.qty) > 0 else "short"
        qty = abs(float(p.qty))
        avg_entry = float(p.avg_entry_price)

        if _has_attached_children(client, symbol):
            log.info("%s: exits already present", symbol)
            continue

        # Choose a neutral parent limit near current mid as the anchor for bracket children
        bid, ask, mid = get_bid_ask_mid(symbol)
        anchor = _qt(mid if mid else avg_entry, 2)

        tp, sl = _tp_sl_for_position(symbol, side, avg_entry)
        log.info("%s: attaching OCO exits side=%s qty=%.4f | TP=%.2f SL=%.2f", symbol, side, qty, tp, sl)
        try:
            _submit_oco(client, symbol, side, qty, anchor, tp, sl)
        except APIError as e:
            log.warning("attach failed for %s: %s", symbol, e)

def main():
    cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
    # This can also run outside market hours; it only attaches exits.
    while True:
        try:
            run_once(cli)
        except Exception as e:
            log.exception("exit monitor loop error: %s", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
