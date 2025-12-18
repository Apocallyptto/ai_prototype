import os
import time
import logging

from alpaca.trading.client import TradingClient

from services.alpaca_exit_guard import has_exit_orders, place_exit_oco
from utils import compute_atr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("oco_exit_monitor")

POLL_SECONDS = float(os.getenv("OCO_POLL_SECONDS", "5"))
SYMBOLS_FILTER = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ATR_FALLBACK_PCT = float(os.getenv("ATR_PCT", "0.01"))

EXIT_OCO_PREFIX = os.getenv("EXIT_OCO_PREFIX", "EXIT-OCO").strip() or "EXIT-OCO"

tc = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=(os.getenv("TRADING_MODE", "paper") == "paper"),
)


def _is_exit_oco(order) -> bool:
    cid = getattr(order, "client_order_id", None) or ""
    sym = getattr(order, "symbol", None)
    if not sym:
        return False
    return cid.startswith(f"{EXIT_OCO_PREFIX}-{str(sym).upper()}-")


def _has_open_non_exit_orders(symbol: str) -> bool:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    r = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol.upper()], limit=500)
    orders = list(tc.get_orders(r))
    return any(not _is_exit_oco(o) for o in orders)


def _compute_tp_sl(symbol: str, position_qty: float, avg_entry: float):
    atr_val, _last = compute_atr(symbol)
    atr = float(atr_val or 0.0)
    if atr <= 0:
        atr = avg_entry * ATR_FALLBACK_PCT

    if position_qty > 0:
        tp = avg_entry + atr * TP_ATR_MULT
        sl = avg_entry - atr * SL_ATR_MULT
    else:
        tp = avg_entry - atr * TP_ATR_MULT
        sl = avg_entry + atr * SL_ATR_MULT

    return round(tp, 2), round(sl, 2)


def main():
    log.info("Started | poll=%.1fs | symbols_filter=%s", POLL_SECONDS, "ALL" if not SYMBOLS_FILTER else SYMBOLS_FILTER)

    while True:
        try:
            positions = tc.get_all_positions()

            for p in positions:
                sym = str(p.symbol).upper()
                if SYMBOLS_FILTER and sym not in SYMBOLS_FILTER:
                    continue

                try:
                    qty = float(p.qty)
                    avg_entry = float(p.avg_entry_price)
                except Exception:
                    continue

                if qty == 0:
                    continue

                if _has_open_non_exit_orders(sym):
                    continue

                if has_exit_orders(tc, sym):
                    continue

                tp, sl = _compute_tp_sl(sym, qty, avg_entry)
                oid = place_exit_oco(tc, sym, qty, tp, sl)
                log.info("Placed EXIT-OCO | %s qty=%s avg=%.2f -> TP=%.2f SL=%.2f | order_id=%s",
                         sym, int(abs(round(qty))), avg_entry, tp, sl, oid)

        except Exception as e:
            log.exception("monitor error: %s", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
