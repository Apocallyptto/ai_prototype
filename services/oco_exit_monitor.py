# services/oco_exit_monitor.py
import os
import time
import logging

from services.alpaca_exit_guard import (
    get_trading_client,
    has_exit_orders,
    cancel_exit_orders,
    place_exit_oco,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("oco_exit_monitor")


def _symbols_filter() -> set[str] | None:
    raw = os.getenv("SYMBOLS", "").strip()
    if not raw:
        return None
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _price_decimals() -> int:
    try:
        return int(os.getenv("PRICE_DECIMALS", "2"))
    except Exception:
        return 2


def _round_price(x: float) -> float:
    d = _price_decimals()
    return round(float(x), d)


def _ref_price(position) -> float | None:
    """
    Vyber referenčnú cenu:
    - entry (avg_entry_price) alebo current (current_price)
    """
    mode = os.getenv("OCO_REF_PRICE", "entry").strip().lower()

    avg_entry = getattr(position, "avg_entry_price", None)
    current = getattr(position, "current_price", None)

    def f(v):
        try:
            return float(v)
        except Exception:
            return None

    avg_entry_f = f(avg_entry)
    current_f = f(current)

    if mode == "current":
        return current_f or avg_entry_f
    return avg_entry_f or current_f


def _tp_sl_from_pct(ref: float, qty: float) -> tuple[float, float]:
    tp_pct = float(os.getenv("OCO_TP_PCT", "0.015"))  # 1.5%
    sl_pct = float(os.getenv("OCO_SL_PCT", "0.010"))  # 1.0%

    if qty > 0:  # LONG
        tp = ref * (1.0 + tp_pct)
        sl = ref * (1.0 - sl_pct)
    else:        # SHORT
        tp = ref * (1.0 - tp_pct)
        sl = ref * (1.0 + sl_pct)

    return _round_price(tp), _round_price(sl)


def main():
    tc = get_trading_client()
    poll = float(os.getenv("POLL_SECONDS", "5"))
    symbols_filter = _symbols_filter()

    log.info("Started | poll=%ss | symbols_filter=%s", poll, sorted(symbols_filter) if symbols_filter else "ALL")

    while True:
        try:
            positions = tc.get_all_positions()

            for p in positions:
                symbol = str(p.symbol).upper()

                if symbols_filter and symbol not in symbols_filter:
                    continue

                try:
                    qty = float(p.qty)
                except Exception:
                    continue

                if qty == 0:
                    continue

                # 1) Idempotencia: ak už máme EXIT-OCO pre symbol, nerob nič
                if has_exit_orders(tc, symbol):
                    continue

                ref = _ref_price(p)
                if not ref or ref <= 0:
                    log.warning("Skip %s: missing ref price (avg_entry/current).", symbol)
                    continue

                tp, sl = _tp_sl_from_pct(ref, qty)

                # sanity: nech TP/SL dáva zmysel
                if qty > 0 and not (sl < ref < tp):
                    log.warning("Skip %s: invalid LONG tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue
                if qty < 0 and not (tp < ref < sl):
                    log.warning("Skip %s: invalid SHORT tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue

                oid = place_exit_oco(tc, symbol, qty, tp, sl)
                log.info("Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s", symbol, qty, ref, tp, sl, oid)

        except Exception as e:
            log.exception("Monitor loop error: %s", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()
