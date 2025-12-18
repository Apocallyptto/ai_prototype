# services/oco_exit_monitor.py
import os
import time
import logging

from services.alpaca_exit_guard import (
    get_trading_client,
    has_exit_orders,
    list_non_exit_closing_orders,
    place_exit_oco,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("oco_exit_monitor")


def _symbols_filter():
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
    return round(float(x), _price_decimals())


def _ref_price(position):
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


def _tp_sl_from_pct(ref: float, qty: float):
    tp_pct = float(os.getenv("OCO_TP_PCT", "0.015"))  # 1.5%
    sl_pct = float(os.getenv("OCO_SL_PCT", "0.010"))  # 1.0%

    if qty > 0:  # LONG
        tp = ref * (1.0 + tp_pct)
        sl = ref * (1.0 - sl_pct)
    else:  # SHORT
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

                # ✅ kľúč: keď už existuje NON-EXIT closing order (napr. pending SELL MARKET),
                # tak je qty held_for_orders a EXIT-OCO sa nesmie zakladať.
                closing = list_non_exit_closing_orders(tc, symbol, qty)
                if closing:
                    ids = ",".join(str(getattr(o, "id", "")) for o in closing)
                    log.info("Skip %s: non-exit closing order(s) already open -> %s", symbol, ids)
                    continue

                if has_exit_orders(tc, symbol):
                    continue

                ref = _ref_price(p)
                if not ref or ref <= 0:
                    log.warning("Skip %s: missing ref price (avg_entry/current).", symbol)
                    continue

                tp, sl = _tp_sl_from_pct(ref, qty)

                # sanity
                if qty > 0 and not (sl < ref < tp):
                    log.warning("Skip %s: invalid LONG tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue
                if qty < 0 and not (tp < ref < sl):
                    log.warning("Skip %s: invalid SHORT tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue

                try:
                    oid = place_exit_oco(tc, symbol, qty, tp, sl)
                    log.info("Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s", symbol, qty, ref, tp, sl, oid)
                except Exception as e:
                    msg = str(e)
                    # fallback: keď sa aj tak trafí race-condition
                    if "40310000" in msg and "insufficient qty available for order" in msg:
                        log.info("Skip %s: qty held_for_orders (pending close order).", symbol)
                    else:
                        raise

        except Exception as e:
            log.exception("Monitor loop error: %s", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()
