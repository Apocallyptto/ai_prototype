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
    return round(float(x), _price_decimals())


def _ref_price(position) -> float | None:
    """
    Referenčná cena:
    - "entry" (avg_entry_price) alebo "current" (current_price)
    """
    mode = os.getenv("OCO_REF_PRICE", "entry").strip().lower()

    def f(v):
        try:
            return float(v)
        except Exception:
            return None

    avg_entry = f(getattr(position, "avg_entry_price", None))
    current = f(getattr(position, "current_price", None))

    if mode == "current":
        return current or avg_entry
    return avg_entry or current


def _signed_qty_from_position(p) -> float | None:
    """
    Alpaca často vracia qty kladné a smer je v p.side.
    Takže si spravíme signed qty:
      LONG  -> +qty
      SHORT -> -qty
    """
    try:
        qty = abs(float(getattr(p, "qty", 0)))
    except Exception:
        return None

    side = getattr(p, "side", None)
    side_name = getattr(side, "name", str(side)).upper()

    if "SHORT" in side_name:
        return -qty
    return qty


def _tp_sl_from_pct(ref: float, signed_qty: float) -> tuple[float, float]:
    tp_pct = float(os.getenv("OCO_TP_PCT", "0.015"))  # 1.5%
    sl_pct = float(os.getenv("OCO_SL_PCT", "0.010"))  # 1.0%

    if signed_qty > 0:  # LONG
        tp = ref * (1.0 + tp_pct)
        sl = ref * (1.0 - sl_pct)
    else:               # SHORT
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
            pos_map = {str(p.symbol).upper(): p for p in positions}

            symbols_to_process = set(pos_map.keys())
            if symbols_filter:
                symbols_to_process = symbols_filter

            for symbol in sorted(symbols_to_process):
                p = pos_map.get(symbol)

                # 1) Ak nemáme pozíciu, zruš orphan EXIT-OCO
                if p is None:
                    if has_exit_orders(tc, symbol):
                        n = cancel_exit_orders(tc, symbol)
                        log.info("Canceled orphan EXIT-OCO | %s | canceled=%d", symbol, n)
                    continue

                signed_qty = _signed_qty_from_position(p)
                if not signed_qty:
                    continue

                # 2) Idempotencia: už máme EXIT-OCO -> nerob nič
                if has_exit_orders(tc, symbol):
                    continue

                ref = _ref_price(p)
                if not ref or ref <= 0:
                    log.warning("Skip %s: missing ref price (avg_entry/current).", symbol)
                    continue

                tp, sl = _tp_sl_from_pct(ref, signed_qty)

                # 3) sanity
                if signed_qty > 0 and not (sl < ref < tp):
                    log.warning("Skip %s: invalid LONG tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue
                if signed_qty < 0 and not (tp < ref < sl):
                    log.warning("Skip %s: invalid SHORT tp/sl ref=%s tp=%s sl=%s", symbol, ref, tp, sl)
                    continue

                oid = place_exit_oco(tc, symbol, signed_qty, tp, sl)
                log.info("Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s", symbol, signed_qty, ref, tp, sl, oid)

        except Exception as e:
            log.exception("Monitor loop error: %s", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()
