import os
import time
import logging

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderClass

from services.alpaca_exit_guard import get_trading_client, place_exit_oco

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


def _val(x) -> str:
    if x is None:
        return ""
    if hasattr(x, "value"):
        try:
            return str(x.value).lower()
        except Exception:
            pass
    return str(x).lower()


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
    tp_pct = float(os.getenv("OCO_TP_PCT", "0.015"))
    sl_pct = float(os.getenv("OCO_SL_PCT", "0.010"))

    if qty > 0:  # LONG
        tp = ref * (1.0 + tp_pct)
        sl = ref * (1.0 - sl_pct)
    else:  # SHORT
        tp = ref * (1.0 - tp_pct)
        sl = ref * (1.0 + sl_pct)

    return _round_price(tp), _round_price(sl)


def _side_is_close(order_side, position_qty: float) -> bool:
    want = "sell" if position_qty > 0 else "buy"
    v = _val(order_side)
    return v == want or v.endswith(want)


def _is_exit_like(order) -> bool:
    ocv = _val(getattr(order, "order_class", None))
    if ocv in ("oco", "bracket"):
        return True
    legs = getattr(order, "legs", None) or []
    return len(legs) > 0


def _is_parent_with_legs(order) -> bool:
    legs = getattr(order, "legs", None)
    if legs is None:
        return False
    try:
        return len(legs) > 0
    except Exception:
        return True


def _get_open_orders(tc):
    try:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)) or []
    except TypeError:
        return tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)) or []


def _orders_by_symbol(orders):
    by = {}
    for o in orders:
        sym = str(getattr(o, "symbol", "")).upper()
        if sym:
            by.setdefault(sym, []).append(o)
    return by


def _list_non_exit_closing_orders(orders_for_symbol, position_qty: float):
    out = []
    for o in orders_for_symbol or []:
        if not _side_is_close(getattr(o, "side", None), position_qty):
            continue
        if _is_exit_like(o):
            continue
        out.append(o)
    return out


def _list_oco_exit_parents(orders_for_symbol, position_qty: float):
    out = []
    for o in orders_for_symbol or []:
        if _val(getattr(o, "order_class", None)) != "oco":
            continue
        if not _side_is_close(getattr(o, "side", None), position_qty):
            continue
        if not _is_parent_with_legs(o):
            continue
        out.append(o)
    return out


def _needs_tif_repair(order) -> bool:
    if _val(getattr(order, "time_in_force", None)) == "day":
        return True
    for leg in (getattr(order, "legs", None) or []):
        tif = leg.get("time_in_force") if isinstance(leg, dict) else getattr(leg, "time_in_force", None)
        if _val(tif) == "day":
            return True
    return False


def _collect_cancel_ids(order):
    ids = []
    pid = getattr(order, "id", None)
    if pid:
        ids.append(str(pid))
    for leg in (getattr(order, "legs", None) or []):
        lid = leg.get("id") if isinstance(leg, dict) else getattr(leg, "id", None)
        if lid:
            ids.append(str(lid))
    # unique
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _cancel_ids(tc, ids, symbol: str):
    for oid in ids:
        try:
            tc.cancel_order_by_id(oid)
        except Exception as e:
            log.info("Cancel failed %s id=%s err=%s", symbol, oid, e)


def main():
    tc = get_trading_client()
    poll = float(os.getenv("POLL_SECONDS", "5"))
    heartbeat = float(os.getenv("HEARTBEAT_SECONDS", "60"))
    symbols_filter = _symbols_filter()

    log.info("Started | poll=%ss | heartbeat=%ss | symbols_filter=%s",
             poll, heartbeat, sorted(symbols_filter) if symbols_filter else "ALL")

    last_hb = time.monotonic()

    while True:
        loop_started = time.monotonic()
        stats = {"positions": 0, "protected": 0, "repaired": 0, "placed": 0, "skipped_closing": 0, "errors": 0}
        unprotected = []

        try:
            positions = tc.get_all_positions() or []
            open_orders = _get_open_orders(tc)
            by_symbol = _orders_by_symbol(open_orders)

            stats["positions"] = len(positions)

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

                orders_sym = by_symbol.get(symbol, [])

                # 1) ak už existuje non-exit closing order, nič nezakladaj
                closing = _list_non_exit_closing_orders(orders_sym, qty)
                if closing:
                    stats["skipped_closing"] += 1
                    continue

                # 2) ak existuje EXIT OCO parent, skontroluj TIF a prípadne oprav
                oco_parents = _list_oco_exit_parents(orders_sym, qty)
                if oco_parents:
                    if any(_needs_tif_repair(o) for o in oco_parents):
                        ids = []
                        for o in oco_parents:
                            ids.extend(_collect_cancel_ids(o))
                        log.info("Repair EXIT OCO TIF (DAY->GTC) | %s | cancel_ids=%s", symbol, ",".join(ids))
                        _cancel_ids(tc, ids, symbol)
                        stats["repaired"] += 1

                        ref = _ref_price(p)
                        if ref and ref > 0:
                            tp, sl = _tp_sl_from_pct(ref, qty)
                            try:
                                place_exit_oco(tc, symbol, qty, tp, sl)
                                stats["placed"] += 1
                            except Exception as e:
                                stats["errors"] += 1
                                log.exception("Repair place_exit_oco failed %s: %s", symbol, e)
                    else:
                        stats["protected"] += 1
                    continue

                # 3) ak existuje iný exit-like (napr BRACKET), ber to ako protected
                if any(_is_exit_like(o) and _side_is_close(getattr(o, "side", None), qty) for o in orders_sym):
                    stats["protected"] += 1
                    continue

                # 4) inak založ nový EXIT OCO
                ref = _ref_price(p)
                if not ref or ref <= 0:
                    unprotected.append(symbol)
                    continue

                tp, sl = _tp_sl_from_pct(ref, qty)

                if qty > 0 and not (sl < ref < tp):
                    unprotected.append(symbol)
                    continue
                if qty < 0 and not (tp < ref < sl):
                    unprotected.append(symbol)
                    continue

                try:
                    oid = place_exit_oco(tc, symbol, qty, tp, sl)
                    log.info("Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s",
                             symbol, qty, ref, tp, sl, oid)
                    stats["placed"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    log.exception("place_exit_oco error %s: %s", symbol, e)
                    unprotected.append(symbol)

            if time.monotonic() - last_hb >= heartbeat:
                last_hb = time.monotonic()
                log.info(
                    "Heartbeat | positions=%d protected=%d placed=%d repaired=%d skipped_closing=%d open_orders=%d errors=%d | unprotected=%s",
                    stats["positions"], stats["protected"], stats["placed"], stats["repaired"],
                    stats["skipped_closing"], len(open_orders), stats["errors"],
                    unprotected if unprotected else "[]",
                )

        except Exception as e:
            stats["errors"] += 1
            log.exception("Monitor loop error: %s", e)

        elapsed = time.monotonic() - loop_started
        time.sleep(max(0.0, poll - elapsed))


if __name__ == "__main__":
    main()
