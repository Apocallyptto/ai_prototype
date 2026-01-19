import os
import time
import uuid
import logging
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderClass,
    OrderSide,
    TimeInForce,
)

try:
    # available in newer alpaca-py
    from alpaca.trading.enums import PositionIntent
except Exception:  # pragma: no cover
    PositionIntent = None

log = logging.getLogger("oco_exit_monitor")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _val(x):
    return getattr(x, "value", x)


def _as_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _round_price(px: float) -> float:
    # US equities are usually 2dp; keep it simple
    return round(float(px), 2)


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")

    # safety-first: default paper=True if not set
    paper = _env_bool("ALPACA_PAPER", default=True)
    return TradingClient(key, secret, paper=paper)


def _tp_sl_from_pct(ref_price: float, position_qty: float) -> tuple[float, float]:
    tp_pct = _env_float("OCO_TP_PCT", 0.015)
    sl_pct = _env_float("OCO_SL_PCT", 0.010)

    if position_qty > 0:  # long -> exits are SELL
        tp = ref_price * (1.0 + tp_pct)
        sl = ref_price * (1.0 - sl_pct)
    else:  # short -> exits are BUY
        tp = ref_price * (1.0 - tp_pct)
        sl = ref_price * (1.0 + sl_pct)

    return _round_price(tp), _round_price(sl)


def _get_ref_price(position) -> float | None:
    # alpaca position has current_price & avg_entry_price (strings)
    for attr in ("current_price", "avg_entry_price"):
        v = _as_float(getattr(position, attr, None))
        if v and v > 0:
            return v
    return None


def _symbols_filter() -> list[str] | None:
    s = (os.getenv("SYMBOLS_FILTER") or "").strip()
    if not s:
        return None
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _list_oco_parents(tc: TradingClient, symbol: str):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
    orders = tc.get_orders(req) or []
    out = []
    for o in orders:
        if getattr(o, "symbol", None) != symbol:
            continue
        if str(_val(getattr(o, "order_class", None))).lower() != "oco":
            continue
        out.append(o)
    return out


def _collect_cancel_ids(oco_parent) -> list[str]:
    ids: list[str] = []
    if getattr(oco_parent, "id", None):
        ids.append(str(oco_parent.id))

    for leg in (getattr(oco_parent, "legs", None) or []):
        if isinstance(leg, dict):
            lid = leg.get("id")
        else:
            lid = getattr(leg, "id", None)
        if lid:
            ids.append(str(lid))

    # de-dupe while preserving order
    seen = set()
    uniq = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _extract_tp_sl_from_oco_parent(oco_parent) -> tuple[float | None, float | None]:
    # Take-profit: parent is typically the LIMIT price
    tp = _as_float(getattr(oco_parent, "limit_price", None))

    # Stop-loss: often present on the STOP leg
    sl = None

    # Some alpaca versions include stop_loss on the parent
    stop_loss = getattr(oco_parent, "stop_loss", None)
    if stop_loss is not None:
        if isinstance(stop_loss, dict):
            sl = _as_float(stop_loss.get("stop_price"))
        else:
            sl = _as_float(getattr(stop_loss, "stop_price", None))

    if sl is None:
        for leg in (getattr(oco_parent, "legs", None) or []):
            if isinstance(leg, dict):
                typ = str(leg.get("order_type") or leg.get("type") or "").lower()
                if "stop" in typ:
                    sl = _as_float(leg.get("stop_price"))
                    if sl is not None:
                        break
            else:
                typ = str(getattr(leg, "order_type", None) or getattr(leg, "type", None) or "").lower()
                if "stop" in typ:
                    sl = _as_float(getattr(leg, "stop_price", None))
                    if sl is not None:
                        break

    if tp is not None:
        tp = _round_price(tp)
    if sl is not None:
        sl = _round_price(sl)

    return tp, sl


def _submit_exit_oco(
    tc: TradingClient,
    symbol: str,
    position_qty: float,
    tp: float,
    sl: float,
    tif: TimeInForce = TimeInForce.GTC,
):
    side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY

    intent = None
    if PositionIntent is not None:
        if position_qty > 0:
            intent = getattr(PositionIntent, "SELL_TO_CLOSE", None)
        else:
            intent = getattr(PositionIntent, "BUY_TO_CLOSE", None)

    qty = abs(float(position_qty))
    qty = int(qty) if float(qty).is_integer() else qty

    kwargs = {}
    if intent is not None:
        kwargs["position_intent"] = intent

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        order_class=OrderClass.OCO,
        time_in_force=tif,
        limit_price=tp,
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
        client_order_id=f"exit-oco-{symbol.lower()}-{uuid.uuid4().hex[:8]}",
        extended_hours=False,
        **kwargs,
    )

    return tc.submit_order(req)


def _cancel_ids(tc: TradingClient, ids: list[str]) -> None:
    for oid in ids:
        try:
            tc.cancel_order_by_id(oid)
        except Exception:
            # already canceled / not cancelable -> ignore
            pass


def _tif_is_day(order) -> bool:
    tif = str(_val(getattr(order, "time_in_force", ""))).lower()
    return tif == "day"


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    poll = _env_float("POLL_SECONDS", 5.0)
    heartbeat = _env_float("HEARTBEAT_SECONDS", 60.0)
    sym_filter = _symbols_filter()

    tc = get_trading_client()

    log.info(
        "Started | poll=%.1fs | heartbeat=%.1fs | symbols_filter=%s",
        poll,
        heartbeat,
        sym_filter,
    )

    last_hb = 0.0

    while True:
        stats = {
            "positions": 0,
            "protected": 0,
            "placed": 0,
            "repaired": 0,
            "skipped_closing": 0,
            "open_orders": 0,
            "errors": 0,
            "unprotected": [],
        }

        try:
            positions = tc.get_all_positions() or []
            stats["positions"] = len(positions)

            # open orders count (for heartbeat)
            try:
                stats["open_orders"] = len(tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False, limit=500)) or [])
            except Exception:
                stats["open_orders"] = 0

            for p in positions:
                sym = getattr(p, "symbol", None)
                if not sym:
                    continue
                sym = str(sym).upper()

                if sym_filter and sym not in sym_filter:
                    continue

                qty = _as_float(getattr(p, "qty", None)) or 0.0
                if qty == 0:
                    continue

                # If the position is being closed, skip placing/repairing exits
                if str(_val(getattr(p, "side", ""))).lower() == "closed":
                    stats["skipped_closing"] += 1
                    continue

                oco_parents = _list_oco_parents(tc, sym)

                if oco_parents:
                    # If any existing OCO is DAY, repair it to GTC but KEEP tp/sl as-is.
                    if any(_tif_is_day(o) for o in oco_parents):
                        tp, sl = _extract_tp_sl_from_oco_parent(oco_parents[0])
                        if tp is None or sl is None:
                            ref = _get_ref_price(p)
                            if ref is None:
                                raise RuntimeError(f"{sym}: cannot compute TP/SL (no ref price)")
                            tp, sl = _tp_sl_from_pct(ref, qty)

                        cancel_ids = []
                        for o in oco_parents:
                            cancel_ids.extend(_collect_cancel_ids(o))

                        # de-dupe
                        cancel_ids = list(dict.fromkeys(cancel_ids))

                        log.info(
                            "Repair EXIT OCO TIF (DAY->GTC) | %s | cancel_ids=%s | keep tp=%s sl=%s",
                            sym,
                            ",".join(cancel_ids),
                            tp,
                            sl,
                        )
                        _cancel_ids(tc, cancel_ids)

                        new_o = _submit_exit_oco(tc, sym, qty, tp, sl, tif=TimeInForce.GTC)
                        stats["repaired"] += 1
                        log.info("Repaired EXIT OCO | %s | order_id=%s tif=%s", sym, getattr(new_o, "id", None), getattr(new_o, "time_in_force", None))

                    stats["protected"] += 1
                    continue

                # No OCO -> place one
                ref = _get_ref_price(p)
                if ref is None:
                    stats["errors"] += 1
                    log.warning("%s: no ref price on position -> cannot place exit OCO", sym)
                    stats["unprotected"].append(sym)
                    continue

                tp, sl = _tp_sl_from_pct(ref, qty)

                new_o = _submit_exit_oco(tc, sym, qty, tp, sl, tif=TimeInForce.GTC)
                stats["placed"] += 1
                stats["protected"] += 1
                log.info(
                    "Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s",
                    sym,
                    qty,
                    _round_price(ref),
                    tp,
                    sl,
                    getattr(new_o, "id", None),
                )

        except Exception:
            stats["errors"] += 1
            log.exception("cycle_error")

        now = time.time()
        if now - last_hb >= heartbeat:
            log.info(
                "Heartbeat | positions=%d protected=%d placed=%d repaired=%d skipped_closing=%d open_orders=%d errors=%d | unprotected=%s",
                stats["positions"],
                stats["protected"],
                stats["placed"],
                stats["repaired"],
                stats["skipped_closing"],
                stats["open_orders"],
                stats["errors"],
                stats["unprotected"],
            )
            last_hb = now

        time.sleep(poll)


if __name__ == "__main__":
    main()
