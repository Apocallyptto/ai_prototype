import os
import time
import uuid
import logging
from datetime import datetime, timezone, timedelta

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


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


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


def _as_dt(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo is not None else x.replace(tzinfo=timezone.utc)
    try:
        s = str(x)
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _round_price(px: float) -> float:
    return round(float(px), 2)


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")

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


def _is_oco(order) -> bool:
    return str(_val(getattr(order, "order_class", None))).lower() == "oco"


def _is_ours_exit(order) -> bool:
    coid = (getattr(order, "client_order_id", "") or "")
    return coid.startswith("exit-oco-")


def _tif_str(order) -> str:
    return str(_val(getattr(order, "time_in_force", ""))).lower()


def _side_str(order) -> str:
    return str(_val(getattr(order, "side", ""))).lower()


def _status_str(order) -> str:
    return str(_val(getattr(order, "status", ""))).lower()


def _qty_norm(x) -> float | None:
    v = _as_float(x)
    if v is None:
        return None
    return abs(float(v))


def _list_oco_parents(tc: TradingClient, symbol: str):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
    orders = tc.get_orders(req) or []
    out = []
    for o in orders:
        if str(getattr(o, "symbol", "") or "").upper() != symbol:
            continue
        if not _is_oco(o):
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

    seen = set()
    uniq = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _extract_tp_sl_from_oco_parent(oco_parent) -> tuple[float | None, float | None]:
    """
    Alpaca OCO representation often looks like you pasted:
      - parent is LIMIT with limit_price = take-profit
      - legs contains 1 STOP (held)
      - parent.take_profit/stop_loss can be None
    So we treat parent.limit_price as TP and STOP leg stop_price as SL.
    """
    tp = None
    sl = None

    # ---- TP candidates ----
    tp = _as_float(getattr(oco_parent, "limit_price", None))

    if tp is None:
        take_profit = getattr(oco_parent, "take_profit", None)
        if take_profit is not None:
            if isinstance(take_profit, dict):
                tp = _as_float(take_profit.get("limit_price"))
            else:
                tp = _as_float(getattr(take_profit, "limit_price", None))

    if tp is None:
        # sometimes TP is a leg
        for leg in (getattr(oco_parent, "legs", None) or []):
            if isinstance(leg, dict):
                typ = str(leg.get("order_type") or leg.get("type") or "").lower()
                if "limit" in typ:
                    tp = _as_float(leg.get("limit_price"))
                    if tp is not None:
                        break
            else:
                typ = str(getattr(leg, "order_type", None) or getattr(leg, "type", None) or "").lower()
                if "limit" in typ:
                    tp = _as_float(getattr(leg, "limit_price", None))
                    if tp is not None:
                        break

    # ---- SL candidates ----
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
        # these may appear as None in the returned model for OCO; that's OK
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
            pass


def _expected_exit_side(position_qty: float) -> str:
    return "sell" if position_qty > 0 else "buy"


def _is_in_flight(order) -> bool:
    """
    Avoid cancel/replace loops when the exit is already executing.
    """
    inflight = {"partially_filled", "pending_cancel", "pending_replace", "pending_review"}
    if _status_str(order) in inflight:
        return True

    for leg in (getattr(order, "legs", None) or []):
        if isinstance(leg, dict):
            st = str(leg.get("status") or "").lower()
        else:
            st = _status_str(leg)
        if st in inflight:
            return True

    return False


def _is_valid_exit_oco_for_position(order, position_qty: float, allow_stop_only: bool) -> tuple[bool, str]:
    if not _is_oco(order):
        return False, "not_oco"

    tif = _tif_str(order)
    if tif != "gtc":
        return False, f"bad_tif:{tif}"

    if not _is_ours_exit(order):
        return False, "not_ours"

    exp_side = _expected_exit_side(position_qty)
    if _side_str(order) != exp_side:
        return False, f"bad_side:{_side_str(order)}!= {exp_side}"

    oq = _qty_norm(getattr(order, "qty", None))
    pq = _qty_norm(position_qty)
    if oq is not None and pq is not None:
        if abs(oq - pq) > 1e-9:
            return False, f"bad_qty:{oq}!= {pq}"

    tp, sl = _extract_tp_sl_from_oco_parent(order)
    if sl is None:
        return False, "missing_sl"
    if (tp is None) and (not allow_stop_only):
        return False, "missing_tp"

    return True, "ok"


def _needs_renew(order, renew_days: int) -> bool:
    if renew_days <= 0:
        return False
    exp = _as_dt(getattr(order, "expires_at", None))
    if exp is None:
        return False
    now = datetime.now(timezone.utc)
    return exp <= (now + timedelta(days=renew_days))


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    poll = _env_float("POLL_SECONDS", 5.0)
    heartbeat = _env_float("HEARTBEAT_SECONDS", 60.0)
    sym_filter = _symbols_filter()

    treat_any_oco_as_protected = _env_bool("OCO_TREAT_ANY_OCO_AS_PROTECTED", True)
    allow_stop_only = _env_bool("OCO_ALLOW_STOP_ONLY", False)
    renew_days = _env_int("OCO_RENEW_DAYS", 3)

    tc = get_trading_client()

    log.info(
        "Started | poll=%.1fs | heartbeat=%.1fs | symbols_filter=%s | treat_any_oco_as_protected=%s | allow_stop_only=%s | renew_days=%s",
        poll,
        heartbeat,
        sym_filter,
        treat_any_oco_as_protected,
        allow_stop_only,
        renew_days,
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

            try:
                stats["open_orders"] = len(
                    tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False, limit=500)) or []
                )
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

                if str(_val(getattr(p, "side", ""))).lower() == "closed":
                    stats["skipped_closing"] += 1
                    continue

                oco_parents = _list_oco_parents(tc, sym)
                ours = [o for o in oco_parents if _is_ours_exit(o)]
                others = [o for o in oco_parents if not _is_ours_exit(o)]

                if ours:
                    def _created(o):
                        return _as_dt(getattr(o, "created_at", None)) or datetime(1970, 1, 1, tzinfo=timezone.utc)

                    ours_sorted = sorted(ours, key=_created, reverse=True)
                    primary = ours_sorted[0]

                    if _is_in_flight(primary):
                        stats["protected"] += 1
                        continue

                    ok, reason = _is_valid_exit_oco_for_position(primary, qty, allow_stop_only)

                    if ok and _needs_renew(primary, renew_days):
                        ok = False
                        reason = f"renew_before_expiry({renew_days}d)"

                    if ok and len(ours_sorted) > 1:
                        ok = False
                        reason = f"duplicate_exits({len(ours_sorted)})"

                    if not ok:
                        tp, sl = _extract_tp_sl_from_oco_parent(primary)
                        if sl is None or (tp is None and not allow_stop_only):
                            ref = _get_ref_price(p)
                            if ref is None:
                                stats["errors"] += 1
                                log.warning("%s: cannot repair exit OCO (%s) because no ref price", sym, reason)
                                stats["unprotected"].append(sym)
                                continue
                            tp, sl = _tp_sl_from_pct(ref, qty)

                        cancel_ids = []
                        for o in ours_sorted:
                            if _is_in_flight(o):
                                cancel_ids = []
                                break
                            cancel_ids.extend(_collect_cancel_ids(o))

                        if not cancel_ids:
                            stats["protected"] += 1
                            continue

                        cancel_ids = list(dict.fromkeys(cancel_ids))

                        log.info(
                            "Repair EXIT OCO | %s | reason=%s | cancel_ids=%s | tp=%s sl=%s",
                            sym,
                            reason,
                            ",".join(cancel_ids),
                            tp,
                            sl,
                        )
                        _cancel_ids(tc, cancel_ids)
                        new_o = _submit_exit_oco(tc, sym, qty, tp, sl, tif=TimeInForce.GTC)
                        stats["repaired"] += 1
                        stats["protected"] += 1
                        log.info(
                            "Repaired EXIT OCO | %s | order_id=%s tif=%s expires_at=%s",
                            sym,
                            getattr(new_o, "id", None),
                            getattr(new_o, "time_in_force", None),
                            getattr(new_o, "expires_at", None),
                        )
                    else:
                        stats["protected"] += 1

                    continue

                if others and treat_any_oco_as_protected:
                    stats["protected"] += 1
                    continue

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
                    "Placed EXIT OCO | %s qty=%s ref=%s tp=%s sl=%s | order_id=%s expires_at=%s",
                    sym,
                    qty,
                    _round_price(ref),
                    tp,
                    sl,
                    getattr(new_o, "id", None),
                    getattr(new_o, "expires_at", None),
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
