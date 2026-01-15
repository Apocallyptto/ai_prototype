import os
import time
import logging
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, OrderClass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

logger = logging.getLogger("reprice_stale")

# --- Tunables (env overrides) ---
STALE_SECONDS = int(os.getenv("REPRICE_STALE_SECONDS", "45"))
CANCEL_AFTER_SECONDS = int(os.getenv("REPRICE_CANCEL_AFTER_SECONDS", "300"))  # 0 = disable
MAX_SPREAD_PCT = float(os.getenv("REPRICE_MAX_SPREAD_PCT", "0.40"))           # percent of mid
MIN_ABS_MOVE = float(os.getenv("REPRICE_MIN_ABS_MOVE", "0.01"))              # dollars
LOOP_SLEEP_SECS = int(os.getenv("REPRICE_LOOP_SLEEP_SECS", "2"))
MAX_REPRICES_PER_ORDER = int(os.getenv("REPRICE_MAX_REPRICES_PER_ORDER", "20"))

TICK = Decimal("0.01")


def make_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    paper = (os.getenv("ALPACA_PAPER", "1").strip().lower() not in ("0", "false", "no"))

    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in env.")

    trading = TradingClient(api_key, api_secret, paper=paper)
    data_client = StockHistoricalDataClient(api_key, api_secret)
    return trading, data_client


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _age_seconds(order) -> float | None:
    ts = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
    if ts is None:
        return None
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (_now_utc() - ts).total_seconds()
    except Exception:
        return None


def _is_child(order) -> bool:
    return getattr(order, "parent_order_id", None) not in (None, "")


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def get_open_parent_limit_orders(trading: TradingClient):
    """Open parent LIMIT/STOP_LIMIT only; SIMPLE or BRACKET. No OCO, no legs."""
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
        orders = trading.get_orders(req) or []
    except TypeError:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = trading.get_orders(req) or []

    parents = []
    for o in orders:
        if _is_child(o):
            continue

        if getattr(o, "order_type", None) not in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            continue

        if getattr(o, "order_class", None) not in (OrderClass.SIMPLE, OrderClass.BRACKET):
            continue

        filled = _as_float(getattr(o, "filled_qty", None)) or 0.0
        if filled > 0:
            continue

        parents.append(o)

    return parents


def get_quote(data_client: StockHistoricalDataClient, symbol: str):
    req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    out = data_client.get_stock_latest_quote(req)
    q = out.get(symbol)
    if q is None or q.bid_price is None or q.ask_price is None:
        return None
    bid = float(q.bid_price)
    ask = float(q.ask_price)
    if bid <= 0 or ask <= 0:
        return None
    return bid, ask


def _round_tick(x: Decimal) -> Decimal:
    return x.quantize(TICK, rounding=ROUND_HALF_UP)


def compute_new_price(order, bid: float, ask: float) -> float | None:
    mid = (bid + ask) / 2.0
    spread_abs = ask - bid
    spread_pct = (spread_abs / mid) * 100.0

    if spread_pct > MAX_SPREAD_PCT:
        logger.info("%s: skip (wide spread %.3f%% bid=%.2f ask=%.2f)", order.symbol, spread_pct, bid, ask)
        return None

    old_px = _as_float(getattr(order, "limit_price", None))
    if old_px is None:
        return None

    if order.side == OrderSide.BUY:
        target = _round_tick(Decimal(str(bid)) + TICK)
        # Do not cross
        if float(target) >= ask:
            return None
        # Only move if we become more aggressive
        if float(target) <= old_px:
            return None
    else:
        target = _round_tick(Decimal(str(ask)) - TICK)
        if float(target) <= bid:
            return None
        if float(target) >= old_px:
            return None

    new_px = float(target)
    if abs(new_px - old_px) < MIN_ABS_MOVE:
        return None

    return round(new_px, 2)


def run_once(trading: TradingClient, data_client: StockHistoricalDataClient,
             reprice_counts: dict[str, int], dry_run: bool):
    orders = get_open_parent_limit_orders(trading)

    if not orders:
        logger.info("no open orders to reprice")
        return

    for o in orders:
        oid = str(getattr(o, "id", ""))
        if not oid:
            continue

        age = _age_seconds(o)
        if age is None or age < STALE_SECONDS:
            continue

        if CANCEL_AFTER_SECONDS > 0 and age >= CANCEL_AFTER_SECONDS:
            logger.info("%s: cancel stale order age=%.0fs id=%s", o.symbol, age, oid)
            if not dry_run:
                try:
                    trading.cancel_order_by_id(oid)
                except Exception as e:
                    logger.error("%s: cancel failed id=%s err=%s", o.symbol, oid, e)
            continue

        count = reprice_counts[oid]
        if count >= MAX_REPRICES_PER_ORDER:
            continue

        quote = get_quote(data_client, o.symbol)
        if not quote:
            continue
        bid, ask = quote

        new_px = compute_new_price(o, bid, ask)
        if new_px is None:
            continue

        count += 1
        reprice_counts[oid] = count

        base_coid = (getattr(o, "client_order_id", None) or "noprefix").split("-rp")[0]
        new_coid = f"{base_coid}-rp{count}"

        logger.info(
            "%s: reprice #%d | %.2f -> %.2f (age=%.0fs bid=%.2f ask=%.2f) coid=%s",
            o.symbol, count, float(o.limit_price), new_px, age, bid, ask, new_coid
        )

        if dry_run:
            continue

        try:
            replace_req = ReplaceOrderRequest(limit_price=new_px, client_order_id=new_coid)
            trading.replace_order_by_id(order_id=oid, order_data=replace_req)
        except Exception as e:
            logger.error("%s: replace failed id=%s err=%s", o.symbol, oid, e)


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    dry_run = os.getenv("DRY_RUN", "0") == "1"
    run_once_mode = os.getenv("RUN_ONCE", "0") == "1"

    trading, data_client = make_clients()
    reprice_counts: dict[str, int] = defaultdict(int)

    logger.info(
        "reprice_stale | start | dry_run=%s run_once=%s stale=%ss cancel_after=%ss",
        dry_run, run_once_mode, STALE_SECONDS, CANCEL_AFTER_SECONDS
    )

    if run_once_mode:
        run_once(trading, data_client, reprice_counts, dry_run=dry_run)
        return

    while True:
        try:
            run_once(trading, data_client, reprice_counts, dry_run=dry_run)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception("reprice loop error: %s", e)
        time.sleep(LOOP_SLEEP_SECS)


if __name__ == "__main__":
    main()
