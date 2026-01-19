import os
import time
import logging
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


logger = logging.getLogger("reprice_stale")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ---------- ENV / CONFIG ----------
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
RUN_ONCE = os.getenv("RUN_ONCE", "0") == "1"

# Reprice only if order is older than this many seconds
STALE_SECONDS = int(os.getenv("REPRICE_STALE_SECONDS", "45"))

# Cancel (instead of reprice) if order is older than this many seconds (0 disables)
CANCEL_AFTER_SECONDS = int(os.getenv("REPRICE_CANCEL_AFTER_SECONDS", "300"))

# Loop sleep when market open / normal mode
LOOP_SLEEP_SECS = float(os.getenv("REPRICE_LOOP_SLEEP_SECONDS", "2"))

# Skip repricing if spread too wide (percent of mid, e.g. 0.40 = 0.40%)
MAX_SPREAD_PCT = float(os.getenv("REPRICE_MAX_SPREAD_PCT", "0.40"))

# Skip repricing if quote timestamp is too old
QUOTE_MAX_AGE_SECONDS = int(os.getenv("REPRICE_QUOTE_MAX_AGE_SECONDS", "60"))

# Market-open gate
REQUIRE_MARKET_OPEN = os.getenv("REPRICE_REQUIRE_MARKET_OPEN", "1") == "1"
CLOSED_MARKET_SLEEP_SECONDS = int(os.getenv("REPRICE_CLOSED_MARKET_SLEEP_SECONDS", "60"))


# ---------- HELPERS ----------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt_maybe(dt) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        s = str(dt)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def order_age_seconds(o) -> float | None:
    ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
    dt = parse_dt_maybe(ts)
    if dt is None:
        return None
    return (utc_now() - dt).total_seconds()


def quote_fields(q) -> tuple[float, float, datetime | None]:
    bid = getattr(q, "bid_price", None)
    ask = getattr(q, "ask_price", None)

    if bid is None:
        bid = getattr(q, "b", None)
    if ask is None:
        ask = getattr(q, "a", None)

    ts = getattr(q, "timestamp", None)
    if ts is None:
        ts = getattr(q, "t", None)

    bid_f = float(bid) if bid is not None else 0.0
    ask_f = float(ask) if ask is not None else 0.0
    ts_dt = parse_dt_maybe(ts)

    return bid_f, ask_f, ts_dt


def quote_age_seconds(q_ts: datetime | None) -> float | None:
    if q_ts is None:
        return None
    return (utc_now() - q_ts).total_seconds()


def make_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars.")
    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)
    return trading, data


def is_market_open(trading: TradingClient) -> bool:
    try:
        c = trading.get_clock()
        return bool(getattr(c, "is_open", False))
    except Exception:
        # fail-open: ak clock zlyhá, radšej bež “po starom”
        return True


def is_parent_simple_limit(o) -> bool:
    ot = getattr(o, "type", None) or getattr(o, "order_type", None)
    if ot is None or str(ot).lower() != str(OrderType.LIMIT).lower():
        return False

    # ignore child/legs if SDK provides parent_order_id
    if getattr(o, "parent_order_id", None):
        return False

    # ignore OCO/Bracket/OTO etc
    oc = getattr(o, "order_class", None)
    if oc is not None:
        s = str(oc).upper()
        if "OCO" in s or "BRACKET" in s or "OTO" in s:
            return False

    return True


def get_open_parent_limit_orders(trading: TradingClient):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=False)
    orders = trading.get_orders(req) or []
    return [o for o in orders if is_parent_simple_limit(o)]


def compute_new_limit_price(order, bid: float, ask: float) -> float | None:
    if bid <= 0 or ask <= 0:
        return None
    if ask < bid:
        return None

    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None

    spread_pct = (ask - bid) / mid * 100.0
    if spread_pct > MAX_SPREAD_PCT:
        logger.info(
            "%s: skip (wide spread %.3f%% bid=%.2f ask=%.2f)",
            getattr(order, "symbol", "?"),
            spread_pct,
            bid,
            ask,
        )
        return None

    side = getattr(order, "side", None)
    if side is None:
        return None

    # Buy more aggressive at ask, sell more aggressive at bid
    if str(side).lower() == str(OrderSide.BUY).lower():
        return float(ask)
    if str(side).lower() == str(OrderSide.SELL).lower():
        return float(bid)

    return None


def floats_close(a: float, b: float, tick: float = 0.01) -> bool:
    return abs(a - b) < (tick / 2.0)


# ---------- CORE ----------
def run_once(trading: TradingClient, data: StockHistoricalDataClient, dry_run: bool) -> float | None:
    """
    One scan:
      1) cancel old orders (CANCEL_AFTER_SECONDS) ALWAYS (even when market closed)
      2) reprice only when market open (if REQUIRE_MARKET_OPEN=1)
    Returns an optional sleep override (e.g. CLOSED_MARKET_SLEEP_SECONDS).
    """
    orders = get_open_parent_limit_orders(trading)

    # 1) CANCEL PASS (allowed even when market closed)
    canceled = 0
    canceled_ids: set[str] = set()

    if CANCEL_AFTER_SECONDS > 0 and orders:
        for o in orders:
            oid = str(getattr(o, "id", "") or "")
            if not oid:
                continue
            age = order_age_seconds(o)
            if age is None:
                continue
            if age >= CANCEL_AFTER_SECONDS:
                sym = getattr(o, "symbol", "?")
                logger.info("%s: cancel (age=%.0fs >= %ss) id=%s", sym, age, CANCEL_AFTER_SECONDS, oid)
                canceled += 1
                canceled_ids.add(oid)
                if not dry_run:
                    try:
                        trading.cancel_order_by_id(oid)
                    except Exception as e:
                        logger.error("%s: cancel failed id=%s err=%s", sym, oid, e)

    # Remove canceled from further processing
    if canceled_ids:
        orders = [o for o in orders if str(getattr(o, "id", "") or "") not in canceled_ids]

    # 2) MARKET GATE (skip repricing when market is closed)
    if REQUIRE_MARKET_OPEN:
        market_open = is_market_open(trading)
        if not market_open:
            if orders or canceled:
                logger.info(
                    "market closed -> skipping reprices (open_orders=%d canceled=%d) -> sleeping %ss",
                    len(orders),
                    canceled,
                    CLOSED_MARKET_SLEEP_SECONDS,
                )
            else:
                logger.info("market closed -> idle (no orders) -> sleeping %ss", CLOSED_MARKET_SLEEP_SECONDS)
            return float(CLOSED_MARKET_SLEEP_SECONDS)

    # If no orders (or all got canceled), we're done
    if not orders:
        if canceled == 0:
            logger.info("no open orders to reprice")
        return None

    # 3) REPRICE PASS (only when market open or gate disabled)
    repriced = 0

    for o in orders:
        oid = str(getattr(o, "id", "") or "")
        if not oid:
            continue

        sym = getattr(o, "symbol", "?")

        age = order_age_seconds(o)
        if age is None or age < STALE_SECONDS:
            continue

        # quote
        try:
            q = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym)).get(sym)
        except Exception as e:
            logger.error("%s: quote fetch failed err=%s", sym, e)
            continue

        if not q:
            continue

        bid, ask, q_ts = quote_fields(q)
        q_age = quote_age_seconds(q_ts)
        if q_age is not None and q_age > QUOTE_MAX_AGE_SECONDS:
            logger.info("%s: skip (quote too old %.0fs)", sym, q_age)
            continue

        new_px = compute_new_limit_price(o, bid, ask)
        if new_px is None:
            continue

        old_px = getattr(o, "limit_price", None)
        try:
            old_f = float(old_px) if old_px is not None else None
        except Exception:
            old_f = None

        if old_f is not None and floats_close(old_f, new_px):
            continue

        logger.info("%s: reprice id=%s old=%s -> new=%.2f bid=%.2f ask=%.2f", sym, oid, old_px, new_px, bid, ask)

        if dry_run:
            continue

        try:
            trading.replace_order_by_id(
                oid,
                ReplaceOrderRequest(limit_price=str(round(new_px, 2))),
            )
            repriced += 1
        except Exception as e:
            logger.error("%s: replace failed id=%s err=%s", sym, oid, e)

    if repriced == 0 and canceled == 0:
        logger.info("no open orders to reprice")

    return None


def main():
    logger.info(
        "reprice_stale | start | dry_run=%s run_once=%s stale=%ss cancel_after=%ss",
        DRY_RUN,
        RUN_ONCE,
        STALE_SECONDS,
        CANCEL_AFTER_SECONDS,
    )

    trading, data = make_clients()

    while True:
        sleep_s = LOOP_SLEEP_SECS
        try:
            override = run_once(trading, data, DRY_RUN)
            if override is not None:
                sleep_s = override
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("loop error: %s", e)

        if RUN_ONCE:
            break

        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
