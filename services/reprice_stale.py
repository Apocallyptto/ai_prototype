# services/reprice_stale.py
from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


logger = logging.getLogger("reprice_stale")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    return float(v)


def _tick_size(price: float) -> float:
    # Reasonable default: sub-$1 stocks can require finer ticks.
    return 0.0001 if price < 1.0 else 0.01


def _round_to_tick(price: float) -> float:
    if price <= 0:
        return price
    tick = _tick_size(price)
    return math.floor(price / tick + 0.5) * tick


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_order_ts(o) -> Optional[datetime]:
    # Alpaca order objects usually have submitted_at/created_at
    ts = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
    if ts is None:
        return None
    # Ensure timezone-aware
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _is_limit_order(o) -> bool:
    # robust across alpaca-py versions
    typ = getattr(o, "type", None) or getattr(o, "order_type", None) or ""
    return str(typ).upper().endswith("LIMIT")


def _order_class(o) -> str:
    oc = getattr(o, "order_class", "") or ""
    return str(oc).lower()


def _coid(o) -> str:
    return str(getattr(o, "client_order_id", "") or "")


def _should_skip_by_coid(coid: str) -> bool:
    # hard skip exit/oco ids
    coid_l = coid.lower()
    return (
        coid_l.startswith("exit-")
        or coid_l.startswith("oco-")
        or "-exit-" in coid_l
        or "-tp-" in coid_l
        or "-sl-" in coid_l
    )


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def _get_quote(data: StockHistoricalDataClient, symbol: str) -> Optional[Quote]:
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        resp = data.get_stock_latest_quote(req)
        q = resp.get(symbol)
        if not q:
            return None
        bid = float(getattr(q, "bid_price", 0) or 0)
        ask = float(getattr(q, "ask_price", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        return Quote(bid=bid, ask=ask)
    except Exception as e:
        logger.warning("%s: quote fetch failed: %s", symbol, e)
        return None


def _spread_pct(q: Quote) -> float:
    mid = (q.bid + q.ask) / 2.0
    if mid <= 0:
        return 999.0
    return (q.ask - q.bid) / mid * 100.0


def _compute_new_price(o, q: Quote) -> float:
    # Use mid, rounded to tick, clamped inside bid/ask
    mid = (q.bid + q.ask) / 2.0
    new_px = _round_to_tick(mid)

    # keep inside market
    if new_px > q.ask:
        new_px = _round_to_tick(q.ask)
    if new_px < q.bid:
        new_px = _round_to_tick(q.bid)

    return float(new_px)


def _market_is_open(trading: TradingClient) -> bool:
    try:
        clk = trading.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        # If clock fails, be conservative: treat as closed
        return False


def _get_open_orders(trading: TradingClient):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    try:
        orders = trading.get_orders(req) or []
    except TypeError:
        # older signature
        orders = trading.get_orders(filter=req) or []
    return orders


def _filter_reprice_candidates(orders):
    out = []
    for o in orders:
        if not _is_limit_order(o):
            continue
        if _order_class(o) != "simple":
            # do NOT touch bracket/oco exits
            continue
        coid = _coid(o)
        if coid and _should_skip_by_coid(coid):
            continue
        out.append(o)
    return out


def run_once(
    trading: TradingClient,
    data: StockHistoricalDataClient,
    stale_seconds: int,
    cancel_after_seconds: int,
    max_spread_pct: float,
    dry_run: bool,
    reprice_counts: dict[str, int],
    max_reprices_per_order: int,
):
    orders = _filter_reprice_candidates(_get_open_orders(trading))

    if not orders:
        logger.info("no open orders to reprice")
        return

    now = _now_utc()

    for o in orders:
        oid = str(getattr(o, "id", "") or "")
        sym = str(getattr(o, "symbol", "") or "")
        old_px = float(getattr(o, "limit_price", 0) or 0)
        ts = _get_order_ts(o)
        if not oid or not sym or old_px <= 0 or ts is None:
            continue

        age = (now - ts).total_seconds()

        # Safety cap per process lifetime
        if reprice_counts[oid] >= max_reprices_per_order:
            continue

        # Cancel first if too old
        if cancel_after_seconds > 0 and age >= cancel_after_seconds:
            if dry_run:
                logger.info("%s: would_cancel age=%.0fs id=%s limit=%.2f", sym, age, oid, old_px)
            else:
                try:
                    trading.cancel_order_by_id(oid)
                    logger.info("%s: canceled age=%.0fs id=%s limit=%.2f", sym, age, oid, old_px)
                except Exception as e:
                    logger.warning("%s: cancel failed id=%s: %s", sym, oid, e)
            continue

        # Only reprice when stale
        if age < stale_seconds:
            continue

        q = _get_quote(data, sym)
        if not q:
            continue

        sp = _spread_pct(q)
        if sp > max_spread_pct:
            logger.info("%s: skip (wide spread %.3f%% bid=%.2f ask=%.2f)", sym, sp, q.bid, q.ask)
            continue

        new_px = _compute_new_price(o, q)
        if math.isclose(new_px, old_px, rel_tol=0.0, abs_tol=_tick_size(old_px)):
            continue

        if dry_run:
            logger.info(
                "%s: would_replace age=%.0fs id=%s old=%.2f new=%.2f bid=%.2f ask=%.2f",
                sym, age, oid, old_px, new_px, q.bid, q.ask
            )
            continue

        try:
            req = ReplaceOrderRequest(limit_price=new_px)
            trading.replace_order_by_id(oid, req)
            reprice_counts[oid] += 1
            logger.info(
                "%s: replaced age=%.0fs id=%s old=%.2f new=%.2f bid=%.2f ask=%.2f",
                sym, age, oid, old_px, new_px, q.bid, q.ask
            )
        except Exception as e:
            logger.warning("%s: replace failed id=%s: %s", sym, oid, e)


def main():
    # minimal logging (your container/other modules can override format)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    dry_run = _env_bool("DRY_RUN", False)
    run_once_flag = _env_bool("RUN_ONCE", False)

    stale_seconds = _env_int("REPRICE_STALE_SECONDS", 45)
    cancel_after_seconds = _env_int("REPRICE_CANCEL_AFTER_SECONDS", 300)
    require_market_open = _env_bool("REPRICE_REQUIRE_MARKET_OPEN", True)
    max_spread_pct = _env_float("REPRICE_MAX_SPREAD_PCT", 0.5)

    loop_sleep = _env_int("REPRICE_LOOP_SLEEP_SECS", 60)
    max_reprices = _env_int("REPRICE_MAX_REPRICES_PER_ORDER", 10)

    logger.info(
        "reprice_stale | start | dry_run=%s run_once=%s stale=%ss cancel_after=%ss",
        dry_run, run_once_flag, stale_seconds, cancel_after_seconds
    )

    trading = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
    data = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))

    reprice_counts: dict[str, int] = defaultdict(int)

    while True:
        try:
            if require_market_open and not _market_is_open(trading):
                logger.info("market closed -> idle (no orders) -> sleeping %ss", loop_sleep if not run_once_flag else 0)
                if run_once_flag:
                    return
                time.sleep(loop_sleep)
                continue

            run_once(
                trading=trading,
                data=data,
                stale_seconds=stale_seconds,
                cancel_after_seconds=cancel_after_seconds,
                max_spread_pct=max_spread_pct,
                dry_run=dry_run,
                reprice_counts=reprice_counts,
                max_reprices_per_order=max_reprices,
            )

            if run_once_flag:
                return

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception("reprice loop error: %s", e)

        time.sleep(loop_sleep)


if __name__ == "__main__":
    main()
