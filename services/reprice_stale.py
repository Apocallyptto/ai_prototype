import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest

logger = logging.getLogger("reprice_stale")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in ("1", "true", "True", "yes", "YES", "on", "ON")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class RepriceConfig:
    # behavior
    dry_run: bool
    run_once: bool

    # timing
    loop_sleep_seconds: int
    stale_seconds: int  # >=0 enables repricing, <0 disables repricing
    cancel_after_seconds: int  # >=0 enables cancel, <0 disables cancel

    # quote guards
    max_spread_pct: float
    max_quote_age_seconds: int

    # replace guards
    min_abs_move: float
    max_reprices_per_order: int

    # market clock gate
    require_market_open: bool
    closed_market_sleep_seconds: int

    # log noise control
    idle_log_every_seconds: int

    @staticmethod
    def from_env() -> "RepriceConfig":
        return RepriceConfig(
            dry_run=_env_bool("DRY_RUN", "0"),
            run_once=_env_bool("RUN_ONCE", "0"),
            loop_sleep_seconds=_env_int("REPRICE_LOOP_SLEEP_SECONDS", 2),
            stale_seconds=_env_int("REPRICE_STALE_SECONDS", 45),
            cancel_after_seconds=_env_int("REPRICE_CANCEL_AFTER_SECONDS", 300),
            max_spread_pct=_env_float("REPRICE_MAX_SPREAD_PCT", 2.0),
            max_quote_age_seconds=_env_int("REPRICE_MAX_QUOTE_AGE_SECONDS", 30),
            min_abs_move=_env_float("REPRICE_MIN_ABS_MOVE", 0.02),
            max_reprices_per_order=_env_int("REPRICE_MAX_REPRICES_PER_ORDER", 5),
            require_market_open=_env_bool("REPRICE_REQUIRE_MARKET_OPEN", "1"),
            closed_market_sleep_seconds=_env_int("REPRICE_CLOSED_MARKET_SLEEP_SECONDS", 60),
            idle_log_every_seconds=_env_int("REPRICE_IDLE_LOG_EVERY_SECONDS", 60),
        )


def make_clients() -> Tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("ALPACA_API_KEY") or ""
    secret = os.getenv("ALPACA_API_SECRET") or ""
    paper = _env_bool("ALPACA_PAPER", "1")  # optional; default paper mode for safety
    trading = TradingClient(key, secret, paper=paper)
    data_client = StockHistoricalDataClient(key, secret)
    return trading, data_client


def is_market_open(trading: TradingClient) -> bool:
    """
    Fail-open: if the clock call fails, behave as if market is open (keeps prior behavior).
    """
    try:
        c = trading.get_clock()
        return bool(getattr(c, "is_open", False))
    except Exception as e:
        logger.warning("clock check failed -> fail-open (treat as open): %s", e)
        return True


def _order_ts(o: Any) -> Optional[datetime]:
    # prefer submitted_at; otherwise created_at; otherwise updated_at
    for attr in ("submitted_at", "created_at", "updated_at"):
        ts = getattr(o, attr, None)
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def _order_age_seconds(o: Any, now: datetime) -> Optional[int]:
    ts = _order_ts(o)
    if not ts:
        return None
    try:
        return int((now - ts).total_seconds())
    except Exception:
        return None


def _order_class_str(o: Any) -> str:
    oc = getattr(o, "order_class", None)
    return (str(oc) if oc is not None else "").lower()


def _is_simple_limit_order(o: Any) -> bool:
    # Only touch non-OCO simple LIMIT orders (entry orders).
    if getattr(o, "type", None) != OrderType.LIMIT and getattr(o, "order_type", None) != OrderType.LIMIT:
        return False

    oc_s = _order_class_str(o)
    if "oco" in oc_s or "bracket" in oc_s or "oto" in oc_s:
        return False

    # Some SDK versions return None for simple (oc_s == "")
    if oc_s and "simple" not in oc_s:
        return False

    # skip anything with legs (bracket/oco)
    legs = getattr(o, "legs", None)
    if legs:
        return False

    filled_qty = getattr(o, "filled_qty", None)
    if filled_qty not in (None, "0", 0, "0.0"):
        return False

    return True


def get_open_orders(trading: TradingClient) -> List[Any]:
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    return trading.get_orders(req) or []


def get_latest_quote(
    data_client: StockHistoricalDataClient, symbol: str, now: datetime
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Returns (bid, ask, quote_age_seconds). Any item may be None if missing.
    """
    try:
        resp = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[symbol]))
        q = None
        if isinstance(resp, dict):
            q = resp.get(symbol)
        else:
            q = getattr(resp, symbol, None)

        if q is None:
            return None, None, None

        bid = _as_float(getattr(q, "bid_price", None))
        ask = _as_float(getattr(q, "ask_price", None))

        ts = getattr(q, "timestamp", None) or getattr(q, "t", None)
        quote_ts = None
        if isinstance(ts, datetime):
            quote_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

        age = int((now - quote_ts).total_seconds()) if quote_ts else None
        return bid, ask, age
    except Exception as e:
        logger.warning("%s: quote fetch failed: %s", symbol, e)
        return None, None, None


def _is_buy(side: Any) -> bool:
    return str(side).lower().endswith("buy")


def compute_new_limit_price(side: Any, bid: float, ask: float) -> float:
    # price inside the spread (slightly aggressive)
    mid = (bid + ask) / 2.0
    if _is_buy(side):
        px = min(ask, mid + 0.01)
    else:
        px = max(bid, mid - 0.01)
    return round(px, 2)


def _spread_pct(bid: float, ask: float) -> Optional[float]:
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return ((ask - bid) / mid) * 100.0


def run_iteration(
    trading: TradingClient,
    data_client: StockHistoricalDataClient,
    reprice_counts: Dict[str, int],
    cfg: RepriceConfig,
    last_idle_log_at: List[float],
) -> int:
    """
    Runs one iteration. Returns recommended sleep seconds for the next loop.
    """
    now = _utcnow()
    orders = get_open_orders(trading)
    candidates = [o for o in orders if _is_simple_limit_order(o)]

    canceled = 0
    to_reprice: List[Tuple[Any, int]] = []

    # 1) cancel pass (works even when market is closed)
    for o in candidates:
        age = _order_age_seconds(o, now)
        if age is None:
            continue

        if cfg.cancel_after_seconds >= 0 and age >= cfg.cancel_after_seconds:
            oid = str(getattr(o, "id", ""))
            sym = getattr(o, "symbol", "?")
            if cfg.dry_run:
                logger.info("%s: would_cancel (age=%ss >= %ss) id=%s", sym, age, cfg.cancel_after_seconds, oid)
            else:
                try:
                    trading.cancel_order_by_id(oid)
                    logger.info("%s: cancel (age=%ss >= %ss) id=%s", sym, age, cfg.cancel_after_seconds, oid)
                except Exception as e:
                    logger.warning("%s: cancel_failed id=%s err=%s", sym, oid, e)
                    continue
            canceled += 1
            continue

        if cfg.stale_seconds >= 0 and age >= cfg.stale_seconds:
            to_reprice.append((o, age))

    open_orders_count = len(candidates)

    # 2) market-open gate before repricing
    if cfg.require_market_open and cfg.stale_seconds >= 0:
        market_open = is_market_open(trading)
        if not market_open:
            sleep_s = max(cfg.closed_market_sleep_seconds, cfg.loop_sleep_seconds)

            # reduce noise: print this idle message at most once per idle_log_every_seconds
            now_mono = time.monotonic()
            should_log = (now_mono - last_idle_log_at[0]) >= max(1, cfg.idle_log_every_seconds)
            if should_log or canceled > 0 or open_orders_count > 0:
                if open_orders_count == 0:
                    msg = "market closed -> idle (no orders)"
                else:
                    msg = f"market closed -> skipping reprices (open_orders={open_orders_count} canceled={canceled})"
                if cfg.run_once:
                    logger.info(msg)
                else:
                    logger.info("%s -> sleeping %ss", msg, sleep_s)
                last_idle_log_at[0] = now_mono

            return sleep_s

    # 3) repricing pass (only if enabled + market open or gate disabled)
    if cfg.stale_seconds < 0:
        # repricing disabled
        now_mono = time.monotonic()
        if (now_mono - last_idle_log_at[0]) >= max(1, cfg.idle_log_every_seconds):
            logger.info("repricing disabled (REPRICE_STALE_SECONDS < 0) | open_orders=%s canceled=%s", open_orders_count, canceled)
            last_idle_log_at[0] = now_mono
        return cfg.loop_sleep_seconds

    if not to_reprice:
        # Avoid spamming every loop
        now_mono = time.monotonic()
        if (now_mono - last_idle_log_at[0]) >= max(1, cfg.idle_log_every_seconds):
            logger.info("no open orders to reprice")
            last_idle_log_at[0] = now_mono
        return cfg.loop_sleep_seconds

    for o, age in to_reprice:
        oid = str(getattr(o, "id", ""))
        sym = getattr(o, "symbol", "?")
        side = getattr(o, "side", None)
        old_px = _as_float(getattr(o, "limit_price", None))

        if not oid or side is None or old_px is None:
            logger.info("%s: skip (missing fields)", sym)
            continue

        if reprice_counts.get(oid, 0) >= cfg.max_reprices_per_order:
            logger.info("%s: skip (max reprices reached=%s) id=%s", sym, cfg.max_reprices_per_order, oid)
            continue

        bid, ask, q_age = get_latest_quote(data_client, sym, now)
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            logger.info("%s: skip (no quote) id=%s", sym, oid)
            continue

        if q_age is not None and q_age > cfg.max_quote_age_seconds:
            logger.info("%s: skip (quote too old %ss) id=%s", sym, q_age, oid)
            continue

        sp = _spread_pct(bid, ask)
        if sp is not None and sp > cfg.max_spread_pct:
            logger.info("%s: skip (wide spread %.3f%% bid=%.2f ask=%.2f) id=%s", sym, sp, bid, ask, oid)
            continue

        new_px = compute_new_limit_price(side, bid, ask)

        if abs(new_px - old_px) < cfg.min_abs_move:
            logger.info("%s: skip (price change too small %.4f < %.4f) id=%s", sym, abs(new_px - old_px), cfg.min_abs_move, oid)
            continue

        if cfg.dry_run:
            logger.info("%s: would_replace age=%ss id=%s old=%.2f new=%.2f bid=%.2f ask=%.2f", sym, age, oid, old_px, new_px, bid, ask)
        else:
            try:
                trading.replace_order_by_id(oid, ReplaceOrderRequest(limit_price=str(new_px)))
                logger.info("%s: replaced age=%ss id=%s old=%.2f new=%.2f bid=%.2f ask=%.2f", sym, age, oid, old_px, new_px, bid, ask)
            except Exception as e:
                logger.warning("%s: replace_failed id=%s err=%s", sym, oid, e)
                continue

        reprice_counts[oid] = reprice_counts.get(oid, 0) + 1

    return cfg.loop_sleep_seconds


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = RepriceConfig.from_env()
    trading, data_client = make_clients()
    reprice_counts: Dict[str, int] = defaultdict(int)
    last_idle_log_at = [0.0]  # mutable holder for monotonic timestamp

    logger.info(
        "reprice_stale | start | dry_run=%s run_once=%s stale=%ss cancel_after=%ss",
        cfg.dry_run, cfg.run_once, cfg.stale_seconds, cfg.cancel_after_seconds
    )

    if cfg.run_once:
        run_iteration(trading, data_client, reprice_counts, cfg, last_idle_log_at)
        return

    while True:
        try:
            sleep_s = run_iteration(trading, data_client, reprice_counts, cfg, last_idle_log_at)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception("reprice loop error: %s", e)
            sleep_s = cfg.loop_sleep_seconds

        time.sleep(max(1, int(sleep_s)))


if __name__ == "__main__":
    main()
