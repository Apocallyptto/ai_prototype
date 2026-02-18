import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, CancelOrderResponse
from alpaca.trading.enums import QueryOrderStatus, OrderSide
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("reprice_stale")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _resolve_mode() -> str:
    return (os.getenv("TRADING_MODE") or "paper").strip().lower()


def _resolve_paper() -> bool:
    # Explicit override wins (useful for emergency forcing paper)
    if os.getenv("ALPACA_PAPER") is not None:
        return _env_bool("ALPACA_PAPER", True)
    return _resolve_mode() != "live"


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # alpaca returns ISO; python can parse with fromisoformat if it has offset
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_open_order(o) -> bool:
    st = (getattr(o, "status", "") or "").lower()
    return st in ("new", "accepted", "partially_filled", "held", "pending_new", "pending_cancel")


def _quote_mid(data: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        q = data.get_stock_latest_quote(req)
        obj = q.get(symbol)
        if not obj:
            return None
        bid = getattr(obj, "bid_price", None)
        ask = getattr(obj, "ask_price", None)
        if bid is None or ask is None:
            return None
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0
    except Exception as e:
        logger.warning("quote_mid error for %s: %s", symbol, e)
        return None


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b) * 100.0


def _should_skip_afterhours(allow_ah: bool) -> bool:
    return not allow_ah


def _market_is_open(tc: TradingClient) -> bool:
    clk = tc.get_clock()
    return bool(getattr(clk, "is_open", False))


def _filter_reprice_candidates(orders) -> list:
    out = []
    for o in orders:
        if not _is_open_order(o):
            continue
        # only limit orders we can reprice
        if (getattr(o, "type", "") or "").lower() != "limit":
            continue
        out.append(o)
    return out


def _cancel_if_too_old(tc: TradingClient, o, cancel_after_s: int) -> bool:
    if cancel_after_s <= 0:
        return False
    submitted_at = _parse_ts(getattr(o, "submitted_at", None))
    if not submitted_at:
        return False
    age_s = (_utcnow() - submitted_at).total_seconds()
    if age_s < cancel_after_s:
        return False
    try:
        tc.cancel_order_by_id(getattr(o, "id"))
        return True
    except Exception as e:
        logger.warning("cancel_order_by_id failed id=%s sym=%s: %s", getattr(o, "id"), getattr(o, "symbol"), e)
        return False


def _reprice_one(tc: TradingClient, data: StockHistoricalDataClient, o, max_spread_pct: float) -> bool:
    sym = (getattr(o, "symbol", "") or "").upper()
    side = (getattr(o, "side", "") or "").lower()
    limit_price = getattr(o, "limit_price", None)
    if limit_price is None:
        return False

    mid = _quote_mid(data, sym)
    if mid is None:
        return False

    # avoid repricing when spread too wide (mid far from current limit)
    lp = float(limit_price)
    spread = abs(lp - mid)
    sp_pct = _pct(spread, mid)
    if max_spread_pct > 0 and sp_pct > max_spread_pct:
        return False

    # nudge toward mid depending on side
    new_price = mid
    try:
        tc.replace_order_by_id(getattr(o, "id"), limit_price=str(round(new_price, 2)))
        return True
    except Exception as e:
        logger.warning("replace_order_by_id failed id=%s sym=%s: %s", getattr(o, "id"), sym, e)
        return False


def _get_open_orders(tc: TradingClient, symbols: Optional[list[str]] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True, symbols=symbols)
    return tc.get_orders(req) or []


def run_once(
    trading: TradingClient,
    data: StockHistoricalDataClient,
    symbols: list[str],
    stale_seconds: int,
    cancel_after_seconds: int,
    max_spread_pct: float,
    require_market_open: bool,
    allow_ah: bool,
    counts: dict[str, int],
):
    if require_market_open and not _market_is_open(trading):
        return

    if _should_skip_afterhours(allow_ah) and not _market_is_open(trading):
        return

    orders = _filter_reprice_candidates(_get_open_orders(trading, symbols=symbols))

    now = _utcnow()
    for o in orders:
        sym = (getattr(o, "symbol", "") or "").upper()
        submitted_at = _parse_ts(getattr(o, "submitted_at", None))
        if not submitted_at:
            continue

        age_s = (now - submitted_at).total_seconds()
        if stale_seconds > 0 and age_s < stale_seconds:
            continue

        if _cancel_if_too_old(trading, o, cancel_after_seconds):
            counts[f"{sym}:canceled_old"] += 1
            continue

        if _reprice_one(trading, data, o, max_spread_pct=max_spread_pct):
            counts[f"{sym}:repriced"] += 1


def main():
    poll_seconds = _env_int("POLL_SECONDS", 10)
    stale_seconds = _env_int("STALE_SECONDS", 60)
    cancel_after_seconds = _env_int("CANCEL_AFTER_SECONDS", 0)
    max_spread_pct = _env_float("MAX_SPREAD_PCT", 0.25)
    require_market_open = _env_bool("REQUIRE_MARKET_OPEN", True)
    allow_ah = _env_bool("ALLOW_AH", False)

    symbols = [s.strip().upper() for s in (os.getenv("SYMBOLS", "") or "").split(",") if s.strip()]
    if not symbols:
        symbols = ["AAPL", "MSFT", "SPY"]

    mode = _resolve_mode()
    paper = _resolve_paper()

    logger.info("reprice_stale | trading | mode=%s paper=%s", mode, paper)

    trading = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=paper)
    data = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))

    counts: dict[str, int] = defaultdict(int)

    while True:
        try:
            run_once(
                trading=trading,
                data=data,
                symbols=symbols,
                stale_seconds=stale_seconds,
                cancel_after_seconds=cancel_after_seconds,
                max_spread_pct=max_spread_pct,
                require_market_open=require_market_open,
                allow_ah=allow_ah,
                counts=counts,
            )
        except Exception as e:
            logger.exception("reprice loop error: %s", e)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
