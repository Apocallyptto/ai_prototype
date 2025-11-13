import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Iterable

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    QueryOrderStatus,
    OrderSide,
    OrderType,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    ReplaceOrderRequest,
)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockLatestQuoteRequest

logger = logging.getLogger("reprice_stale")
logging.basicConfig(level=logging.INFO)

# --- Config from env ---------------------------------------------------------

STALE_MINUTES = int(os.getenv("REPRICE_STALE_MINUTES", "15"))
MAX_REPRICES = int(os.getenv("REPRICE_STALE_MAX_REPRICES", "3"))

# Maximum bid/ask spread, in percent of mid, to allow repricing
MAX_SPREAD_PCT = float(os.getenv("REPRICE_STALE_MAX_SPREAD_PCT", "0.6"))  # e.g. 0.6 => 0.6%

# How aggressively to hug the inside market.
# For BUY we place just below bid, for SELL just above ask.
TICK_SIZE = float(os.getenv("REPRICE_STALE_TICK", "0.01"))

# If set (any non-0), we only log what we *would* do, no actual replace.
DRY_RUN_ENV = os.getenv("DRY_RUN", "0").lower()
DRY_RUN_DEFAULT = DRY_RUN_ENV not in ("0", "false", "")


# --- Helpers -----------------------------------------------------------------


def parse_reprice_count(client_order_id: Optional[str]) -> int:
    """
    Extract -rpN suffix from client_order_id, e.g. 'abc-rp2' -> 2.
    If not present, return 0.
    """
    if not client_order_id:
        return 0
    if "-rp" not in client_order_id:
        return 0
    base, suffix = client_order_id.rsplit("-rp", 1)
    try:
        return int(suffix)
    except ValueError:
        return 0


def base_coid(client_order_id: Optional[str]) -> str:
    """
    Remove any -rpN suffix, leaving the base.
    """
    if not client_order_id:
        return ""
    if "-rp" not in client_order_id:
        return client_order_id
    base, _ = client_order_id.rsplit("-rp", 1)
    return base


def make_reprice_coid(client_order_id: Optional[str], next_count: int) -> str:
    """
    Build new client_order_id with -rpN suffix.
    """
    return f"{base_coid(client_order_id) or 'reprice'}-rp{next_count}"


def get_env_symbols() -> Optional[Iterable[str]]:
    """
    Returns list of symbols from env SYMBOLS, or None if not set.
    """
    symbols = os.getenv("SYMBOLS")
    if not symbols:
        return None
    return [s.strip().upper() for s in symbols.split(",") if s.strip()]


def age_minutes(dt: datetime) -> float:
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 60.0


# --- Alpaca clients ----------------------------------------------------------


def make_clients():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("ALPACA_PAPER", "1") != "0"

    trading = TradingClient(key, secret, paper=paper)
    data_client = StockHistoricalDataClient(key, secret)
    return trading, data_client


def get_latest_quotes(
    data_client: StockHistoricalDataClient, symbols: Iterable[str]
) -> Dict[str, object]:
    """
    Fetch latest NBBO quotes for given symbols.
    Returns dict: symbol -> Quote object.
    """
    symbols = list(set([s.upper() for s in symbols]))
    if not symbols:
        return {}

    req = StockLatestQuoteRequest(
        symbol_or_symbols=symbols,
        feed=DataFeed.SIP,
    )
    quotes = data_client.get_stock_latest_quote(req)
    # alpaca-py returns either dict or single model depending on len(symbols)
    if isinstance(quotes, dict):
        return quotes
    # Single symbol case
    return {symbols[0]: quotes}


# --- Core reprice logic ------------------------------------------------------


def compute_new_limit(
    side: OrderSide,
    old_limit: float,
    bid: float,
    ask: float,
) -> float:
    """
    Decide new limit price based on side and current inside market.
    We want to get closer to the market but not cross.
    """
    mid = (bid + ask) / 2.0

    if side == OrderSide.BUY:
        # Place just below bid, but never above old_limit
        target = max(bid - TICK_SIZE, 0.01)
        new_limit = min(old_limit, target)
    else:
        # SELL: place just above ask, but never below old_limit
        target = ask + TICK_SIZE
        new_limit = max(old_limit, target)

    # Round to cents
    return round(new_limit, 2)


def replace_order(
    trading: TradingClient,
    order,
    new_limit: float,
    new_client_order_id: str,
    dry_run: bool = False,
):
    """
    Small wrapper so you can import & inspect it:
        from jobs.reprice_stale import replace_order
    Uses the *new* Alpaca-py style: order_data=ReplaceOrderRequest(...).
    """
    logger.info(
        "replace_order | id=%s symbol=%s side=%s old=%.2f new=%.2f coid=%s dry_run=%s",
        order.id,
        order.symbol,
        order.side,
        float(order.limit_price or 0.0),
        new_limit,
        new_client_order_id,
        dry_run,
    )

    if dry_run:
        return

    req = ReplaceOrderRequest(
        limit_price=new_limit,
        client_order_id=new_client_order_id,
    )

    # NOTE: new Alpaca-py signature: order_data=<ReplaceOrderRequest>
    trading.replace_order_by_id(order.id, order_data=req)


def run_once(trading: TradingClient, data_client: StockHistoricalDataClient):
    watched_symbols = get_env_symbols()
    dry_run = DRY_RUN_DEFAULT

    # Get all open orders (SDK v2+ style: filter=GetOrdersRequest(...))
    filter_req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=False,
        limit=50,
    )
    orders = list(trading.get_orders(filter_req))

    limit_orders = [
        o
        for o in orders
        if o.type == OrderType.LIMIT
        and o.side in (OrderSide.BUY, OrderSide.SELL)
        and o.submitted_at is not None
        and (watched_symbols is None or o.symbol.upper() in watched_symbols)
    ]

    if not limit_orders:
        logger.info("reprice_stale:no open limit orders to reprice")
        return

    # Fetch quotes for involved symbols
    quotes = get_latest_quotes(
        data_client,
        [o.symbol for o in limit_orders],
    )

    for o in limit_orders:
        submitted_at: datetime = o.submitted_at
        mins_old = age_minutes(submitted_at)

        if mins_old < STALE_MINUTES:
            continue

        q = quotes.get(o.symbol)
        if not q:
            continue

        bid = float(q.bid_price or 0.0)
        ask = float(q.ask_price or 0.0)
        if bid <= 0 or ask <= 0 or ask <= bid:
            continue

        mid = (bid + ask) / 2.0
        spread_abs = ask - bid
        spread_pct = (spread_abs / mid) * 100.0

        if spread_pct > MAX_SPREAD_PCT:
            logger.info(
                "reprice_stale:%s: skip reprice (wide spread skip %s bid=%.2f ask=%.2f mid=%.2f abs=%.4f pct=%.3f%%)",
                o.symbol,
                o.symbol,
                bid,
                ask,
                mid,
                spread_abs,
                spread_pct,
            )
            continue

        old_limit = float(o.limit_price)
        new_limit = compute_new_limit(o.side, old_limit, bid, ask)

        if abs(new_limit - old_limit) < 0.0001:
            # no meaningful change
            continue

        rp_count = parse_reprice_count(o.client_order_id)
        if rp_count >= MAX_REPRICES:
            logger.info(
                "reprice_stale:%s: reached max reprices (%s >= %s), skip for order id=%s",
                o.symbol,
                rp_count,
                MAX_REPRICES,
                o.id,
            )
            continue

        new_coid = make_reprice_coid(o.client_order_id, rp_count + 1)

        logger.info(
            "reprice_stale:%s: reprice #%d | %.2f -> %.2f  (bid=%.2f ask=%.2f)  coid=%s",
            o.symbol,
            rp_count + 1,
            old_limit,
            new_limit,
            bid,
            ask,
            new_coid,
        )

        try:
            replace_order(trading, o, new_limit, new_coid, dry_run=dry_run)
        except Exception as e:
            logger.error("reprice_stale: error replacing order %s: %s", o.id, e)


def main():
    trading, data_client = make_clients()

    # In cron mode we want this to loop forever with a short sleep,
    # because it's launched from cron_v2 every cycle.
    while True:
        try:
            run_once(trading, data_client)
        except Exception as e:
            logger.exception("reprice loop error: %s", e)
        time.sleep(1)


if __name__ == "__main__":
    main()
