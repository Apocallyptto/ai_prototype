import os
import time
import logging
from collections import defaultdict
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, OrderClass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


logger = logging.getLogger("reprice_stale")


# --- Tunable params (env overrides our defaults) ---

# Max allowed spread (% of mid) – if wider, we skip repricing to avoid bad fills.
MAX_SPREAD_PCT = float(os.getenv("REPRICE_MAX_SPREAD_PCT", "0.40"))

# Min absolute move needed (in $) to bother repricing.
MIN_ABS_MOVE = float(os.getenv("REPRICE_MIN_ABS_MOVE", "0.01"))

# How often we loop when run as a daemon (cron still calls it once per cycle).
LOOP_SLEEP_SECS = int(os.getenv("REPRICE_LOOP_SLEEP_SECS", "1"))

# Safety cap: how many times we reprice the *same* order id in this process.
MAX_REPRICES_PER_ORDER = int(os.getenv("REPRICE_MAX_REPRICES_PER_ORDER", "20"))


def make_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("ALPACA_PAPER", "1") != "0"

    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in env.")

    trading = TradingClient(api_key, api_secret, paper=paper)
    data_client = StockHistoricalDataClient(api_key, api_secret)
    return trading, data_client


def get_open_limit_orders(trading: TradingClient):
    """
    Fetch open parent limit orders (we don't touch children legs directly).
    Uses the *new* alpaca-py style: GetOrdersRequest + filter=...
    """
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,  # include legs; we'll skip children manually
    )
    orders = trading.get_orders(filter=req)

    parents = []
    for o in orders:
        # Only limit / stop-limit orders.
        if o.order_type not in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            continue

        # Skip bracket / other complex classes for now
        # – we expect parents to be SIMPLE or BRACKET parents.
        # If your setup marks parents as BRACKET, we still keep them;
        # we just don't want to touch children.
        if o.order_class not in (OrderClass.SIMPLE, OrderClass.BRACKET):
            continue

        # If this is a child of a bracket, Alpaca usually gives parent ids;
        # you can extend this filter later if you want.
        parents.append(o)

    return parents


def get_quote(data_client: StockHistoricalDataClient, symbol: str):
    """
    Get latest NBBO quote for symbol.
    """
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


def compute_new_price(order, bid: float, ask: float) -> float | None:
    """
    Decide a new limit price based on side and current quote.
    We keep your existing "hug the inside" behaviour:
      - BUY → just above bid
      - SELL → just below ask
    But we still respect min move and spread caps.
    """
    mid = (bid + ask) / 2.0
    spread_abs = ask - bid
    spread_pct = (spread_abs / mid) * 100.0

    if spread_pct > MAX_SPREAD_PCT:
        logger.info(
            "AAPL: skip reprice (wide spread skip %s bid=%.2f ask=%.2f mid=%.2f abs=%.4f pct=%.3f%%)",
            order.symbol,
            bid,
            ask,
            mid,
            spread_abs,
            spread_pct,
        )
        return None

    old_px = float(order.limit_price) if order.limit_price is not None else None
    if old_px is None:
        return None

    # For US stocks we typically tick at 0.01, so we round to 2 decimals.
    tick = Decimal("0.01")

    if order.side == OrderSide.BUY:
        # Place near the bid, but slightly inside to stay competitive
        target = Decimal(str(bid)) + tick
        if float(target) >= old_px:
            # We're already better than or equal to bid; no point moving.
            return None
    else:
        # SELL – place near ask but slightly inside
        target = Decimal(str(ask)) - tick
        if float(target) <= old_px:
            return None

    new_px = float(target)

    # Require a minimum move to avoid useless replace spam
    if abs(new_px - old_px) < MIN_ABS_MOVE:
        return None

    return round(new_px, 2)


def run_once(trading: TradingClient, data_client: StockHistoricalDataClient,
             reprice_counts: dict[str, int], dry_run: bool = False):
    orders = get_open_limit_orders(trading)

    if not orders:
        logger.info("no open orders to reprice")
        return

    for o in orders:
        # Cap per-run/instance reprice count for safety
        count = reprice_counts[o.id]
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
        reprice_counts[o.id] = count

        # Build new client_order_id so we can track the lineage
        base_coid = (o.client_order_id or "noprefix").split("-rp")[0]
        new_coid = f"{base_coid}-rp{count}"

        logger.info(
            "%s: reprice #%d | %.2f -> %.2f  (bid=%.2f ask=%.2f)  coid=%s",
            o.symbol,
            count,
            float(o.limit_price),
            new_px,
            bid,
            ask,
            new_coid,
        )

        if dry_run:
            continue

        try:
            # ✅ Correct alpaca-py usage:
            #   ReplaceOrderRequest + order_data=...
            replace_req = ReplaceOrderRequest(
                limit_price=new_px,
                client_order_id=new_coid,
            )
            trading.replace_order_by_id(
                order_id=o.id,
                order_data=replace_req,
            )
        except Exception as e:
            logger.error("error replacing order %s: %s", o.id, e)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    dry_run = os.getenv("DRY_RUN", "0") == "1"
    trading, data_client = make_clients()
    reprice_counts: dict[str, int] = defaultdict(int)

    logger.info("reprice_stale | starting loop | dry_run=%s", dry_run)

    while True:
        try:
            run_once(trading, data_client, reprice_counts, dry_run=dry_run)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("reprice loop error: %s", e)
        time.sleep(LOOP_SLEEP_SECS)


if __name__ == "__main__":
    main()
