import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

log = logging.getLogger("reprice_stale")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# === ENV KNOBS ===
STALE_AFTER_SECONDS = int(os.getenv("STALE_AFTER_SECONDS", "300"))   # 5 min
REPRICE_EVERY_SECONDS = int(os.getenv("REPRICE_EVERY_SECONDS", "90"))
MAX_REPRICES = int(os.getenv("MAX_REPRICES", "4"))
CANCEL_AFTER_REPRICES = os.getenv("CANCEL_AFTER_REPRICES", "1") == "1"

MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.12"))
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def _qt(x: float, places: int = 2) -> float:
    return round(float(x) + 1e-9, places)

def _age_seconds(ts) -> int:
    if ts is None:
        return 0
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - ts).total_seconds())

def _market_is_open(trading: TradingClient) -> bool:
    try:
        clk = trading.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False

def _latest_bid_ask_mid(data_client: StockHistoricalDataClient, symbol: str):
    try:
        q = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        qsym = q[symbol]
        bid = float(qsym.bid_price) if qsym.bid_price is not None else None
        ask = float(qsym.ask_price) if qsym.ask_price is not None else None
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        return bid, ask, mid
    except Exception:
        try:
            t = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
            last = float(t.price)
            return None, None, last
        except Exception:
            return None, None, None

def _guarded_limit(data_client: StockHistoricalDataClient, symbol: str, side: str, px_hint: Optional[float]):
    bid, ask, mid = _latest_bid_ask_mid(data_client, symbol)
    if bid is not None and ask is not None and mid is not None:
        spread_abs = max(0.0, ask - bid)
        spread_pct = (spread_abs / mid) * 100.0 if mid else 999.0
        if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
            raise RuntimeError(
                f"wide spread skip {symbol} bid={bid:.2f} ask={ask:.2f} mid={mid:.2f} "
                f"abs={spread_abs:.4f} pct={spread_pct:.3f}%"
            )
        slip = QUOTE_PRICE_SLIPPAGE
        if side.lower() == "buy":
            raw = max(ask, (px_hint if px_hint else ask))
            raw = min(raw + slip, ask + slip)
        else:
            raw = min(bid, (px_hint if px_hint else bid))
            raw = max(raw - slip, bid - slip)
        return _qt(raw, 2), bid, ask
    if px_hint is None:
        raise RuntimeError(f"no quotes and no px_hint for {symbol}")
    return _qt(px_hint, 2), None, None

def _is_parent_limit(o) -> bool:
    oc = getattr(o, "order_class", None)
    if oc not in ("bracket", "simple", None):
        return False
    if getattr(o, "status", "") not in ("new", "accepted", "pending_new"):
        return False
    if getattr(o, "type", "").lower() != "limit":
        return False
    try:
        filled_qty = float(getattr(o, "filled_qty", 0) or 0)
    except Exception:
        filled_qty = 0.0
    return filled_qty == 0.0

def _count_reprices(o) -> int:
    coid = (getattr(o, "client_order_id", "") or "")
    for i in range(7):
        if coid.endswith(f"-rp{i}"):
            return i
    return 0

def _next_coid(o) -> str:
    base = (getattr(o, "client_order_id", "") or "").split("-rp")[0]
    return f"{base}-rp{_count_reprices(o)+1}"

def run_once(trading: TradingClient, data_client: StockHistoricalDataClient):
    if not _market_is_open(trading):
        log.info("market closed -> skipping reprice loop")
        return

    open_orders = trading.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
        )
    )

    for o in open_orders:
        if not _is_parent_limit(o):
            continue

        age = _age_seconds(getattr(o, "submitted_at", None))
        if age < STALE_AFTER_SECONDS:
            continue

        rcount = _count_reprices(o)
        if rcount >= MAX_REPRICES:
            if CANCEL_AFTER_REPRICES:
                log.info("%s: max reprices reached -> cancel", o.symbol)
                if not DRY_RUN:
                    try:
                        trading.cancel_order_by_id(o.id)
                    except APIError as e:
                        log.warning("%s: cancel failed: %s", o.symbol, e)
            else:
                log.info("%s: max reprices reached -> leaving as is", o.symbol)
            continue

        try:
            new_px, bid, ask = _guarded_limit(data_client, o.symbol, o.side.lower(), float(o.limit_price))
        except Exception as e:
            log.info("%s: skip reprice (%s)", o.symbol, e)
            continue

        new_coid = _next_coid(o)
        log.info("%s: reprice #%d | %.2f -> %.2f  (bid=%s ask=%s)  coid=%s",
                 o.symbol, rcount+1, float(o.limit_price), new_px,
                 f"{bid:.2f}" if bid is not None else "n/a",
                 f"{ask:.2f}" if ask is not None else "n/a",
                 new_coid)

        if DRY_RUN:
            continue

        try:
            trading.replace_order_by_id(o.id, limit_price=new_px, client_order_id=new_coid)
        except APIError as e:
            log.warning("%s: replace failed: %s", o.symbol, e)

def main():
    trading = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    last = 0.0
    while True:
        now = time.time()
        if now - last >= REPRICE_EVERY_SECONDS:
            last = now
            try:
                run_once(trading, data_client)
            except Exception as e:
                log.exception("reprice loop error: %s", e)
        time.sleep(1)

if __name__ == "__main__":
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        log.warning("ALPACA_API_KEY/SECRET not set; this script will fail to connect.")
    main()
