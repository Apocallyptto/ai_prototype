import os
import time
import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

LOG = logging.getLogger("alpaca_exit_guard")
logging.basicConfig(level=logging.INFO)


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    paper = None

    if paper is None:
        # Resolve from TRADING_MODE first (live|paper), then allow ALPACA_PAPER override.
        mode = (os.getenv("TRADING_MODE") or "paper").strip().lower()
        if mode not in ("live", "paper"):
            mode = "paper"
        paper = (mode != "live")

        if os.getenv("ALPACA_PAPER") is not None:
            v = (os.getenv("ALPACA_PAPER") or "").strip().lower()
            paper = v not in ("0", "false", "no", "off")

    return TradingClient(key, secret, paper=paper)


def _get_open_orders(tc: TradingClient, symbol: Optional[str] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True)
    orders = tc.get_orders(filter=req) or []
    if symbol:
        symbol = symbol.upper()
        orders = [o for o in orders if (getattr(o, "symbol", "") or "").upper() == symbol]
    return orders


def main():
    poll = int(os.getenv("EXIT_GUARD_POLL_SECONDS", "15"))
    symbol = (os.getenv("SYMBOL") or "").strip().upper() or None

    tc = get_trading_client()
    LOG.info("alpaca_exit_guard starting | poll=%ss | symbol=%s", poll, symbol or "*")

    while True:
        try:
            orders = _get_open_orders(tc, symbol=symbol)
            LOG.info("open_orders=%s", len(orders))
        except Exception as e:
            LOG.exception("exit_guard_error: %r", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()
