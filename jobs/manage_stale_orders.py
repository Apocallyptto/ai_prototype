# jobs/manage_stale_orders.py
import os, logging
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manage_stale_orders")

ALPACA_KEY = os.getenv("ALPACA_API_KEY","")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET","")
MAX_MIN = int(os.getenv("STALE_CANCEL_MINUTES","45"))
APCA_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

def _client():
    return TradingClient(ALPACA_KEY, ALPACA_SEC, paper="paper" in APCA_BASE)

def _get_open_orders(cli: TradingClient):
    """
    Version-proof: prefer status='open' string; if SDK requires enums, fall back.
    """
    try:
        return list(cli.get_orders(status="open"))
    except Exception as e1:
        log.warning("get_orders(status='open') failed: %s | trying fallback", e1)
        try:
            from alpaca.trading.enums import OrderStatus  # optional
            return list(cli.get_orders(status=OrderStatus.OPEN))
        except Exception as e2:
            log.warning("fallback get_orders(OrderStatus.OPEN) failed: %s", e2)
            return []

def main():
    cli = _client()
    orders = _get_open_orders(cli)
    if not orders:
        log.info("manage_stale_orders | none open")
        return

    now = datetime.now(timezone.utc)
    canceled = 0

    for o in orders:
        try:
            # Only parent entries of bracket orders (ignore legs)
            if (o.order_class or "").lower() != "bracket":
                continue
            # Some SDKs expose created_at or submitted_at; pick whichever exists
            created = getattr(o, "created_at", None) or getattr(o, "submitted_at", None)
            if created is None:
                # Safety: if no timestamp, skip
                continue

            age_min = (now - created).total_seconds() / 60.0
            if age_min >= MAX_MIN:
                try:
                    cli.cancel_order_by_id(o.id)
                    canceled += 1
                    log.info("canceled stale entry %s (%d min) %s %s lmt=%s",
                             o.id, int(age_min), getattr(o, "symbol", "?"),
                             getattr(o, "side", "?"), getattr(o, "limit_price", "?"))
                except Exception as e:
                    log.warning("cancel failed %s: %s", o.id, e)
        except Exception as e:
            log.warning("order loop error: %s", e)

    if canceled:
        log.info("manage_stale_orders | canceled %d orders", canceled)
    else:
        log.info("manage_stale_orders | none stale")

if __name__ == "__main__":
    main()
