# jobs/manage_stale_orders.py
import os, logging
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manage_stale_orders")

ALPACA_KEY = os.getenv("ALPACA_API_KEY","")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET","")
APCA_BASE  = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
MAX_MIN    = int(os.getenv("STALE_CANCEL_MINUTES","45"))

OPENISH = {"new","accepted","held","pending_new","open","partially_filled"}  # strings only

def _client():
    return TradingClient(ALPACA_KEY, ALPACA_SEC, paper="paper" in APCA_BASE)

def _safe_get_orders(cli: TradingClient):
    # Try no-arg first (most compatible)
    try:
        return list(cli.get_orders())
    except Exception as e:
        log.warning("get_orders() failed: %s", e)
        # Try request object if available
        try:
            from alpaca.trading.requests import GetOrdersRequest
            req = GetOrdersRequest()  # no filters; some SDKs require a request object
            return list(cli.get_orders(filter=req))
        except Exception as e2:
            log.warning("fallback get_orders(filter=req) failed: %s", e2)
            return []

def main():
    cli = _client()
    orders = _safe_get_orders(cli)
    if not orders:
        log.info("manage_stale_orders | none open")
        return

    now = datetime.now(timezone.utc)
    canceled = 0

    for o in orders:
        try:
            status = str(getattr(o, "status", "")).lower()
            if status not in OPENISH:
                continue
            # only parent entries of bracket orders (ignore child legs)
            if (str(getattr(o, "order_class", "")).lower() != "bracket") or getattr(o, "parent_order_id", None):
                # if parent_order_id exists, it's a child; skip
                continue

            created = getattr(o, "created_at", None) or getattr(o, "submitted_at", None) or now
            age_min = (now - created).total_seconds() / 60.0
            if age_min < MAX_MIN:
                continue

            try:
                cli.cancel_order_by_id(o.id)
                canceled += 1
                log.info("canceled stale entry %s (%d min) %s %s lmt=%s",
                         o.id, int(age_min),
                         getattr(o, "symbol", "?"), getattr(o, "side", "?"),
                         getattr(o, "limit_price", "?"))
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
