# jobs/manage_stale_orders.py
import os, logging
from datetime import datetime, timezone, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manage_stale_orders")

ALPACA_KEY = os.getenv("ALPACA_API_KEY","")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET","")
MAX_MIN = int(os.getenv("STALE_CANCEL_MINUTES","45"))

def _trading_client():
    return TradingClient(ALPACA_KEY, ALPACA_SEC, paper=True)

def main():
    cli = _trading_client()
    try:
        orders = list(cli.get_orders(status=OrderStatus.OPEN))
    except Exception as e:
        log.warning("get_orders failed: %s", e)
        return

    now = datetime.now(timezone.utc)
    n=0
    for o in orders:
        # only consider parent entry legs
        if (o.order_class or "").lower() != "bracket":
            continue
        if o.status.value not in ("new","accepted","pending_new","open"):
            continue
        created = o.created_at or o.submitted_at or now
        age_min = (now - created).total_seconds() / 60.0
        if age_min >= MAX_MIN:
            try:
                cli.cancel_order_by_id(o.id)
                n+=1
                log.info("canceled stale entry %s (%s min)", o.id, int(age_min))
            except Exception as e:
                log.warning("cancel failed %s: %s", o.id, e)
    if n:
        log.info("manage_stale_orders | canceled %d orders", n)
    else:
        log.info("manage_stale_orders | none stale")

if __name__ == "__main__":
    main()
