# jobs/partial_fills.py
import os, logging, time
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("partial_fills")

def _cli():
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def main():
    cli = _cli()
    max_age = int(os.getenv("PARTIAL_MAX_AGE_SEC", "300"))

    try:
        orders = cli.get_orders()
    except Exception as e:
        log.warning("get_orders failed: %s", e)
        return

    now = time.time()
    n_cancel = 0
    for o in orders:
        status = str(getattr(o, "status", "")).lower()
        if status not in ("partially_filled", "new", "open", "accepted", "pending_new"):
            continue

        created_at = getattr(o, "created_at", None)
        if not created_at:
            continue
        # created_at may be ISO string; be robust
        try:
            ts = getattr(created_at, "timestamp", None)
            age = now - (ts() if callable(ts) else created_at.timestamp())
        except Exception:
            age = 0

        filled = float(getattr(o, "filled_qty", 0) or 0)
        qty    = float(getattr(o, "qty", 0) or 0)

        if filled > 0 and filled < qty and age >= max_age:
            try:
                cli.cancel_order(o.id)
                n_cancel += 1
                log.info("Canceled lingering partial %s %s filled=%s/%s age=%.0fs",
                         o.symbol, o.side, filled, qty, age)
            except Exception as e:
                log.warning("cancel failed %s: %s", o.id, e)

    log.info("partial_fills | canceled=%d", n_cancel)

if __name__ == "__main__":
    main()
