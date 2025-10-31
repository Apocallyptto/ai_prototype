# tools/free_bp.py
import os, logging, time
from alpaca.trading.client import TradingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("free_bp")

def main():
    cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
    try:
        log.info("Cancelling all open orders…")
        cli.cancel_orders()
    except Exception as e:
        log.warning("cancel_orders failed: %s", e)

    try:
        log.info("Closing all positions…")
        cli.close_all_positions(cancel_orders=True)
    except Exception as e:
        log.warning("close_all_positions failed: %s", e)

    time.sleep(2)
    acct = cli.get_account()
    log.info("Now: cash=%s buying_power=%s equity=%s",
             getattr(acct, "cash", "?"),
             getattr(acct, "buying_power", "?"),
             getattr(acct, "equity", "?"))

if __name__ == "__main__":
    main()
