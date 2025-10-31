# tools/bp_log.py
import os, logging
from alpaca.trading.client import TradingClient

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bp_log")

def main():
    cli = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
    acct = cli.get_account()
    log.info("ACCOUNT | equity=%s cash=%s buying_power=%s portfolio_value=%s",
             getattr(acct, "equity", "?"),
             getattr(acct, "cash", "?"),
             getattr(acct, "buying_power", "?"),
             getattr(acct, "portfolio_value", "?"))

if __name__ == "__main__":
    main()
