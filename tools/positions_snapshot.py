import os, logging, math
from datetime import datetime
from alpaca.trading.client import TradingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("positions_snapshot")

def _cli():
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def main():
    cli = _cli()
    acct = cli.get_account()
    cash = float(getattr(acct, "cash", 0) or 0)
    bp   = float(getattr(acct, "buying_power", 0) or 0)
    eq   = float(getattr(acct, "equity", 0) or 0)
    pv   = float(getattr(acct, "portfolio_value", 0) or 0)
    log.info("ACCOUNT | equity=%.2f cash=%.2f buying_power=%.2f portfolio_value=%.2f", eq, cash, bp, pv)

    pos = cli.get_all_positions()
    if not pos:
        log.info("No open positions.")
        return

    total_long = total_short = total_unreal = 0.0
    print("\nSYMBOL  SIDE   QTY     AVG_PX   MKT_PX   PNL$    NOTIONAL")
    print("------  -----  ------  -------  -------  ------  --------")
    for p in pos:
        sym   = p.symbol
        side  = str(p.side).split(".")[-1].upper()
        qty   = float(p.qty)
        avg   = float(p.avg_entry_price or 0)
        mkt   = float(p.current_price or 0)
        pnl   = float(p.unrealized_pl or 0)
        notion= qty * mkt
        if side == "LONG":
            total_long += notion
        else:
            total_short += notion
        total_unreal += pnl
        print(f"{sym:<6}  {side:<5}  {qty:<6.2f}  {avg:<7.2f}  {mkt:<7.2f}  {pnl:<6.2f}  {notion:<8.2f}")

    print("\nEXPOSURE | long=%.2f  short=%.2f  net=%.2f  UPL=%.2f" %
          (total_long, total_short, total_long-total_short, total_unreal))

if __name__ == "__main__":
    main()
