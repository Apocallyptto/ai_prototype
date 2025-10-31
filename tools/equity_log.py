import os, logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from alpaca.trading.client import TradingClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("equity_log")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

def _cli():
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def _ensure_table(engine):
    with engine.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            ts timestamptz PRIMARY KEY,
            equity numeric,
            cash numeric,
            buying_power numeric,
            portfolio_value numeric
        );
        """))

def main():
    eng = create_engine(DB_URL)
    _ensure_table(eng)
    cli = _cli()
    acct = cli.get_account()
    row = {
        "ts": datetime.now(timezone.utc),
        "equity": float(getattr(acct, "equity", 0) or 0),
        "cash": float(getattr(acct, "cash", 0) or 0),
        "buying_power": float(getattr(acct, "buying_power", 0) or 0),
        "portfolio_value": float(getattr(acct, "portfolio_value", 0) or 0),
    }
    with eng.begin() as con:
        con.execute(text("""
            INSERT INTO equity_snapshots(ts,equity,cash,buying_power,portfolio_value)
            VALUES (:ts,:equity,:cash,:buying_power,:portfolio_value)
            ON CONFLICT (ts) DO NOTHING
        """), row)
    log.info("logged equity=%.2f bp=%.2f cash=%.2f pv=%.2f",
             row["equity"], row["buying_power"], row["cash"], row["portfolio_value"])

if __name__ == "__main__":
    main()
