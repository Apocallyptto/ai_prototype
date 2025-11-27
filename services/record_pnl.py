import os
import time
import logging
from datetime import date

import sqlalchemy as sa
from sqlalchemy import text
from alpaca.trading.client import TradingClient

from utils import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pnl_recorder")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
PNL_POLL_SECONDS = int(os.getenv("PNL_POLL_SECONDS", "3600"))  # default: 1h

# DB + Alpaca klient
engine = get_engine()
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=os.getenv("TRADING_MODE", "paper") == "paper",
)


def already_recorded_today() -> bool:
    """Check, whether we already have PnL record for today."""
    sql = text(
        """
        SELECT 1
        FROM daily_pnl
        WHERE as_of_date = CURRENT_DATE
          AND portfolio_id = :pid
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        row = conn.execute(sql, {"pid": PORTFOLIO_ID}).first()
    return row is not None


def record_pnl_once():
    """Fetch account from Alpaca and store one row into daily_pnl."""
    if already_recorded_today():
        logger.info("PnL for today already recorded (portfolio_id=%s)", PORTFOLIO_ID)
        return

    try:
        account = trading_client.get_account()
    except Exception as e:
        logger.error("Failed to fetch account from Alpaca: %s", e)
        return

    try:
        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)
        long_mv = float(account.long_market_value)
        short_mv = float(account.short_market_value)
    except Exception as e:
        logger.error("Failed to parse account numbers: %s", e)
        return

    logger.info(
        "Recording PnL for %s | equity=%.2f cash=%.2f bp=%.2f pv=%.2f",
        date.today(),
        equity,
        cash,
        buying_power,
        portfolio_value,
    )

    sql = text(
        """
        INSERT INTO daily_pnl (
            as_of_date,
            portfolio_id,
            equity,
            cash,
            buying_power,
            portfolio_value,
            long_market_value,
            short_market_value
        )
        VALUES (
            CURRENT_DATE,
            :pid,
            :equity,
            :cash,
            :bp,
            :pv,
            :long_mv,
            :short_mv
        )
        ON CONFLICT (as_of_date, portfolio_id)
        DO UPDATE
        SET
            equity = EXCLUDED.equity,
            cash = EXCLUDED.cash,
            buying_power = EXCLUDED.buying_power,
            portfolio_value = EXCLUDED.portfolio_value,
            long_market_value = EXCLUDED.long_market_value,
            short_market_value = EXCLUDED.short_market_value,
            created_at = NOW();
        """
    )

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "pid": PORTFOLIO_ID,
                "equity": equity,
                "cash": cash,
                "bp": buying_power,
                "pv": portfolio_value,
                "long_mv": long_mv,
                "short_mv": short_mv,
            },
        )

    logger.info("PnL record stored / updated successfully.")


def main_loop():
    logger.info(
        "pnl_recorder starting | PORTFOLIO_ID=%s | PNL_POLL_SECONDS=%s",
        PORTFOLIO_ID,
        PNL_POLL_SECONDS,
    )

    while True:
        try:
            record_pnl_once()
        except Exception as e:
            logger.exception("Error in PnL loop: %s", e)
        time.sleep(PNL_POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
