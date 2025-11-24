import os
import time
import logging
import datetime as dt
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from utils.db import get_engine
from utils.atr import compute_atr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_executor")


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.20"))
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
POLL_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "20"))
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")  # default 1
DEFAULT_ENTRY_PRICE = float(os.getenv("DEFAULT_ENTRY_PRICE", "200.0"))

ATR_PCT = float(os.getenv("ATR_PCT", "0.01"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))


# ---------------------------------------------------------
# DB + ALPACA CLIENT
# ---------------------------------------------------------
engine = get_engine()
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=os.getenv("TRADING_MODE", "paper") == "paper",
)


# ---------------------------------------------------------
# FETCH NEW SIGNALS
# ---------------------------------------------------------
def fetch_new_signals() -> List[dict]:
    """
    Fetch ML + RULES signals that match:
      - strength >= MIN_STRENGTH
      - symbol in SYMBOLS
      - source IN ('rules', 'ml_gbc_5m')
      - matches portfolio_id (or portfolio_id IS NULL)
      - created in last 30 minutes
    """

    symbols_list = ",".join(f"'{s.strip()}'" for s in SYMBOLS if s.strip())

    sql_str = f"""
        SELECT
            id,
            created_at,
            symbol,
            side,
            strength,
            source,
            portfolio_id
        FROM signals
        WHERE strength >= :min_strength
          AND symbol IN ({symbols_list})
          AND source IN ('rules', 'ml_gbc_5m')
          AND (portfolio_id = :pid OR portfolio_id IS NULL)
          AND created_at >= (NOW() - INTERVAL '30 minutes')
        ORDER BY created_at ASC
    """

    sql = text(sql_str)

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "min_strength": MIN_STRENGTH,
                "pid": int(PORTFOLIO_ID),
            },
        ).mappings().all()

    return list(rows)


# ---------------------------------------------------------
# ORDER CREATION
# ---------------------------------------------------------
def create_limit_order(symbol: str, side: str, strength: float):
    """
    Place limit order using ATR-based logic.
    """

    atr_val, last_price = compute_atr(symbol)

    if last_price is None:
        last_price = DEFAULT_ENTRY_PRICE

    if side.lower() == "buy":
        entry_price = last_price * (1 + ATR_PCT)
    else:
        entry_price = last_price * (1 - ATR_PCT)

    entry_price = round(entry_price, 2)
    qty = 1

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide(side),
        limit_price=entry_price,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = trading_client.submit_order(req)
        logger.info(
            f"Created order: {symbol} {side.upper()} @ {entry_price} | order_id={order.id}"
        )
    except Exception as e:
        logger.error(f"Failed to create order: {symbol} {side} | {e}")


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main_loop():
    logger.info(
        f"signal_executor starting | "
        f"MIN_STRENGTH={MIN_STRENGTH} | SYMBOLS={SYMBOLS} | "
        f"PORTFOLIO_ID={PORTFOLIO_ID} | POLL={POLL_SECONDS}s | "
        f"ATR_PCT={ATR_PCT:.4f} | TP_ATR_MULT={TP_ATR_MULT} | SL_ATR_MULT={SL_ATR_MULT}"
    )

    while True:
        try:
            signals = fetch_new_signals()

            if not signals:
                logger.info("no new signals")
            else:
                logger.info(f"Processing {len(signals)} signal(s)")

                for sig in signals:
                    symbol = sig["symbol"]
                    side = sig["side"].lower()
                    strength = sig["strength"]
                    source = sig["source"]

                    logger.info(
                        f"EXEC: {symbol} {side.upper()} | "
                        f"strength={strength:.4f} | source={source}"
                    )

                    create_limit_order(symbol, side, strength)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            logger.exception(f"Loop error: {e}")
            time.sleep(POLL_SECONDS)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main_loop()
