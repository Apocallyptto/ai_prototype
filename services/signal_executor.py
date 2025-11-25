import os
import time
import logging
from typing import List

import sqlalchemy as sa
from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from utils import get_engine, compute_atr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_executor")


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.20"))
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
POLL_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "20"))
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")
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
          AND (portfolio_id::text = :pid OR portfolio_id IS NULL)
          AND created_at >= (NOW() - INTERVAL '30 minutes')
        ORDER BY created_at ASC
    """

    logger.info(
        "fetch_new_signals | MIN_STRENGTH=%s | SYMBOLS=%s | PORTFOLIO_ID=%s",
        MIN_STRENGTH,
        SYMBOLS,
        PORTFOLIO_ID,
    )
    logger.info("fetch_new_signals | SQL:\n%s", sql_str)

    sql = text(sql_str)

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "min_strength": MIN_STRENGTH,
                "pid": str(PORTFOLIO_ID),
            },
        ).mappings().all()

    logger.info("fetch_new_signals | fetched %d rows", len(rows))
    return list(rows)


# ---------------------------------------------------------
# HELPERS AROUND OPEN ORDERS
# ---------------------------------------------------------
def get_open_orders_for_symbol(symbol: str):
    """
    Return all OPEN orders for given symbol from Alpaca.
    """
    try:
        orders = trading_client.get_orders(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            nested=False,
        )
        return list(orders)
    except Exception as e:
        logger.error("get_open_orders_for_symbol(%s) failed: %s", symbol, e)
        return []


def cleanup_and_check(symbol: str, side: str) -> bool:
    """
    - If there is an open order with the same side => skip (return False).
    - If there is an opposite-side order => cancel it, then allow new (return True).
    """
    desired_side = side.lower()
    opposite_side = "buy" if desired_side == "sell" else "sell"

    open_orders = get_open_orders_for_symbol(symbol)
    if not open_orders:
        return True

    # 1) same-side -> skip
    for o in open_orders:
        o_side = o.side.value.lower()
        if o_side == desired_side:
            logger.info(
                "Skip %s %s: open %s order %s already exists at %s",
                symbol,
                desired_side.upper(),
                o_side.upper(),
                o.id,
                o.limit_price,
            )
            return False

    # 2) opposite-side -> cancel it, then continue
    for o in open_orders:
        o_side = o.side.value.lower()
        if o_side == opposite_side:
            try:
                trading_client.cancel_order_by_id(o.id)
                logger.info(
                    "Canceled opposite %s order %s for %s at %s before new %s",
                    o_side.upper(),
                    o.id,
                    symbol,
                    o.limit_price,
                    desired_side.upper(),
                )
            except Exception as e:
                logger.error(
                    "Failed to cancel opposite order %s for %s: %s",
                    o.id,
                    symbol,
                    e,
                )
    return True


# ---------------------------------------------------------
# ORDER CREATION
# ---------------------------------------------------------
def create_limit_order(symbol: str, side: str, strength: float):
    """
    Place limit order using ATR-based logic + wash-trade guard:
      - skip if same-side open order already exists
      - cancel opposite-side open order before placing new
    """

    if not cleanup_and_check(symbol, side):
        # same-side order already exists, skip
        return

    # Compute ATR / last price
    atr_val, last_price = compute_atr(symbol)

    if last_price is None:
        last_price = DEFAULT_ENTRY_PRICE

    if side.lower() == "buy":
        entry_price = last_price * (1 + ATR_PCT)
    else:
        entry_price = last_price * (1 - ATR_PCT)

    entry_price = round(entry_price, 2)
    qty = 1  # TODO: neskôr prepočítať podľa risku

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
            "Created order: %s %s @ %s | order_id=%s",
            symbol,
            side.upper(),
            entry_price,
            order.id,
        )
    except Exception as e:
        logger.error("Failed to create order: %s %s | %s", symbol, side, e)


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main_loop():
    logger.info(
        "signal_executor starting | "
        "MIN_STRENGTH=%s | SYMBOLS=%s | "
        "PORTFOLIO_ID=%s | POLL=%ss | "
        "ATR_PCT=%.4f | TP_ATR_MULT=%.1f | SL_ATR_MULT=%.1f",
        MIN_STRENGTH,
        SYMBOLS,
        PORTFOLIO_ID,
        POLL_SECONDS,
        ATR_PCT,
        TP_ATR_MULT,
        SL_ATR_MULT,
    )

    while True:
        try:
            signals = fetch_new_signals()

            if not signals:
                logger.info("no new signals")
            else:
                logger.info("Processing %d signal(s)", len(signals))

                for sig in signals:
                    symbol = sig["symbol"]
                    side = sig["side"].lower()
                    strength = sig["strength"]
                    source = sig["source"]

                    logger.info(
                        "EXEC: %s %s | strength=%.4f | source=%s",
                        symbol,
                        side.upper(),
                        strength,
                        source,
                    )

                    create_limit_order(symbol, side, strength)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            logger.exception("Loop error: %s", e)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
