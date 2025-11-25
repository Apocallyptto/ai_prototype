import os
import time
import logging
from typing import List

from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from utils import get_engine, compute_atr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_executor")


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.20"))

# napr. "AAPL,MSFT,SPY"
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
SYMBOLS = [s.strip().upper() for s in SYMBOLS if s.strip()]

POLL_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "20"))
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")  # default 1
DEFAULT_ENTRY_PRICE = float(os.getenv("DEFAULT_ENTRY_PRICE", "200.0"))

# ATR / entry tuning
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
    Fetch RULES + ML (gbc_5m) signály, ktoré spĺňajú:
      - strength >= MIN_STRENGTH
      - symbol v SYMBOLS
      - source IN ('rules', 'ml_gbc_5m')
      - (portfolio_id = PORTFOLIO_ID alebo NULL)
      - created_at v posledných 30 minútach
    """

    symbols_list = ",".join(f"'{s}'" for s in SYMBOLS)

    sql = text(
        f"""
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
    )

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "min_strength": MIN_STRENGTH,
                # 👇 POSIELAME STRING, NIE int()
                "pid": PORTFOLIO_ID,
            },
        ).mappings().all()

    return list(rows)



# ---------------------------------------------------------
# ORDER CREATION
# ---------------------------------------------------------
def create_limit_order(symbol: str, side: str, strength: float):
    """
    Vytvorí limitný príkaz pomocou ATR logiky (compute_atr).
    """

    # ATR + posledná cena z utils.compute_atr()
    atr_val, last_price = compute_atr(symbol)

    if last_price is None:
        last_price = DEFAULT_ENTRY_PRICE

    side_l = side.lower()
    if side_l == "buy":
        entry_price = last_price * (1 + ATR_PCT)
    else:
        entry_price = last_price * (1 - ATR_PCT)

    entry_price = round(entry_price, 2)
    qty = 1  # zatiaľ fixne, neskôr môžeme napojiť risk mgmt

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide(side_l),
        limit_price=entry_price,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = trading_client.submit_order(req)
        logger.info(
            f"Created order: {symbol} {side_l.upper()} @ {entry_price} | "
            f"order_id={order.id} | strength={strength:.4f}"
        )
    except Exception as e:
        logger.error(f"Failed to create order: {symbol} {side_l} | {e}")


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main_loop():
    logger.info(
        "signal_executor starting | "
        f"MIN_STRENGTH={MIN_STRENGTH} | SYMBOLS={SYMBOLS} | "
        f"PORTFOLIO_ID={PORTFOLIO_ID} | POLL={POLL_SECONDS}s | "
        f"ATR_PCT={ATR_PCT:.4f} | TP_ATR_MULT={TP_ATR_MULT} | "
        f"SL_ATR_MULT={SL_ATR_MULT}"
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
                    side = sig["side"]
                    strength = float(sig["strength"])
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
