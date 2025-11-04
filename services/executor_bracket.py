# --- make local packages importable in IDE & runtime ---
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.requests import LimitOrderRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# === local utilities ===
from tools.atr import get_atr
from tools.quotes import get_bid_ask_mid
from tools.util import pg_connect, market_is_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === ENV CONFIG ===
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.45"))
TP_MULT = float(os.getenv("TP_MULTIPLIER", "1.5"))
SL_MULT = float(os.getenv("SL_MULTIPLIER", "1.0"))
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "0") == "1"
FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
ACCOUNT_FALLBACK_TO_CASH = os.getenv("ACCOUNT_FALLBACK_TO_CASH", "1") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")

# === BRACKET ORDER FUNCTION ===
def submit_bracket_order(client, symbol, side, limit_price, atr):
    """Submit bracket order with OCO (take-profit / stop-loss) logic."""
    try:
        if side == "buy":
            tp_price = round(limit_price + atr * TP_MULT, 2)
            sl_price = round(limit_price - atr * SL_MULT, 2)
        else:
            tp_price = round(limit_price - atr * TP_MULT, 2)
            sl_price = round(limit_price + atr * SL_MULT, 2)

        order = client.submit_order(
            symbol=symbol,
            qty=1,
            side=side,
            type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": tp_price},
            stop_loss={"stop_price": sl_price},
        )

        logger.info(f"{symbol} {side.upper()} | limit={limit_price} TP={tp_price} SL={sl_price} | id={order.id}")
        return order

    except Exception as e:
        logger.error(f"submit_bracket_order failed for {symbol}: {e}")
        return None


# === MAIN EXECUTION ===
def main():
    since_min = int(os.getenv("SINCE_MIN", "180"))
    min_strength = float(os.getenv("MIN_STRENGTH", "0.45"))

    logger.info(f"executor_bracket | since-min={since_min} min_strength={min_strength} | fractional={FRACTIONAL} long_only={LONG_ONLY}")

    # --- Connect DB ---
    conn = pg_connect()
    sql = """
        SELECT symbol, side, strength, px
        FROM signals
        WHERE created_at > NOW() - INTERVAL '%s'
          AND ABS(strength) >= %s
        ORDER BY created_at DESC
    """
    df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    conn.close()

    if df.empty:
        logger.info(f"no qualifying signals in last {since_min} min (>= {min_strength})")
        return

    # --- Alpaca client ---
    c = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)
    acct = c.get_account()
    buying_power = float(acct.buying_power or 0)
    if buying_power < MIN_ACCOUNT_BP_USD:
        if ACCOUNT_FALLBACK_TO_CASH:
            buying_power = float(acct.cash)
            logger.warning(f"buying_power reported 0; falling back to cash={buying_power}")
        else:
            logger.warning("buying_power insufficient — skip run")
            return

    # --- Market hours check ---
    if not ALLOW_AFTER_HOURS and not market_is_open():
        logger.info("market is closed and ALLOW_AFTER_HOURS=0 -> skip this pass")
        return

    # --- Iterate recent signals ---
    for _, row in df.iterrows():
        symbol = row.symbol
        side = row.side.lower()
        strength = float(row.strength)
        px = float(row.px)

        if LONG_ONLY and side == "sell":
            logger.info(f"{symbol}: LONG_ONLY=1 -> skip short")
            continue

        bid, ask, mid = get_bid_ask_mid(symbol)
        if not bid or not ask:
            logger.warning(f"{symbol}: missing bid/ask -> skip")
            continue

        spread_abs = abs(ask - bid)
        spread_pct = spread_abs / mid * 100
        if spread_pct > 0.2:
            logger.info(f"{symbol}: skip wide spread abs={spread_abs:.4f} pct={spread_pct:.3f}%")
            continue

        try:
            atr = get_atr(symbol, period=14, lookback_days=30)
        except Exception as e:
            logger.warning(f"{symbol}: ATR fetch failed: {e} -> skip")
            continue

        limit_price = round(mid, 2)
        submit_bracket_order(c, symbol, side, limit_price, atr)

    logger.info("All qualifying bracket orders submitted.")


if __name__ == "__main__":
    main()
