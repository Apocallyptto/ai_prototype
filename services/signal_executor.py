import os
import time
import logging
from typing import List, Tuple

import sqlalchemy as sa
from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
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

# --- RISK / SIZING ---
MAX_RISK_PER_TRADE_USD = float(os.getenv("MAX_RISK_PER_TRADE_USD", "50"))       # $ riziko na obchod
MAX_QTY_PER_TRADE = int(os.getenv("MAX_QTY_PER_TRADE", "10"))                   # max shares
MAX_NOTIONAL_PER_TRADE_USD = float(os.getenv("MAX_NOTIONAL_PER_TRADE_USD", "1000"))  # max $ veľkosť
MAX_BP_PCT_PER_TRADE = float(os.getenv("MAX_BP_PCT_PER_TRADE", "0.25"))         # max 25% buying power na obchod

# cooldown na chyby typu "insufficient buying power" / "insufficient qty"
INSUFFICIENT_COOLDOWN_SECONDS = int(
    os.getenv("INSUFFICIENT_COOLDOWN_SECONDS", "900")  # 15 min default
)


# ---------------------------------------------------------
# DB + ALPACA CLIENT
# ---------------------------------------------------------
engine = get_engine()
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=os.getenv("TRADING_MODE", "paper") == "paper",
)

# (symbol, side) -> timestamp (do kedy skipnúť kvôli insufficient error)
INSUFFICIENT_COOLDOWN: dict[Tuple[str, str], float] = {}


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
# HELPERS – ACCOUNT & ORDERS
# ---------------------------------------------------------
def get_buying_power() -> float:
    try:
        acc = trading_client.get_account()
        bp = float(acc.buying_power)
        return bp
    except Exception as e:
        logger.error("get_buying_power() failed: %s", e)
        # fallback – radšej žiadny trade ako random
        return 0.0


def get_open_orders_for_symbol(symbol: str):
    """
    Return all OPEN orders for given symbol from Alpaca.
    Používa nový alpaca-py štýl: GetOrdersRequest + filter=
    """
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            limit=500,  # nech vidíme naozaj všetko
        )
        orders = trading_client.get_orders(filter=req)
        orders_list = list(orders) if orders is not None else []
        logger.info(
            "get_open_orders_for_symbol(%s) -> %d open order(s)",
            symbol,
            len(orders_list),
        )
        return orders_list
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
                getattr(o, "limit_price", None),
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
                    getattr(o, "limit_price", None),
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
# POSITION SIZING
# ---------------------------------------------------------
def compute_order_qty(symbol: str, side: str, entry_price: float, atr_val: float | None) -> int:
    """
    Vypočíta qty podľa:
      - MAX_RISK_PER_TRADE_USD
      - MAX_NOTIONAL_PER_TRADE_USD
      - MAX_BP_PCT_PER_TRADE
      - ATR & SL_ATR_MULT
    """
    bp = get_buying_power()
    if bp <= 0:
        logger.warning("compute_order_qty(%s %s) -> buying power <= 0, skip", symbol, side.upper())
        return 0

    # max $ risk na jeden trade – obmedzené aj buying power
    max_risk_dollars = min(MAX_RISK_PER_TRADE_USD, bp * MAX_BP_PCT_PER_TRADE)

    # risk na 1 share = ATR * SL_MULT, fallback 2% ceny
    if atr_val is not None and atr_val > 0:
        risk_per_share = atr_val * SL_ATR_MULT
    else:
        risk_per_share = entry_price * 0.02

    if risk_per_share <= 0:
        logger.warning("compute_order_qty(%s %s) -> risk_per_share <= 0, skip", symbol, side.upper())
        return 0

    qty_by_risk = int(max_risk_dollars / risk_per_share)

    # notional limit = min( MAX_NOTIONAL, MAX_BP_PCT * BP )
    notional_limit = min(MAX_NOTIONAL_PER_TRADE_USD, bp * MAX_BP_PCT_PER_TRADE)
    qty_by_notional = int(notional_limit / entry_price)

    qty = min(qty_by_risk, qty_by_notional, MAX_QTY_PER_TRADE)
    if qty < 1:
        logger.info(
            "compute_order_qty(%s %s) -> qty<1 (risk=%s, notional=%s, bp=%s)",
            symbol,
            side.upper(),
            max_risk_dollars,
            notional_limit,
            bp,
        )
        return 0

    logger.info(
        "compute_order_qty(%s %s) -> qty=%d (risk_per_share=%.4f, max_risk=%.2f, notional_limit=%.2f, bp=%.2f)",
        symbol,
        side.upper(),
        qty,
        risk_per_share,
        max_risk_dollars,
        notional_limit,
        bp,
    )
    return qty


# ---------------------------------------------------------
# ORDER CREATION
# ---------------------------------------------------------
def create_limit_order(symbol: str, side: str, strength: float):
    """
    Place limit order using ATR-based logic + guardrails:
      - skip if same-side open order already exists
      - cancel opposite-side open order before placing new
      - cooldown po 'insufficient buying power/qty' errore
      - position sizing podľa risk parametrov
    """
    side = side.lower()
    key = (symbol, side)
    now_ts = time.time()

    # 0) cooldown po insufficient error
    until = INSUFFICIENT_COOLDOWN.get(key)
    if until is not None and now_ts < until:
        logger.info(
            "Skip %s %s: still in insufficient cooldown until %s",
            symbol,
            side.upper(),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)),
        )
        return

    # 1) wash-trade guard
    if not cleanup_and_check(symbol, side):
        # same-side order already exists, skip
        return

    # 2) Compute ATR / last price
    atr_val, last_price = compute_atr(symbol)

    if last_price is None:
        last_price = DEFAULT_ENTRY_PRICE

    if side == "buy":
        entry_price = last_price * (1 + ATR_PCT)
    else:
        entry_price = last_price * (1 - ATR_PCT)

    entry_price = round(entry_price, 2)

    # 3) position sizing
    qty = compute_order_qty(symbol, side, entry_price, atr_val)
    if qty < 1:
        logger.info("Skip %s %s: computed qty < 1, no trade", symbol, side.upper())
        return

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
            "Created order: %s %s @ %s x%d | order_id=%s",
            symbol,
            side.upper(),
            entry_price,
            qty,
            order.id,
        )
    except Exception as e:
        msg = str(e)
        logger.error("Failed to create order: %s %s | %s", symbol, side, msg)

        # jednoduchý risk guard – ak nemáš buying power/qty,
        # nastavíme cooldown, aby to nespamovalo každých 20 sekúnd
        if "insufficient buying power" in msg or "insufficient qty available" in msg:
            cooldown_until = now_ts + INSUFFICIENT_COOLDOWN_SECONDS
            INSUFFICIENT_COOLDOWN[key] = cooldown_until
            logger.warning(
                "Set insufficient cooldown for %s %s for %d seconds (until %s)",
                symbol,
                side.upper(),
                INSUFFICIENT_COOLDOWN_SECONDS,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until)),
            )


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
