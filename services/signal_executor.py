import os
import time
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

from sqlalchemy import text

from utils import get_engine, compute_atr
from services.alpaca_exit_guard import (
    cancel_exit_orders,
    wait_exit_orders_cleared,
    get_position_qty,
    cancel_related_orders_from_exception,
)

logger = logging.getLogger("signal_executor")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.6"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

ATR_PCT = float(os.getenv("ATR_PCT", "0.01"))  # 1%
DEFAULT_ENTRY_PRICE = float(os.getenv("DEFAULT_ENTRY_PRICE", "100.00"))

RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "0.01"))  # 1% of equity
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))      # 10% of equity per symbol

ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").strip().lower() in ("1", "true", "yes", "y")
CLOSE_WITH_MARKET = os.getenv("CLOSE_WITH_MARKET", "true").strip().lower() in ("1", "true", "yes", "y")
EXIT_CLEAR_RETRIES = int(os.getenv("EXIT_CLEAR_RETRIES", "8"))
EXIT_CLEAR_SLEEP_S = float(os.getenv("EXIT_CLEAR_SLEEP_S", "0.4"))
EXIT_CLEAR_WAIT_S = float(os.getenv("EXIT_CLEAR_WAIT_S", "2.0"))

EXIT_OCO_PREFIX = os.getenv("EXIT_OCO_PREFIX", "EXIT-OCO").strip() or "EXIT-OCO"

ENABLE_DAILY_RISK_GUARD = os.getenv("ENABLE_DAILY_RISK_GUARD", "false").strip().lower() in ("1", "true", "yes", "y")
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "200"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "2.0"))  # e.g. 2%


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------
PROCESSED_SIGNAL_IDS = set()
INSUFFICIENT_COOLDOWN = {}  # (symbol, side)-> epoch seconds


# ---------------------------------------------------------
# ALPACA + DB
# ---------------------------------------------------------
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=(os.getenv("TRADING_MODE", "paper") == "paper"),
)

engine = get_engine()


# ---------------------------------------------------------
# DB â€“ SIGNALS
# ---------------------------------------------------------
def fetch_new_signals():
    sql = text(
        """
        SELECT
            id,
            created_at,
            symbol,
            side,
            strength,
            source,
            portfolio_id
        FROM signals
        WHERE created_at > now() - interval '30 minutes'
          AND strength >= :min_strength
          AND portfolio_id = (:pid)::int
          AND symbol = ANY(:symbols)
        ORDER BY created_at ASC
        """
    )

    with engine.begin() as c:
        rows = c.execute(
            sql,
            {"min_strength": MIN_STRENGTH, "pid": PORTFOLIO_ID, "symbols": SYMBOLS},
        ).mappings().all()

    logger.info("fetch_new_signals | fetched %d rows", len(rows))
    return [dict(r) for r in rows]


def select_signals(signals):
    selected = []
    seen_symbol_side = set()

    for sig in signals:
        sid = sig["id"]
        if sid in PROCESSED_SIGNAL_IDS:
            continue

        sym = str(sig["symbol"]).upper()
        side = str(sig["side"]).lower()

        key = (sym, side)
        if key in seen_symbol_side:
            continue

        seen_symbol_side.add(key)
        selected.append(sig)

    logger.info(
        "select_signals | fetched=%d | new=%d | unique_symbol_side=%d",
        len(signals),
        len(selected),
        len(seen_symbol_side),
    )
    if len(PROCESSED_SIGNAL_IDS) > 10000:
        logger.info("select_signals | clearing PROCESSED_SIGNAL_IDS (size>10000)")
        PROCESSED_SIGNAL_IDS.clear()
    return selected


def mark_signal_processed(sig: dict):
    PROCESSED_SIGNAL_IDS.add(sig["id"])


# ---------------------------------------------------------
# HELPERS â€“ ACCOUNT & ORDERS
# ---------------------------------------------------------
def _is_exit_oco_order(order) -> bool:
    cid = getattr(order, "client_order_id", None) or ""
    sym = getattr(order, "symbol", None)
    if not sym:
        return False
    return cid.startswith(f"{EXIT_OCO_PREFIX}-{str(sym).upper()}-")


def get_buying_power() -> float:
    a = trading_client.get_account()
    return float(a.buying_power)


def get_equity() -> float:
    a = trading_client.get_account()
    return float(a.equity)


def get_open_orders_for_symbol(symbol: str, include_exit_oco: bool = True):
    from alpaca.trading.requests import GetOrdersRequest

    r = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol.upper()], limit=500)
    orders = list(trading_client.get_orders(r))
    if include_exit_oco:
        return orders
    return [o for o in orders if not _is_exit_oco_order(o)]


def cleanup_and_check(symbol: str, desired_side: str) -> bool:
    desired_side = desired_side.lower()
    orders = get_open_orders_for_symbol(symbol, include_exit_oco=False)

    same_side = []
    opp_side = []
    for o in orders:
        o_side = str(o.side).lower()
        if o_side == desired_side:
            same_side.append(o)
        else:
            opp_side.append(o)

    if same_side:
        o = same_side[0]
        lp = getattr(o, "limit_price", None)
        logger.info(
            "Skip %s %s: open %s order %s already exists%s",
            symbol,
            desired_side.upper(),
            desired_side.upper(),
            str(o.id),
            f" at {lp}" if lp else "",
        )
        return False

    canceled = 0
    for o in opp_side:
        try:
            trading_client.cancel_order_by_id(str(o.id))
            canceled += 1
        except Exception as e:
            logger.warning("Failed cancel opposite order %s: %s", o.id, e)

    if canceled:
        logger.info("Canceled %d opposite open order(s) for %s before new %s", canceled, symbol, desired_side.upper())

    return True


def compute_order_qty(symbol: str, side: str, entry_price: float, atr_val: float) -> int:
    eq = get_equity()
    bp = get_buying_power()

    stop_dist = float(atr_val or 0.0)
    if stop_dist <= 0:
        stop_dist = entry_price * ATR_PCT

    risk_dollars = eq * RISK_PCT_PER_TRADE
    qty_risk = int(risk_dollars / stop_dist) if stop_dist > 0 else 0

    max_notional = eq * MAX_POSITION_PCT
    qty_cap = int(max_notional / entry_price) if entry_price > 0 else 0

    qty = max(0, min(qty_risk, qty_cap))

    if side.lower() == "buy":
        qty_bp = int(bp / entry_price) if entry_price > 0 else 0
        qty = min(qty, qty_bp)

    return int(qty)


# ---------------------------------------------------------
# DAILY RISK GUARD (optional)
# ---------------------------------------------------------
def check_daily_risk_guard() -> bool:
    if not ENABLE_DAILY_RISK_GUARD:
        return True

    try:
        with engine.begin() as c:
            row = c.execute(
                text(
                    """
                    SELECT pnl_usd, drawdown_pct
                    FROM daily_pnl
                    WHERE day = CURRENT_DATE
                    LIMIT 1
                    """
                )
            ).mappings().first()
    except Exception as e:
        logger.warning("Daily risk guard: could not query daily_pnl (%s). Guard disabled.", e)
        return True

    if not row:
        return True

    pnl = float(row.get("pnl_usd") or 0.0)
    dd = float(row.get("drawdown_pct") or 0.0)

    if pnl <= -abs(MAX_DAILY_LOSS_USD):
        logger.warning("Daily risk guard HIT: pnl_usd=%.2f <= -%.2f", pnl, abs(MAX_DAILY_LOSS_USD))
        return False
    if dd >= abs(MAX_DRAWDOWN_PCT):
        logger.warning("Daily risk guard HIT: drawdown_pct=%.2f >= %.2f", dd, abs(MAX_DRAWDOWN_PCT))
        return False

    return True


# ---------------------------------------------------------
# EXIT CLEARANCE + ORDER SUBMISSION
# ---------------------------------------------------------
def _side_to_cancel_for_close(order_side: OrderSide) -> OrderSide:
    return OrderSide.SELL if order_side == OrderSide.SELL else OrderSide.BUY


def submit_close_order(symbol: str, order_side: OrderSide, qty: int, last_price: float):
    if qty <= 0:
        logger.info("submit_close_order: qty<=0 for %s, skipping", symbol)
        return None

    side_to_cancel = _side_to_cancel_for_close(order_side)

    for attempt in range(1, EXIT_CLEAR_RETRIES + 1):
        try:
            canceled = cancel_exit_orders(trading_client, symbol, side_to_cancel=side_to_cancel)
            if canceled:
                logger.info("Canceled %d EXIT-OCO order(s) for %s before CLOSE %s (attempt %d/%d)",
                            canceled, symbol, order_side.name, attempt, EXIT_CLEAR_RETRIES)

            wait_exit_orders_cleared(trading_client, symbol, timeout_s=EXIT_CLEAR_WAIT_S, poll_s=0.25)

            if CLOSE_WITH_MARKET:
                req = MarketOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                if order_side == OrderSide.SELL:
                    limit_price = round(last_price * (1 - ATR_PCT), 2)
                else:
                    limit_price = round(last_price * (1 + ATR_PCT), 2)

                req = LimitOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY,
                )

            o = trading_client.submit_order(req)
            logger.info("CLOSE submitted: %s %s qty=%s | id=%s status=%s",
                        symbol, order_side.name, qty, str(o.id), str(o.status))
            return o

        except APIError as e:
            msg = str(e)
            if ("held_for_orders" in msg) or ("insufficient qty" in msg) or ("40310000" in msg):
                rel = cancel_related_orders_from_exception(trading_client, e)
                if rel:
                    logger.info("Canceled %d related order(s) from error for %s", rel, symbol)
                logger.warning("Close submit blocked by held qty (attempt %d/%d) for %s: %s",
                               attempt, EXIT_CLEAR_RETRIES, symbol, msg)
                time.sleep(EXIT_CLEAR_SLEEP_S)
                continue
            raise
        except Exception as e:
            logger.warning("Close submit attempt %d/%d failed for %s: %s", attempt, EXIT_CLEAR_RETRIES, symbol, e)
            time.sleep(EXIT_CLEAR_SLEEP_S)

    logger.error("Failed to submit CLOSE order for %s after %d attempts", symbol, EXIT_CLEAR_RETRIES)
    return None


# ---------------------------------------------------------
# MAIN EXECUTION â€“ PER SIGNAL
# ---------------------------------------------------------
def create_order_from_signal(symbol: str, side: str, strength: float, source: str, signal_id: int):
    side = side.lower()
    symbol = symbol.upper()

    key = (symbol, side)
    now_ts = time.time()
    until = INSUFFICIENT_COOLDOWN.get(key)
    if until is not None and now_ts < until:
        logger.info("Skip %s %s: still in insufficient cooldown until %s",
                    symbol, side.upper(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)))
        return

    pos_qty = get_position_qty(trading_client, symbol)

    atr_val, last_price = compute_atr(symbol)
    if last_price is None:
        last_price = DEFAULT_ENTRY_PRICE

    if not ALLOW_SHORT:
        if side == "sell":
            if pos_qty > 0:
                close_qty = int(abs(round(pos_qty)))
                logger.info("CLOSE intent: %s SELL to close long qty=%s (shorting disabled)", symbol, close_qty)
                submit_close_order(symbol, OrderSide.SELL, close_qty, float(last_price))
            else:
                logger.info("Skip %s SELL: no long position and shorting disabled", symbol)
            return

        if side == "buy" and pos_qty < 0:
            close_qty = int(abs(round(pos_qty)))
            logger.info("CLOSE intent: %s BUY to close short qty=%s (shorting disabled)", symbol, close_qty)
            submit_close_order(symbol, OrderSide.BUY, close_qty, float(last_price))
            return

    if not cleanup_and_check(symbol, side):
        return

    if side == "buy":
        entry_price = float(last_price) * (1 + ATR_PCT)
    else:
        entry_price = float(last_price) * (1 - ATR_PCT)
    entry_price = round(entry_price, 2)

    qty = compute_order_qty(symbol, side, entry_price, float(atr_val or 0.0))
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
        o = trading_client.submit_order(req)
        logger.info("ENTRY submitted: %s %s qty=%s @%.2f | id=%s status=%s | signal_id=%s",
                    symbol, side.upper(), qty, entry_price, str(o.id), str(o.status), signal_id)

    except APIError as e:
        msg = str(e)
        if ("insufficient buying power" in msg.lower()) or ("insufficient qty" in msg.lower()) or ("40310000" in msg):
            INSUFFICIENT_COOLDOWN[key] = time.time() + 120
            logger.warning("Insufficient funds/qty for %s %s, cooldown set: %s", symbol, side.upper(), msg)
            return
        raise


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    logger.info(
        "signal_executor starting | MIN_STRENGTH=%s | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ATR_PCT=%.4f | ALLOW_SHORT=%s | CLOSE_WITH_MARKET=%s | "
        "ENABLE_DAILY_RISK_GUARD=%s | MAX_DAILY_LOSS_USD=%.2f | MAX_DRAWDOWN_PCT=%.2f",
        MIN_STRENGTH,
        SYMBOLS,
        PORTFOLIO_ID,
        POLL_SECONDS,
        ATR_PCT,
        ALLOW_SHORT,
        CLOSE_WITH_MARKET,
        ENABLE_DAILY_RISK_GUARD,
        MAX_DAILY_LOSS_USD,
        MAX_DRAWDOWN_PCT,
    )

    while True:
        try:
            if not check_daily_risk_guard():
                logger.info("Daily risk guard active â€“ skipping signal execution this loop")
                time.sleep(POLL_SECONDS)
                continue

            signals = fetch_new_signals()
            selected = select_signals(signals)

            if not selected:
                logger.info("no new signals to execute")
            else:
                logger.info("Executing %d selected signal(s)", len(selected))

            for sig in selected:
                sym = str(sig["symbol"]).upper()
                side = str(sig["side"]).lower()
                strength = float(sig["strength"])
                source = str(sig.get("source") or "")
                sid = int(sig["id"])

                logger.info("EXEC: %s %s | strength=%.4f | source=%s | signal_id=%s",
                            sym, side.upper(), strength, source, sid)

                try:
                    create_order_from_signal(sym, side, strength, source, sid)
                finally:
                    mark_signal_processed(sig)

        except Exception as e:
            logger.exception("signal_executor loop error: %s", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

