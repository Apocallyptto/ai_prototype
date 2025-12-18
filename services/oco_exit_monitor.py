import os
import time
import logging
from typing import Optional, Dict

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# ------------------------------------------------------------
# Config (env)
# ------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("oco_exit_monitor")

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
EXIT_COOLDOWN_SECONDS = int(os.getenv("EXIT_COOLDOWN_SECONDS", "60"))

MIN_QTY = float(os.getenv("EXIT_MIN_QTY", "1"))
MIN_NOTIONAL = float(os.getenv("EXIT_MIN_NOTIONAL", "25"))

ATR_PCT = float(os.getenv("ATR_PCT", "0.01"))      # default 1% of price
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))

ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") in ("1", "true", "True", "yes", "YES")
FORCE_FIX_EXITS = os.getenv("FORCE_FIX_EXITS", "0") in ("1", "true", "True", "yes", "YES")
CANCEL_EXITS_IF_FLAT = os.getenv("CANCEL_EXITS_IF_FLAT", "1") in ("1", "true", "True", "yes", "YES")

EXIT_CID_PREFIX = "EXIT-OCO"

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
PAPER = TRADING_MODE != "live"


def _latest_mid(dc: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return None


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _is_our_exit_order(o, symbol: str) -> bool:
    cid = getattr(o, "client_order_id", None) or ""
    if not cid.startswith(f"{EXIT_CID_PREFIX}-{symbol}-"):
        return False
    oc = getattr(o, "order_class", None)
    if oc is None:
        return False
    return str(oc).lower().endswith("oco")


def _get_open_orders(tr: TradingClient, symbol: Optional[str] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
    if symbol:
        req.symbols = [symbol]
    return tr.get_orders(req)


def _cancel_orders(tr: TradingClient, orders) -> int:
    n = 0
    for o in orders:
        try:
            tr.cancel_order_by_id(o.id)
            n += 1
        except Exception as e:
            log.warning("cancel failed: %s id=%s err=%s", getattr(o, "symbol", "?"), getattr(o, "id", "?"), e)
    return n


def _compute_tp_sl(mid: float, is_long: bool):
    atr = max(0.01, mid * ATR_PCT)
    if is_long:
        tp = mid + atr * TP_ATR_MULT
        sl = mid - atr * SL_ATR_MULT
    else:
        tp = mid - atr * TP_ATR_MULT
        sl = mid + atr * SL_ATR_MULT
    return atr, round(tp, 2), round(sl, 2)


def _submit_exit_oco(tr: TradingClient, dc: StockHistoricalDataClient, symbol: str, qty_abs: float, is_long: bool):
    mid = _latest_mid(dc, symbol)
    if not mid:
        log.warning("%s: no quote mid available; skip", symbol)
        return None

    notional = mid * qty_abs
    if qty_abs < MIN_QTY or notional < MIN_NOTIONAL:
        log.info("%s: skip exit (qty=%.4f notional=%.2f < mins)", symbol, qty_abs, notional)
        return None

    atr, tp_price, sl_price = _compute_tp_sl(mid, is_long)
    exit_side = OrderSide.SELL if is_long else OrderSide.BUY
    cid = f"{EXIT_CID_PREFIX}-{symbol}-{int(time.time())}"

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty_abs,
        side=exit_side,
        time_in_force=TimeInForce.DAY,
        limit_price=tp_price,
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=sl_price),
        client_order_id=cid,
        extended_hours=ALLOW_AFTER_HOURS,
    )

    log.info(
        "%s: submit EXIT OCO | side=%s qty=%.4f mid=%.4f atr=%.4f tp=%.2f sl=%.2f cid=%s",
        symbol, exit_side, qty_abs, mid, atr, tp_price, sl_price, cid
    )
    return tr.submit_order(req)


def run_once(tr: TradingClient, dc: StockHistoricalDataClient, last_action_ts: Dict[str, float]):
    try:
        positions = tr.get_all_positions()
    except Exception as e:
        log.warning("get_all_positions failed: %s", e)
        return

    pos_by_symbol = {p.symbol.upper(): p for p in positions if p.symbol and p.symbol.upper() in SYMBOLS}

    if not pos_by_symbol:
        log.info("no positions")
        if CANCEL_EXITS_IF_FLAT:
            try:
                open_orders = _get_open_orders(tr)
                stale = [o for o in open_orders if _is_our_exit_order(o, getattr(o, "symbol", "").upper())]
                if stale:
                    n = _cancel_orders(tr, stale)
                    log.warning("flat => canceled %d stale EXIT-OCO order(s)", n)
            except Exception as e:
                log.warning("flat cleanup failed: %s", e)
        return

    now = time.time()
    for symbol, p in pos_by_symbol.items():
        qty = _safe_float(getattr(p, "qty", 0.0))
        qty_abs = abs(qty)
        is_long = qty > 0

        if now - last_action_ts.get(symbol, 0.0) < EXIT_COOLDOWN_SECONDS:
            continue

        try:
            open_sym = _get_open_orders(tr, symbol=symbol)
        except Exception as e:
            log.warning("%s: get_orders failed: %s", symbol, e)
            continue

        ours = [o for o in open_sym if _is_our_exit_order(o, symbol)]

        if ours and not FORCE_FIX_EXITS:
            log.info("%s: exits already present (%d)", symbol, len(ours))
            continue

        if ours and FORCE_FIX_EXITS:
            n = _cancel_orders(tr, ours)
            log.warning("%s: FORCE_FIX_EXITS => canceled %d existing exit order(s)", symbol, n)

        try:
            _submit_exit_oco(tr, dc, symbol, qty_abs, is_long)
            last_action_ts[symbol] = now
        except Exception as e:
            log.warning("%s: submit exit OCO failed: %s", symbol, e)


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    tr = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
    dc = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    log.info(
        "oco_exit_monitor starting | symbols=%s poll=%ss cooldown=%ss atr_pct=%.4f tp_mult=%.2f sl_mult=%.2f paper=%s allow_ah=%s force_fix=%s cancel_exits_if_flat=%s",
        SYMBOLS, POLL_SECONDS, EXIT_COOLDOWN_SECONDS, ATR_PCT, TP_ATR_MULT, SL_ATR_MULT, PAPER, ALLOW_AFTER_HOURS, FORCE_FIX_EXITS, CANCEL_EXITS_IF_FLAT
    )

    last_action_ts: Dict[str, float] = {}
    while True:
        try:
            run_once(tr, dc, last_action_ts)
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
