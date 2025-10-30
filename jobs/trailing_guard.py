# jobs/trailing_guard.py
"""
Trailing-ATR stop manager.
- Monitors open positions.
- When unrealized move ≥ TRAIL_TRIGGER_ATR * ATR, lift/tighten the stop to a
  dynamic ATR-based level (trailing).
- Works with Alpaca bracket child stop orders by attempting an in-place replace.
- Fails safe: if replace is unsupported or fails, it logs and skips.

Tunable envs:
  TRAIL_ENABLE=1
  TRAIL_TRIGGER_ATR=0.5        # when profit reaches 0.5*ATR, start trailing
  TRAIL_MULT_ATR=1.0           # keep stop ~1.0*ATR behind current price
  TRAIL_PERIOD=14
  TRAIL_LOOKBACK_DAYS=30
  TRAIL_NEVER_BELOW_ENTRY=1    # for longs: never drop stop below entry; shorts: never above entry
"""

import os, logging
from typing import Optional, List, Dict
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.requests import ReplaceOrderRequest

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trailing_guard")

# --- env ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY","")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET","")
APCA_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

TRAIL_ENABLE = os.getenv("TRAIL_ENABLE","1") == "1"
TRAIL_TRIGGER_ATR = float(os.getenv("TRAIL_TRIGGER_ATR","0.5"))
TRAIL_MULT_ATR = float(os.getenv("TRAIL_MULT_ATR","1.0"))
TRAIL_PERIOD = int(os.getenv("TRAIL_PERIOD","14"))
TRAIL_LOOKBACK_DAYS = int(os.getenv("TRAIL_LOOKBACK_DAYS","30"))
TRAIL_NEVER_BELOW_ENTRY = os.getenv("TRAIL_NEVER_BELOW_ENTRY","1") == "1"

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

# --- utils ---
def _client() -> TradingClient:
    return TradingClient(ALPACA_KEY, ALPACA_SEC, paper="paper" in APCA_BASE)

def _latest_price(sym: str) -> Optional[float]:
    try:
        p = yf.Ticker(sym).fast_info.last_price
        return float(p) if p is not None else None
    except Exception:
        return None

def _download_5m(sym: str, days: int) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(sym, period=f"{days}d", interval="5m", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None
        return df.rename(columns=str.lower)
    except Exception as e:
        log.warning("bars fail %s: %s", sym, e)
        return None

def _atr(h, l, c, n=14):
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n).mean()

def _compute_atr(sym: str) -> Optional[float]:
    bars = _download_5m(sym, TRAIL_LOOKBACK_DAYS)
    if bars is None or bars.empty:
        return None
    return float(_atr(bars["high"], bars["low"], bars["close"], TRAIL_PERIOD).iloc[-1])

def _find_stop_child_order(orders, symbol: str):
    """
    Heuristic: find an OPEN stop(-limit) child order for this symbol.
    Alpaca returns a list of orders; stop legs typically have 'type' in {'stop','stop_limit'}.
    """
    for o in orders:
        try:
            if (o.symbol == symbol) and (str(getattr(o, "type", "")).lower().startswith("stop")):
                return o
        except Exception:
            continue
    return None

def _replace_stop(cli: TradingClient, order_id: str, new_stop: float) -> bool:
    """
    Try to replace the existing stop price. If not supported, returns False.
    """
    try:
        req = ReplaceOrderRequest(stop_price=round(float(new_stop), 2))
        cli.replace_order_by_id(order_id, req)
        return True
    except Exception as e:
        log.warning("replace stop failed %s: %s", order_id, e)
        return False

def _desired_trailing_stop(side_long: bool, price: float, atr: float, entry: float) -> float:
    if side_long:
        base = price - TRAIL_MULT_ATR * atr
        if TRAIL_NEVER_BELOW_ENTRY:
            base = max(base, entry)
        return base
    else:
        base = price + TRAIL_MULT_ATR * atr
        if TRAIL_NEVER_BELOW_ENTRY:
            base = min(base, entry)
        return base

def main():
    if not TRAIL_ENABLE:
        log.info("trailing_guard disabled; set TRAIL_ENABLE=1 to enable")
        return

    cli = _client()

    # fetch open positions
    try:
        positions = list(cli.get_all_positions())
    except Exception as e:
        log.warning("get_all_positions failed: %s", e)
        return

    if not positions:
        log.info("no open positions; nothing to trail")
        return

    # fetch open orders once (to locate stop legs)
    try:
        open_orders = list(cli.get_orders(status="open"))
    except Exception as e:
        log.warning("get_orders failed: %s", e)
        open_orders = []

    changed = 0
    for pos in positions:
        try:
            sym = pos.symbol
            side_long = (str(getattr(pos, "side", "long")).lower() == "long")
            entry = float(pos.avg_entry_price)
            qty = abs(float(pos.qty))
        except Exception as e:
            log.warning("bad position data: %s", e)
            continue

        price = _latest_price(sym)
        if price is None:
            log.info("%s: no price; skip trail", sym); continue

        atr = _compute_atr(sym)
        if atr is None:
            log.info("%s: no ATR; skip trail", sym); continue

        # Check trigger (unrealized move vs entry)
        move = (price - entry) if side_long else (entry - price)
        if move < TRAIL_TRIGGER_ATR * atr:
            # not yet eligible to trail
            continue

        # Find current stop child
        stop_child = _find_stop_child_order(open_orders, sym)
        if stop_child is None:
            log.info("%s: no stop child order open; skip", sym)
            continue

        # Compute desired trailing stop
        new_stop = _desired_trailing_stop(side_long, price, atr, entry)

        # If existing stop is already tighter, skip
        try:
            current_stop = float(getattr(stop_child, "stop_price", None) or getattr(stop_child, "limit_price", None) or 0.0)
        except Exception:
            current_stop = 0.0

        bump_needed = (new_stop > current_stop + 1e-6) if side_long else (new_stop < current_stop - 1e-6)
        if not bump_needed:
            continue

        ok = _replace_stop(cli, stop_child.id, new_stop)
        if ok:
            changed += 1
            log.info("trail %s | side=%s price=%.2f atr=%.3f entry=%.2f stop: %.2f -> %.2f",
                     sym, "LONG" if side_long else "SHORT", price, atr, entry, current_stop, new_stop)

    if changed:
        log.info("trailing_guard | tightened %d stop(s)", changed)
    else:
        log.info("trailing_guard | no changes")

if __name__ == "__main__":
    main()
