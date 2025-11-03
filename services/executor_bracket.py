# services/executor_bracket.py
import os, sys, time, logging, math
from datetime import datetime, timezone
import psycopg2
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest

from tools.quotes import get_bid_ask_mid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor_bracket")

# ---- ENV / params ----
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

# Risk / ATR
RISK_PER_TRADE_USD = float(os.getenv("RISK_PER_TRADE_USD", "50"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))

# Signal gating
DEFAULT_SINCE_MIN = int(os.getenv("SINCE_MIN", "20"))
DEFAULT_MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.50"))

# Trading switches
FRACTIONAL = os.getenv("ALLOW_FRACTIONAL", "0") == "1"
LONG_ONLY = os.getenv("LONG_ONLY", "1") == "1"

# Market hours gate
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))

# Spread guard
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))      # %
MAX_SPREAD_ABS = float(os.getenv("MAX_SPREAD_ABS", "0.06"))      # $
QUOTE_PRICE_SLIPPAGE = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

# Misc
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "")
ACCOUNT_FALLBACK_TO_CASH = os.getenv("ACCOUNT_FALLBACK_TO_CASH", "1") == "1"


# ---- Alpaca client helpers ----
def _trading_client() -> TradingClient:
    return TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

def _get_account():
    try:
        return _trading_client().get_account()
    except Exception as e:
        log.warning("get_account failed: %s", e)
        return None

def _get_buying_power() -> float:
    """Use Alpaca buying_power; if it's '0' on paper for any reason, optionally fall back to cash."""
    acct = _get_account()
    if not acct:
        return 0.0
    try:
        bp = float(getattr(acct, "buying_power", 0) or 0)
        if bp <= 0 and ACCOUNT_FALLBACK_TO_CASH:
            cash = float(getattr(acct, "cash", 0) or 0)
            log.warning("buying_power reported 0; falling back to cash=%.2f", cash)
            return max(0.0, cash)
        return max(0.0, bp)
    except Exception:
        return 0.0

def _market_open() -> bool:
    try:
        clk = _trading_client().get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return False


# ---- DB / signals ----
def _conn():
    return psycopg2.connect(DB_URL)

def _fetch_signals(since_min: int, min_strength: float) -> pd.DataFrame:
    sql = """
        SELECT s.created_at, s.symbol, s.side, COALESCE(s.scaled_strength, s.strength) AS strength,
               s.atr, s.px
        FROM signals s
        WHERE s.created_at >= NOW() - INTERVAL %s
          AND COALESCE(s.scaled_strength, s.strength) >= %s
        ORDER BY s.created_at DESC
    """
    with _conn() as conn:
        df = pd.read_sql(sql, conn, params=(f"{since_min} minutes", min_strength))
    return df

def _dedupe_latest_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("created_at").groupby("symbol", as_index=False).tail(1)


# ---- Sizing / risk ----
def _qty_from_risk(price: float, atr: float) -> int:
    """Integer shares sized by risk and hard-capped by available buying power."""
    risk_per_trade = RISK_PER_TRADE_USD
    sl_dist = max(0.01, atr * SL_ATR_MULT)
    qty_risk = max(1, int(risk_per_trade / max(sl_dist, 1e-6)))  # at least 1 share if we have BP

    bp = _get_buying_power()
    max_bp_shares = int(max(0, bp) // max(price, 0.01))

    if max_bp_shares <= 0:
        return 0

    qty = min(qty_risk, max_bp_shares)
    if qty < qty_risk:
        log.info("qty capped by BP: %s -> %s (bp=%.2f, px=%.2f)", qty_risk, qty, bp, price)
    return max(0, qty)


# ---- Order submit ----
def _submit_bracket(symbol: str, side_str: str, limit_px: float, tp_px: float, sl_px: float, qty: int):
    """Submit a regular-hours bracket order with explicit extended_hours=False."""
    side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
    if qty <= 0:
        raise ValueError(f"qty must be > 0 for {symbol} {side_str}")

    order = LimitOrderRequest(
        symbol=symbol,
        side=side,
        type=OrderType.LIMIT,
        limit_price=round(limit_px, 2),
        time_in_force=TimeInForce.DAY,
        qty=qty,
        take_profit=TakeProfitRequest(limit_price=round(tp_px, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl_px, 2)),
        extended_hours=False,
    )
    cli = _trading_client()
    return cli.submit_order(order)


# ---- Main ----
def main():
    # CLI flags
    since_min = DEFAULT_SINCE_MIN
    min_strength = DEFAULT_MIN_STRENGTH
    for i, a in enumerate(sys.argv):
        if a == "--since-min" and i + 1 < len(sys.argv):
            try: since_min = int(sys.argv[i + 1])
            except: pass
        if a == "--min-strength" and i + 1 < len(sys.argv):
            try: min_strength = float(sys.argv[i + 1])
            except: pass

    # BP and switches display
    bp = _get_buying_power()
    acct = _get_account()
    fractional = FRACTIONAL
    long_only = LONG_ONLY
    log.info(
        "executor_bracket | since-min=%d min_strength=%.2f | buying_power=%.2f | fractional=%s long_only=%s",
        since_min, min_strength, bp, fractional, long_only
    )

    # Market-hours & min BP gates
    if not ALLOW_AFTER_HOURS and not _market_open():
        log.info("market is closed and ALLOW_AFTER_HOURS=0 -> skip this pass")
        return
    if bp < MIN_ACCOUNT_BP_USD:
        log.info("buying_power %.2f < MIN_ACCOUNT_BP_USD %.2f -> skip", bp, MIN_ACCOUNT_BP_USD)
        return

    # Load candidates
    df = _dedupe_latest_by_symbol(_fetch_signals(since_min, min_strength))
    if df.empty:
        log.info("no qualifying signals in last %d min (>= %.2f)", since_min, min_strength)
        return

    for _, row in df.iterrows():
        sym = str(row["symbol"])
        side = str(row["side"]).lower()  # 'buy' or 'sell'
        strength = float(row["strength"])
        atr = float(row["atr"] or 0.0)
        px = float(row["px"] or 0.0)

        if long_only and side == "sell":
            log.info("%s: LONG_ONLY=1 -> skip opening short", sym)
            continue

        # --- quotes + spread guard ---
        quote = get_bid_ask_mid(sym)
        if quote:
            bid, ask, mid = quote
            spread_abs = max(0.0, ask - bid)
            spread_pct = (spread_abs / mid) * 100.0 if mid > 0 else 999.0

            if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
                log.info(
                    "%s: skip due to wide spread (bid=%.4f ask=%.4f mid=%.4f abs=%.4f pct=%.3f%% > limits)",
                    sym, bid, ask, mid, spread_abs, spread_pct
                )
                continue

            slip = QUOTE_PRICE_SLIPPAGE
            if side == "buy":
                limit_px = min(ask + slip, max(ask, px))
            else:
                limit_px = max(bid - slip, min(bid, px if px else bid))
        else:
            # fallback
            limit_px = round(px, 2)

        # --- compute tp/sl off entry ref px (use px as analytics anchor) ---
        tp_px = px + TP_ATR_MULT * atr if side == "buy" else px - TP_ATR_MULT * atr
        sl_px = px - SL_ATR_MULT * atr if side == "buy" else px + SL_ATR_MULT * atr

        # --- size ---
        qty = _qty_from_risk(price=px if px > 0 else limit_px, atr=atr)
        if qty <= 0:
            log.info("%s: no buying power for qty; skip", sym)
            continue

        # Safety: never submit fractional if disabled
        if fractional:
            # (If you decide to use notional later, add it explicitly here; for now we use integer shares.)
            pass

        # Submit
        try:
            o = _submit_bracket(sym, side, limit_px, tp_px, sl_px, qty)
            log.info(
                "submitted %s %s qty=%s px=%.2f tp=%.2f sl=%.2f id=%s",
                sym, side, qty, float(getattr(o, "limit_price", limit_px)),
                float(getattr(getattr(o, "take_profit", None), "limit_price", tp_px)),
                float(getattr(getattr(o, "stop_loss", None), "stop_price", sl_px)),
                getattr(o, "id", "n/a"),
            )
        except Exception as e:
            log.warning("submit failed %s %s: %s", sym, side, e)
            continue


if __name__ == "__main__":
    main()
