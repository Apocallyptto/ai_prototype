import os
import time
import json
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce

# ✅ MARKET GATE (A+B už máš, takto to len voláme)
from services.market_gate import should_trade_now


# ---------------- Logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("signal_executor")


# ---------------- Helpers ----------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    v = env_str(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


@dataclass
class PositionInfo:
    side: str   # "LONG" / "SHORT"
    qty: float  # absolute qty (positive)


@dataclass
class SignalRow:
    id: int
    created_at: str
    symbol: str
    side: str    # "BUY"/"SELL"
    strength: float
    source: str
    portfolio_id: int


# ---------------- Alpaca market data (simple REST) ----------------
def alpaca_headers(api_key: str, api_secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }


def get_latest_price(data_url: str, api_key: str, api_secret: str, symbol: str, timeout: float = 8.0) -> Optional[float]:
    # Try latest trade
    trade_url = f"{data_url.rstrip('/')}/v2/stocks/{symbol}/trades/latest"
    try:
        r = requests.get(trade_url, headers=alpaca_headers(api_key, api_secret), timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            trade = j.get("trade") or {}
            p = trade.get("p") or trade.get("price")
            if p is not None:
                return float(p)
    except Exception:
        pass

    # Fallback: latest quote (mid)
    quote_url = f"{data_url.rstrip('/')}/v2/stocks/{symbol}/quotes/latest"
    try:
        r = requests.get(quote_url, headers=alpaca_headers(api_key, api_secret), timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            q = j.get("quote") or {}
            ap = q.get("ap") or q.get("ask_price")
            bp = q.get("bp") or q.get("bid_price")
            if ap is not None and bp is not None:
                return (float(ap) + float(bp)) / 2.0
            if ap is not None:
                return float(ap)
            if bp is not None:
                return float(bp)
    except Exception:
        pass

    return None


# ---------------- DB helpers ----------------
def db_engine(db_url: str):
    return create_engine(db_url, pool_pre_ping=True, future=True)


def fetch_unprocessed_signals(engine, symbols: List[str], portfolio_id: int, min_strength: float, lookback_minutes: int = 30) -> List[SignalRow]:
    sql = text("""
        SELECT id, created_at, symbol, side, strength, source, portfolio_id
        FROM signals
        WHERE processed_at IS NULL
          AND created_at > now() - (:lookback_minutes || ' minutes')::interval
          AND strength >= :min_strength
          AND portfolio_id = (:pid)::int
          AND symbol = ANY(:symbols)
        ORDER BY created_at ASC
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {
            "symbols": symbols,
            "pid": int(portfolio_id),
            "min_strength": float(min_strength),
            "lookback_minutes": int(lookback_minutes),
        }).mappings().all()

    out: List[SignalRow] = []
    for r in rows:
        out.append(SignalRow(
            id=int(r["id"]),
            created_at=str(r["created_at"]),
            symbol=str(r["symbol"]).upper(),
            side=str(r["side"]).upper(),
            strength=float(r["strength"]),
            source=str(r.get("source") or "unknown"),
            portfolio_id=int(r.get("portfolio_id") or portfolio_id),
        ))
    return out


def mark_signal(engine, signal_id: int, status: str, note: str = "", alpaca_order_id: Optional[str] = None) -> None:
    sql = text("""
        UPDATE signals
        SET processed_at = now(),
            processed_status = :status,
            processed_note = :note,
            alpaca_order_id = COALESCE(:alpaca_order_id, alpaca_order_id)
        WHERE id = :id
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "id": int(signal_id),
            "status": status,
            "note": note[:4000],
            "alpaca_order_id": alpaca_order_id,
        })


# ---------------- Trading guards ----------------
def get_positions_map(tc: TradingClient) -> Dict[str, PositionInfo]:
    pos_map: Dict[str, PositionInfo] = {}
    for p in tc.get_all_positions():
        sym = str(getattr(p, "symbol", "")).upper()
        side = str(getattr(p, "side", "")).upper()  # PositionSide.LONG/SHORT -> string contains LONG/SHORT
        qty_raw = float(getattr(p, "qty", 0) or 0)
        if qty_raw == 0 or sym == "":
            continue
        if "SHORT" in side:
            pos_map[sym] = PositionInfo(side="SHORT", qty=abs(qty_raw))
        else:
            pos_map[sym] = PositionInfo(side="LONG", qty=abs(qty_raw))
    return pos_map


def get_open_orders_map(tc: TradingClient) -> Dict[str, List[object]]:
    try:
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True))
    except TypeError:
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))

    m: Dict[str, List[object]] = {}
    for o in orders or []:
        sym = str(getattr(o, "symbol", "")).upper()
        if sym == "":
            continue
        m.setdefault(sym, []).append(o)
    return m


def compute_qty(
    price: float,
    buying_power: float,
    atr_pct: float,
    sl_atr_mult: float,
    max_risk_usd: float,
    max_bp_pct: float,
    max_notional_usd: float,
    max_qty: int
) -> int:
    if price <= 0:
        return 0

    # Risk-based qty using ATR_PCT proxy: stop_distance = price * atr_pct * sl_mult
    stop_distance = max(price * max(atr_pct, 0.0001) * max(sl_atr_mult, 0.1), 0.01)
    qty_risk = math.floor(max_risk_usd / stop_distance) if max_risk_usd > 0 else 10**9

    qty_bp = math.floor((buying_power * max_bp_pct) / price) if max_bp_pct > 0 else 10**9
    qty_notional = math.floor(max_notional_usd / price) if max_notional_usd > 0 else 10**9
    qty_cap = max_qty if max_qty > 0 else 10**9

    qty = int(max(0, min(qty_risk, qty_bp, qty_notional, qty_cap)))
    return qty


def is_insufficient_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("40310000" in s) or ("insufficient" in s) or ("buying power" in s) or ("insufficient qty" in s)


# ---------------- Main loop ----------------
def main():
    # Env
    db_url = env_str("DB_URL", "postgresql+psycopg2://postgres:postgres@postgres:5432/trader")

    symbols = parse_symbols(env_str("SYMBOLS", "AAPL,MSFT,SPY"))
    portfolio_id = env_int("PORTFOLIO_ID", 1)

    # Accept both names (old/new)
    min_strength = env_float("MIN_STRENGTH", env_float("EXEC_MIN_STRENGTH", 0.60))
    poll_seconds = env_int("POLL_SECONDS", env_int("EXEC_POLL_SECONDS", 20))

    allow_short = env_bool("ALLOW_SHORT", False)
    long_only = env_bool("LONG_ONLY", False)

    atr_pct = env_float("ATR_PCT", 0.01)
    sl_atr_mult = env_float("SL_ATR_MULT", 1.0)
    tp_atr_mult = env_float("TP_ATR_MULT", 1.0)  # not used here directly, but logged

    max_risk = env_float("MAX_RISK_PER_TRADE_USD", 30.0)
    max_bp_pct = env_float("MAX_BP_PCT_PER_TRADE", 0.05)
    max_notional = env_float("MAX_NOTIONAL_PER_TRADE_USD", 1200.0)
    max_qty = env_int("MAX_QTY_PER_TRADE", 5)

    cooldown_seconds = env_int("INSUFFICIENT_COOLDOWN_SECONDS", 900)
    lookback_minutes = env_int("SIGNALS_LOOKBACK_MINUTES", 30)
    max_signals_per_loop = env_int("MAX_SIGNALS_PER_LOOP", 5)

    # ✅ market gate config
    stop_new_entries_min_before_close = env_int("STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE", 10)

    trading_mode = env_str("TRADING_MODE", "paper").lower()
    paper = (trading_mode != "live")

    api_key = env_str("ALPACA_API_KEY")
    api_secret = env_str("ALPACA_API_SECRET")
    data_url = env_str("ALPACA_DATA_URL", "https://data.alpaca.markets")

    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env")

    engine = db_engine(db_url)
    tc = TradingClient(api_key, api_secret, paper=paper)

    insuff_cooldown_until: Dict[str, datetime] = {}

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | ATR_PCT=%.4f | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_RISK=%.2f | MAX_BP_PCT=%.4f | MAX_NOTIONAL=%.2f | MAX_QTY=%s | "
        "TP_ATR_MULT=%.2f | SL_ATR_MULT=%.2f | STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE=%s",
        min_strength, symbols, portfolio_id, poll_seconds, atr_pct,
        allow_short, long_only, max_risk, max_bp_pct, max_notional, max_qty,
        tp_atr_mult, sl_atr_mult, stop_new_entries_min_before_close
    )

    while True:
        try:
            # ✅ MARKET GATE (hneď na začiatku loopu)
            ok, reason, clock = should_trade_now(stop_new_entries_min_before_close)
            if not ok:
                logger.info(
                    "market_gate | skip trading | reason=%s | is_open=%s | ts=%s | next_open=%s | next_close=%s",
                    reason,
                    clock.get("is_open"),
                    clock.get("timestamp"),
                    clock.get("next_open"),
                    clock.get("next_close"),
                )
                time.sleep(poll_seconds)
                continue

            # Refresh account/positions/open orders each loop (simple + stable)
            account = tc.get_account()
            buying_power = float(getattr(account, "buying_power", 0) or 0)

            pos_map = get_positions_map(tc)
            open_orders_map = get_open_orders_map(tc)

            # Fetch signals
            rows = fetch_unprocessed_signals(engine, symbols, portfolio_id, min_strength, lookback_minutes=lookback_minutes)
            logger.info("fetch_new_signals | fetched %s rows", len(rows))

            if not rows:
                time.sleep(poll_seconds)
                continue

            # Deduplicate: keep only earliest per (symbol, side)
            seen = set()
            selected: List[SignalRow] = []
            for r in rows:
                key = (r.symbol, r.side)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(r)
                if len(selected) >= max_signals_per_loop:
                    break

            logger.info("select_signals | fetched=%s | selected=%s | unique_symbol_side=%s",
                        len(rows), len(selected), len(seen))

            for s in selected:
                sym = s.symbol.upper()
                side = s.side.upper()

                # Cooldown guard
                cd_until = insuff_cooldown_until.get(sym)
                if cd_until and now_utc() < cd_until:
                    mark_signal(engine, s.id, "skipped", f"cooldown_until={cd_until.isoformat()}")
                    continue

                # Strategy guards
                if long_only and side == "SELL":
                    mark_signal(engine, s.id, "skipped", "LONG_ONLY=1")
                    continue
                if side == "SELL" and not allow_short:
                    mark_signal(engine, s.id, "skipped", "ALLOW_SHORT=0")
                    continue

                # Open-order guard
                if open_orders_map.get(sym):
                    mark_signal(engine, s.id, "skipped", f"open_orders_present={len(open_orders_map[sym])}")
                    continue

                # Position guard (no pyramiding)
                p = pos_map.get(sym)
                if p:
                    if p.side == "LONG" and side == "BUY":
                        mark_signal(engine, s.id, "skipped", "already_long_no_pyramiding")
                        continue
                    if p.side == "SHORT" and side == "SELL":
                        mark_signal(engine, s.id, "skipped", "already_short_no_pyramiding")
                        continue

                    # Flip: close first, do not open opposite entry in same loop
                    if p.side == "LONG" and side == "SELL":
                        close_req = MarketOrderRequest(symbol=sym, qty=p.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                        try:
                            o = tc.submit_order(close_req)
                            mark_signal(engine, s.id, "skipped", f"flip_close_submitted LONG->SELL qty={p.qty}", str(getattr(o, "id", "")))
                            logger.info("FLIP_CLOSE submitted: %s SELL qty=%s | id=%s", sym, p.qty, getattr(o, "id", None))
                        except Exception as e:
                            mark_signal(engine, s.id, "error", f"flip_close_failed {e}")
                            logger.exception("flip_close_failed: %s", e)
                        continue

                    if p.side == "SHORT" and side == "BUY":
                        close_req = MarketOrderRequest(symbol=sym, qty=p.qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                        try:
                            o = tc.submit_order(close_req)
                            mark_signal(engine, s.id, "skipped", f"flip_close_submitted SHORT->BUY qty={p.qty}", str(getattr(o, "id", "")))
                            logger.info("FLIP_CLOSE submitted: %s BUY qty=%s | id=%s", sym, p.qty, getattr(o, "id", None))
                        except Exception as e:
                            mark_signal(engine, s.id, "error", f"flip_close_failed {e}")
                            logger.exception("flip_close_failed: %s", e)
                        continue

                # Compute price + qty
                price = get_latest_price(data_url, api_key, api_secret, sym)
                if not price or price <= 0:
                    mark_signal(engine, s.id, "error", "no_price_from_alpaca_data")
                    continue

                qty = compute_qty(
                    price=price,
                    buying_power=buying_power,
                    atr_pct=atr_pct,
                    sl_atr_mult=sl_atr_mult,
                    max_risk_usd=max_risk,
                    max_bp_pct=max_bp_pct,
                    max_notional_usd=max_notional,
                    max_qty=max_qty,
                )

                if qty < 1:
                    mark_signal(engine, s.id, "skipped", "computed_qty<1")
                    continue

                # Submit limit at near-last price (small bias)
                if side == "BUY":
                    order_side = OrderSide.BUY
                    limit_price = round(price * 1.0005, 2)
                else:
                    order_side = OrderSide.SELL
                    limit_price = round(price * 0.9995, 2)

                req = LimitOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )

                try:
                    o = tc.submit_order(req)
                    oid = str(getattr(o, "id", "")) if o else ""
                    mark_signal(engine, s.id, "submitted", f"entry_submitted qty={qty} limit={limit_price}", oid)
                    logger.info("ENTRY submitted: %s %s qty=%s @%.2f | id=%s | signal_id=%s",
                                sym, side, qty, limit_price, oid, s.id)
                except Exception as e:
                    if is_insufficient_error(e):
                        until = now_utc() + timedelta(seconds=cooldown_seconds)
                        insuff_cooldown_until[sym] = until
                        mark_signal(engine, s.id, "skipped", f"insufficient_set_cooldown_until={until.isoformat()} | {e}")
                        logger.warning("Insufficient for %s %s: cooldown set until %s | err=%s", sym, side, until, e)
                    else:
                        mark_signal(engine, s.id, "error", f"submit_failed {e}")
                        logger.exception("submit_failed: %s", e)

                # tiny spacing to be nice to APIs
                time.sleep(0.15)

        except Exception as e:
            logger.exception("signal_executor loop error: %s", e)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
