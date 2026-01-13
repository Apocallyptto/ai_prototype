# services/signal_executor.py
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest

logger = logging.getLogger("signal_executor")


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Signal:
    id: int
    created_at: datetime
    symbol: str
    side: str
    strength: float
    source: str
    portfolio_id: int


@dataclass
class ExecResult:
    status: str  # submitted / skipped / error
    note: str
    alpaca_order_id: Optional[str] = None


def parse_symbols(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip().upper() for p in s.split(",")]
    return [p for p in parts if p]


def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "postgres"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "trader"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
    )


def fetch_new_signals(
    conn,
    portfolio_id: int,
    symbols: List[str],
    min_strength: float,
    limit: int = 200,
) -> List[Signal]:
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if symbols:
            cur.execute(
                """
                select id, created_at, symbol, side, strength, source, portfolio_id
                from signals
                where processed_at is null
                  and portfolio_id = %s
                  and symbol = any(%s)
                  and strength >= %s
                order by created_at asc
                limit %s
                """,
                (portfolio_id, symbols, min_strength, limit),
            )
        else:
            cur.execute(
                """
                select id, created_at, symbol, side, strength, source, portfolio_id
                from signals
                where processed_at is null
                  and portfolio_id = %s
                  and strength >= %s
                order by created_at asc
                limit %s
                """,
                (portfolio_id, min_strength, limit),
            )
        rows = cur.fetchall() or []
        out: List[Signal] = []
        for r in rows:
            out.append(
                Signal(
                    id=int(r["id"]),
                    created_at=r["created_at"],
                    symbol=str(r["symbol"]).upper(),
                    side=str(r["side"]).lower(),
                    strength=float(r["strength"]),
                    source=str(r["source"]),
                    portfolio_id=int(r["portfolio_id"]),
                )
            )
        logger.info("fetch_new_signals | fetched %d rows", len(out))
        return out


def mark_processed(conn, signal_id: int, status: str, note: str, alpaca_order_id: Optional[str] = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            update signals
            set processed_at = now(),
                processed_status = %s,
                processed_note = %s,
                alpaca_order_id = %s
            where id = %s
            """,
            (status, note, alpaca_order_id, signal_id),
        )
    conn.commit()


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = env_bool("ALPACA_PAPER", True)
    return TradingClient(key, secret, paper=paper)


def side_to_order_side(side: str) -> OrderSide:
    return OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_last_price_fallback(tc: TradingClient, symbol: str) -> Optional[float]:
    # Prefer recent trade price, fallback to None
    # Note: trading client has get_asset etc; for price we'd typically use data client.
    # Here we just return None and rely on caller's price logic.
    return None


def cancel_opposite_open_orders(tc: TradingClient, symbol: str, desired_side: OrderSide) -> int:
    """
    Cancel open opposite-side orders for same symbol to avoid wash-trade issues.
    """
    canceled = 0
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = tc.get_orders(req) or []
        for o in orders:
            if getattr(o, "symbol", None) != symbol:
                continue
            if getattr(o, "side", None) == desired_side:
                continue
            try:
                tc.cancel_order_by_id(o.id)
                canceled += 1
            except Exception as e:
                logger.warning("cancel_opposite_open_orders failed | %s | %s", o.id, e)
    except Exception as e:
        logger.warning("cancel_opposite_open_orders list failed | %s", e)
    return canceled


def dedupe_recent_alpaca(
    tc: TradingClient,
    symbol: str,
    side: OrderSide,
    dedupe_minutes: int,
) -> bool:
    """
    Returns True if there is a recent order for (symbol, side) within dedupe window.
    """
    try:
        now = utcnow()
        since = now - timedelta(minutes=dedupe_minutes)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200)
        orders = tc.get_orders(req) or []
        for o in orders:
            if getattr(o, "symbol", None) != symbol:
                continue
            if getattr(o, "side", None) != side:
                continue
            submitted_at = getattr(o, "submitted_at", None)
            if submitted_at is None:
                continue
            try:
                # submitted_at is tz-aware
                if submitted_at >= since:
                    return True
            except Exception:
                pass
        return False
    except Exception:
        return False


def fetch_open_positions(tc: TradingClient) -> Dict[str, Any]:
    positions = tc.get_all_positions() or []
    out: Dict[str, Any] = {}
    for p in positions:
        out[str(p.symbol).upper()] = p
    return out


def count_open_orders(tc: TradingClient) -> int:
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = tc.get_orders(req) or []
        return len(orders)
    except Exception:
        return 0


def daily_pnl_guard(tc: TradingClient, daily_loss_stop_pct: float, max_daily_loss_usd: float) -> Tuple[bool, str]:
    """
    Very simple daily guard: compares equity vs last_equity_snapshot if exists in env.
    (In production you'd read today's start equity from DB / snapshots.)
    For now: if env START_EQUITY is set, compare.
    """
    start_eq = os.getenv("START_EQUITY")
    if not start_eq:
        return True, "no_start_equity"
    try:
        start = float(start_eq)
        a = tc.get_account()
        eq = float(a.equity)
        dd = start - eq
        dd_pct = (dd / start) * 100.0 if start > 0 else 0.0
        if dd_pct >= daily_loss_stop_pct:
            return False, f"daily_loss_stop_pct_hit dd_pct={dd_pct:.2f}%"
        if dd >= max_daily_loss_usd:
            return False, f"max_daily_loss_usd_hit dd={dd:.2f}"
        return True, f"ok dd={dd:.2f} dd_pct={dd_pct:.2f}%"
    except Exception as e:
        return True, f"guard_error_ignored:{type(e).__name__}"


def symbol_cooldown_ok(conn, portfolio_id: int, symbol: str, cooldown_seconds: int) -> Tuple[bool, str]:
    if cooldown_seconds <= 0:
        return True, "cooldown_disabled"
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            select max(processed_at) as last_processed
            from signals
            where portfolio_id = %s
              and symbol = %s
              and processed_at is not null
            """,
            (portfolio_id, symbol),
        )
        r = cur.fetchone()
        last = r["last_processed"] if r else None
        if not last:
            return True, "cooldown_ok_no_history"
        now = utcnow()
        try:
            delta = (now - last).total_seconds()
        except Exception:
            # last might be naive; assume utc
            last = last.replace(tzinfo=timezone.utc)
            delta = (now - last).total_seconds()
        if delta >= cooldown_seconds:
            return True, f"cooldown_ok_{int(delta)}s"
        return False, f"symbol_cooldown_{cooldown_seconds}s"


def pick_signal(conn, signal_id: int, ttl_seconds: int) -> bool:
    """
    Mark as picked to avoid multiple executors grabbing same signal.
    We set processed_status='picked' but keep processed_at null.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            update signals
            set processed_status = 'picked',
                processed_note = 'picked'
            where id = %s
              and processed_at is null
              and (processed_status is null)
            """,
            (signal_id,),
        )
        ok = cur.rowcount == 1
    conn.commit()
    return ok


def release_stale_picks(conn, ttl_seconds: int) -> int:
    """
    If some signals stuck in picked for too long, release them.
    (Note: we don't store picked_at, so we use created_at as proxy,
     but it's still useful.)
    """
    if ttl_seconds <= 0:
        return 0
    cutoff = utcnow() - timedelta(seconds=ttl_seconds)
    with conn.cursor() as cur:
        cur.execute(
            """
            update signals
            set processed_status = null,
                processed_note = 'pick_ttl_released'
            where processed_at is null
              and processed_status = 'picked'
              and created_at < %s
            """,
            (cutoff,),
        )
        n = cur.rowcount
    conn.commit()
    return n


def select_signals(signals: List[Signal]) -> List[Signal]:
    """
    Keep order, but reduce duplicates by (symbol, side) taking first occurrence.
    """
    seen = set()
    out = []
    for s in signals:
        key = (s.symbol, s.side)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    logger.info(
        "select_signals | fetched=%d | selected=%d | unique_symbol_side=%d",
        len(signals),
        len(out),
        len(seen),
    )
    return out


def compute_order_qty(symbol: str, max_qty: int) -> int:
    return max(1, int(max_qty))


def compute_limit_price(tc: TradingClient, symbol: str, side: OrderSide) -> Optional[float]:
    # Placeholder: in your project you likely compute via latest bar/quote.
    # We'll return None to let router decide, but this file uses limit orders,
    # so caller should provide a value elsewhere. For now we use a minimal hack:
    return get_last_price_fallback(tc, symbol)


def submit_limit_order(
    tc: TradingClient,
    symbol: str,
    side: OrderSide,
    qty: int,
    limit_price: float,
    tif: TimeInForce = TimeInForce.DAY,
) -> str:
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type=OrderType.LIMIT,
        time_in_force=tif,
        limit_price=limit_price,
    )
    o = tc.submit_order(req)
    return str(o.id)


def can_short(side: str, allow_short: bool) -> bool:
    if side.lower() == "sell" and not allow_short:
        return False
    return True


def risk_guard_ok(
    tc: TradingClient,
    max_open_positions: int,
    max_open_orders: int,
    enable_daily_risk_guard: bool,
    daily_loss_stop_pct: float,
    max_daily_loss_usd: float,
) -> Tuple[bool, str]:
    """
    Hard block when:
      - too many open positions
      - too many open orders
      - daily loss guard triggers
    """
    positions = tc.get_all_positions() or []
    open_orders = []
    try:
        open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)) or []
    except Exception:
        open_orders = []

    if max_open_positions > 0 and len(positions) >= max_open_positions:
        return False, f"max_open_positions_reached:{len(positions)}/{max_open_positions}"
    if max_open_orders > 0 and len(open_orders) >= max_open_orders:
        return False, f"max_open_orders_reached:{len(open_orders)}/{max_open_orders}"

    if enable_daily_risk_guard:
        ok, why = daily_pnl_guard(tc, daily_loss_stop_pct, max_daily_loss_usd)
        if not ok:
            return False, why

    return True, "ok"


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    symbols = parse_symbols(os.getenv("SYMBOLS", "AAPL,MSFT,SPY,NVDA,AMD"))
    portfolio_id = env_int("PORTFOLIO_ID", 1)
    min_strength = env_float("MIN_STRENGTH", 0.60)
    poll_seconds = env_int("POLL_SECONDS", 20)

    allow_short = env_bool("ALLOW_SHORT", True)
    long_only = env_bool("LONG_ONLY", False)

    max_notional = env_float("MAX_NOTIONAL", 1200.0)
    max_qty = env_int("MAX_QTY", 5)

    dedupe_minutes = env_int("ALPACA_DEDUPE_MINUTES", 10)
    cancel_opposite = env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    max_open_positions = env_int("MAX_OPEN_POSITIONS", 3)
    max_open_orders = env_int("MAX_OPEN_ORDERS", 5)

    daily_loss_stop_pct = env_float("DAILY_LOSS_STOP_PCT", 1.0)
    max_daily_loss_usd = env_float("MAX_DAILY_LOSS_USD", 200.0)
    enable_daily_risk_guard = env_bool("ENABLE_DAILY_RISK_GUARD", False)

    symbol_cooldown_seconds = env_int("SYMBOL_COOLDOWN_SECONDS", 0)
    pick_ttl_seconds = env_int("PICK_TTL_SECONDS", 0)

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | ALPACA_DEDUPE_MINUTES=%s | "
        "CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%.1f | "
        "MAX_DAILY_LOSS_USD=%.1f | ENABLE_DAILY_RISK_GUARD=%s | SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s",
        min_strength,
        symbols,
        portfolio_id,
        poll_seconds,
        allow_short,
        long_only,
        max_notional,
        max_qty,
        dedupe_minutes,
        cancel_opposite,
        max_open_positions,
        max_open_orders,
        daily_loss_stop_pct,
        max_daily_loss_usd,
        enable_daily_risk_guard,
        symbol_cooldown_seconds,
        pick_ttl_seconds,
    )

    tc = get_trading_client()

    while True:
        try:
            conn = get_db_conn()

            # release stale picks
            released = release_stale_picks(conn, pick_ttl_seconds)
            if released:
                logger.warning("release_stale_picks | released=%d", released)

            # global risk guard before doing anything
            ok, why = risk_guard_ok(
                tc,
                max_open_positions=max_open_positions,
                max_open_orders=max_open_orders,
                enable_daily_risk_guard=enable_daily_risk_guard,
                daily_loss_stop_pct=daily_loss_stop_pct,
                max_daily_loss_usd=max_daily_loss_usd,
            )
            if not ok:
                logger.warning("risk_guard | blocked new entries | %s", why)
                conn.close()
                time.sleep(poll_seconds)
                continue

            new_signals = fetch_new_signals(
                conn,
                portfolio_id=portfolio_id,
                symbols=symbols,
                min_strength=min_strength,
                limit=200,
            )
            selected = select_signals(new_signals)

            for s in selected:
                # mid-batch guard
                ok2, why2 = risk_guard_ok(
                    tc,
                    max_open_positions=max_open_positions,
                    max_open_orders=max_open_orders,
                    enable_daily_risk_guard=enable_daily_risk_guard,
                    daily_loss_stop_pct=daily_loss_stop_pct,
                    max_daily_loss_usd=max_daily_loss_usd,
                )
                if not ok2:
                    # don't mark anything else; leave remaining unprocessed
                    logger.warning("risk_guard | blocked mid-batch | %s | unpicked_remaining=%d", why2, len(selected) - selected.index(s))
                    break

                # long_only filter
                if long_only and s.side.lower() != "buy":
                    mark_processed(conn, s.id, "skipped", "long_only")
                    logger.info("skip | sid=%s %s %s | long_only", s.id, s.symbol, s.side)
                    continue

                # short disable
                if not can_short(s.side, allow_short):
                    mark_processed(conn, s.id, "skipped", "short_disabled")
                    logger.info("skip | sid=%s %s %s | short_disabled", s.id, s.symbol, s.side)
                    continue

                # symbol cooldown (based on last processed_at)
                ok_cd, note_cd = symbol_cooldown_ok(conn, portfolio_id, s.symbol, symbol_cooldown_seconds)
                if not ok_cd:
                    mark_processed(conn, s.id, "skipped", note_cd)
                    logger.info("skip | sid=%s %s %s | %s", s.id, s.symbol, s.side, note_cd)
                    continue

                # pick (simple lock)
                if not pick_signal(conn, s.id, pick_ttl_seconds):
                    # someone else picked it
                    logger.info("skip | sid=%s %s %s | pick_failed_already_picked", s.id, s.symbol, s.side)
                    continue

                side_enum = side_to_order_side(s.side)

                # dedupe in Alpaca
                if dedupe_minutes > 0 and dedupe_recent_alpaca(tc, s.symbol, side_enum, dedupe_minutes):
                    mark_processed(conn, s.id, "skipped", f"dedupe_alpaca_{dedupe_minutes}m")
                    logger.info("skip | sid=%s %s %s | dedupe_alpaca_%sm", s.id, s.symbol, s.side, dedupe_minutes)
                    continue

                # optionally cancel opposite open orders
                if cancel_opposite:
                    c = cancel_opposite_open_orders(tc, s.symbol, side_enum)
                    if c:
                        logger.info("cancel_opposite_open_orders | symbol=%s canceled=%d", s.symbol, c)

                # qty sizing (very simple)
                qty = compute_order_qty(s.symbol, max_qty)

                # submit as limit at current-ish price: you likely fill this from your quote logic elsewhere.
                # To avoid None, we approximate with last known trade from positions or account? Not available here.
                # We'll set a dummy and expect your project to override compute_limit_price().
                # If None -> skip.
                limit_price = compute_limit_price(tc, s.symbol, side_enum)
                if limit_price is None:
                    mark_processed(conn, s.id, "skipped", "no_limit_price")
                    logger.info("skip | sid=%s %s %s | no_limit_price", s.id, s.symbol, s.side)
                    continue

                # notional guard
                notional = float(limit_price) * float(qty)
                if max_notional > 0 and notional > max_notional:
                    mark_processed(conn, s.id, "skipped", f"max_notional_exceeded {notional:.2f}>{max_notional:.2f}")
                    logger.info("skip | sid=%s %s %s | max_notional_exceeded %.2f>%.2f", s.id, s.symbol, s.side, notional, max_notional)
                    continue

                try:
                    oid = submit_limit_order(tc, s.symbol, side_enum, qty, float(limit_price))
                    mark_processed(conn, s.id, "submitted", "submitted", oid)
                    logger.info(
                        "submitted | sid=%s | %s %s qty=%s limit=%s | alpaca_id=%s",
                        s.id,
                        s.symbol,
                        s.side,
                        qty,
                        limit_price,
                        oid,
                    )
                except Exception as e:
                    mark_processed(conn, s.id, "error", f"submit_failed:{type(e).__name__}")
                    logger.exception("submit_failed | sid=%s %s %s | %s", s.id, s.symbol, s.side, e)

            conn.close()

        except Exception as e:
            logger.exception("loop_error | %s", e)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
