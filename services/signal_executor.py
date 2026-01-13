# services/signal_executor.py
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
from sqlalchemy import create_engine, text as sql

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

log = logging.getLogger("signal_executor")


# -------------------------
# env helpers
# -------------------------
def env_str(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return d if v is None else str(v).strip()


def env_int(k: str, d: int) -> int:
    v = env_str(k, "")
    return d if v == "" else int(float(v))


def env_float(k: str, d: float) -> float:
    v = env_str(k, "")
    return d if v == "" else float(v)


def env_bool(k: str, d: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return d
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# -------------------------
# db
# -------------------------
def get_engine():
    db_url = env_str("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True, future=True)


def fetch_unprocessed_signals(engine, portfolio_id: int, symbols: List[str], min_strength: float, limit: int):
    if not symbols:
        return []
    q = sql(
        """
        select id, created_at, symbol, side, strength
        from signals
        where portfolio_id = :pid
          and processed_at is null
          and strength >= :min_strength
          and symbol = any(:symbols)
        order by created_at asc
        limit :limit
        """
    )
    with engine.begin() as c:
        rows = c.execute(
            q, {"pid": portfolio_id, "symbols": symbols, "min_strength": min_strength, "limit": limit}
        ).mappings().all()

    out = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "symbol": str(r["symbol"]).upper(),
                "side": str(r["side"]).lower(),
                "strength": float(r["strength"]),
            }
        )
    return out


def select_signals(rows: List[dict], max_unique: int):
    """1 signal per (symbol, side); prefer higher strength."""
    best: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        k = (r["symbol"], r["side"])
        cur = best.get(k)
        if cur is None or r["strength"] > cur["strength"] or (
            r["strength"] == cur["strength"] and r["created_at"] > cur["created_at"]
        ):
            best[k] = r

    selected = list(best.values())
    selected.sort(key=lambda x: (-x["strength"], x["created_at"]))
    return selected[:max_unique]


def mark_picked(engine, ids: List[int]):
    if not ids:
        return
    q = sql(
        """
        update signals
        set processed_at = now(),
            processed_status = 'picked',
            processed_note = 'picked'
        where id = any(:ids)
          and processed_at is null
        """
    )
    with engine.begin() as c:
        c.execute(q, {"ids": ids})


def unpick(engine, ids: List[int], note: str):
    if not ids:
        return
    q = sql(
        """
        update signals
        set processed_at = null,
            processed_status = null,
            processed_note = :note
        where id = any(:ids)
          and processed_status = 'picked'
        """
    )
    with engine.begin() as c:
        c.execute(q, {"ids": ids, "note": note})


def mark_processed(engine, sid: int, status: str, note: str):
    q = sql(
        """
        update signals
        set processed_at = now(),
            processed_status = :status,
            processed_note = :note
        where id = :id
        """
    )
    with engine.begin() as c:
        c.execute(q, {"id": sid, "status": status, "note": note})


def release_stale_picks(engine, ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        return 0
    q = sql(
        """
        update signals
        set processed_at = null,
            processed_status = null,
            processed_note = 'auto_unpick_ttl'
        where processed_status = 'picked'
          and processed_at < (now() - (:ttl * interval '1 second'))
        """
    )
    with engine.begin() as c:
        res = c.execute(q, {"ttl": ttl_seconds})
        return int(res.rowcount or 0)


def symbol_in_cooldown(engine, symbol: str, cooldown_seconds: int) -> bool:
    if cooldown_seconds <= 0:
        return False
    q = sql(
        """
        select max(processed_at) as last_ts
        from signals
        where symbol = :sym
          and processed_at is not null
          and processed_status in ('submitted','filled')
        """
    )
    with engine.begin() as c:
        row = c.execute(q, {"sym": symbol}).mappings().first()
    last_ts = row["last_ts"] if row else None
    if last_ts is None:
        return False
    if getattr(last_ts, "tzinfo", None) is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    return last_ts >= (now_utc() - timedelta(seconds=cooldown_seconds))


# -------------------------
# alpaca helpers
# -------------------------
def get_tc() -> TradingClient:
    key = env_str("ALPACA_API_KEY")
    sec = env_str("ALPACA_API_SECRET")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing")
    paper = env_bool("ALPACA_PAPER", True)
    return TradingClient(key, sec, paper=paper)


def get_latest_price(symbol: str) -> Optional[float]:
    data_url = env_str("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
    feed = env_str("ALPACA_DATA_FEED", "")
    headers = {
        "APCA-API-KEY-ID": env_str("ALPACA_API_KEY"),
        "APCA-API-SECRET-KEY": env_str("ALPACA_API_SECRET"),
    }
    params = {"feed": feed} if feed else None

    # latest trade
    try:
        r = requests.get(f"{data_url}/v2/stocks/{symbol}/trades/latest", headers=headers, params=params, timeout=10)
        if r.ok:
            j = r.json() or {}
            p = (j.get("trade") or {}).get("p")
            if p is not None:
                return float(p)
    except Exception:
        pass

    # latest quote
    try:
        r = requests.get(f"{data_url}/v2/stocks/{symbol}/quotes/latest", headers=headers, params=params, timeout=10)
        if r.ok:
            j = r.json() or {}
            q = j.get("quote") or {}
            bp, ap = q.get("bp"), q.get("ap")
            if bp is not None and ap is not None:
                return (float(bp) + float(ap)) / 2.0
            if ap is not None:
                return float(ap)
            if bp is not None:
                return float(bp)
    except Exception:
        pass

    return None


def open_positions_count(tc: TradingClient) -> int:
    try:
        return len(tc.get_all_positions() or [])
    except Exception:
        return 0


def open_orders_count(tc: TradingClient) -> int:
    try:
        return len(tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)) or [])
    except Exception:
        return 0


def market_gate_ok(tc: TradingClient, close_minutes: int) -> bool:
    try:
        clock = tc.get_clock()
        if not clock.is_open:
            return False
        if close_minutes <= 0:
            return True
        nc = clock.next_close
        if getattr(nc, "tzinfo", None) is None:
            nc = nc.replace(tzinfo=timezone.utc)
        return (nc - now_utc()) > timedelta(minutes=close_minutes)
    except Exception:
        return True  # fail-open


def daily_loss_guard_tripped(tc: TradingClient, stop_pct: float, max_loss_usd: float) -> Tuple[bool, str]:
    try:
        a = tc.get_account()
        equity = float(a.equity)
        last_eq = float(getattr(a, "last_equity", 0) or 0)
        if last_eq <= 0:
            return False, "no_last_equity"
        loss = max(0.0, last_eq - equity)
        loss_pct = (loss / last_eq) * 100.0
        if max_loss_usd > 0 and loss >= max_loss_usd:
            return True, f"max_daily_loss_usd_reached:{loss:.2f}>={max_loss_usd:.2f}"
        if stop_pct > 0 and loss_pct >= stop_pct:
            return True, f"daily_loss_stop_pct_reached:{loss_pct:.2f}%>={stop_pct:.2f}%"
        return False, f"ok loss={loss:.2f} ({loss_pct:.2f}%)"
    except Exception as e:
        return False, f"guard_error:{type(e).__name__}"


def alpaca_dedupe(tc: TradingClient, symbol: str, side: OrderSide, minutes: int) -> bool:
    if minutes <= 0:
        return False
    after = now_utc() - timedelta(minutes=minutes)
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200, symbols=[symbol], after=after)
        for o in (tc.get_orders(req) or []):
            if str(o.symbol).upper() == symbol and o.side == side:
                return True
        return False
    except Exception:
        return False


def cancel_opposite_open_orders(tc: TradingClient, symbol: str, desired_side: OrderSide) -> int:
    opposite = OrderSide.SELL if desired_side == OrderSide.BUY else OrderSide.BUY
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, symbols=[symbol])
        orders = tc.get_orders(req) or []
        n = 0
        for o in orders:
            if o.side == opposite:
                try:
                    tc.cancel_order_by_id(o.id)
                    n += 1
                except Exception:
                    pass
        return n
    except Exception:
        return 0


# -------------------------
# main
# -------------------------
def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s:%(name)s:%(message)s")

    MIN_STRENGTH = env_float("MIN_STRENGTH", 0.75)
    SYMBOLS = [s.strip().upper() for s in env_str("SYMBOLS", "AAPL,MSFT,SPY,NVDA,AMD").split(",") if s.strip()]
    PORTFOLIO_ID = env_int("PORTFOLIO_ID", 1)
    POLL_SECONDS = env_int("POLL_SECONDS", 20)

    ALLOW_SHORT = env_bool("ALLOW_SHORT", False)
    LONG_ONLY = env_bool("LONG_ONLY", False)

    MAX_NOTIONAL = env_float("MAX_NOTIONAL", 200.0)
    MAX_QTY = env_int("MAX_QTY", 1)

    ALPACA_DEDUPE_MINUTES = env_int("ALPACA_DEDUPE_MINUTES", 2)
    CANCEL_OPPOSITE_OPEN_ORDERS = env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 1)
    MAX_OPEN_ORDERS = env_int("MAX_OPEN_ORDERS", 1)

    ENABLE_DAILY_RISK_GUARD = env_bool("ENABLE_DAILY_RISK_GUARD", True)
    DAILY_LOSS_STOP_PCT = env_float("DAILY_LOSS_STOP_PCT", 1.0)
    MAX_DAILY_LOSS_USD = env_float("MAX_DAILY_LOSS_USD", 200.0)

    SYMBOL_COOLDOWN_SECONDS = env_int("SYMBOL_COOLDOWN_SECONDS", 60)
    PICK_TTL_SECONDS = env_int("PICK_TTL_SECONDS", 120)

    MARKET_CLOSE_MINUTES = env_int("MARKET_CLOSE_MINUTES", 15)

    FETCH_LIMIT = env_int("FETCH_LIMIT", 200)
    SELECT_MAX = env_int("SELECT_MAX", 5)

    engine = get_engine()
    tc = get_tc()

    log.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | ALPACA_DEDUPE_MINUTES=%s | "
        "CANCEL_OPPOSITE_OPEN_ORDERS=%s | MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | "
        "DAILY_LOSS_STOP_PCT=%.1f | MAX_DAILY_LOSS_USD=%.1f | ENABLE_DAILY_RISK_GUARD=%s | "
        "SYMBOL_COOLDOWN_SECONDS=%s | PICK_TTL_SECONDS=%s",
        MIN_STRENGTH,
        SYMBOLS,
        PORTFOLIO_ID,
        POLL_SECONDS,
        ALLOW_SHORT,
        LONG_ONLY,
        MAX_NOTIONAL,
        MAX_QTY,
        ALPACA_DEDUPE_MINUTES,
        CANCEL_OPPOSITE_OPEN_ORDERS,
        MAX_OPEN_POSITIONS,
        MAX_OPEN_ORDERS,
        DAILY_LOSS_STOP_PCT,
        MAX_DAILY_LOSS_USD,
        ENABLE_DAILY_RISK_GUARD,
        SYMBOL_COOLDOWN_SECONDS,
        PICK_TTL_SECONDS,
    )

    while True:
        try:
            released = release_stale_picks(engine, PICK_TTL_SECONDS)
            if released:
                log.warning("release_stale_picks | released=%s older_than=%ss", released, PICK_TTL_SECONDS)

            if not market_gate_ok(tc, MARKET_CLOSE_MINUTES):
                log.info("market_gate | not entering new trades (market closed or too close to close)")
                time.sleep(POLL_SECONDS)
                continue

            if ENABLE_DAILY_RISK_GUARD:
                tripped, reason = daily_loss_guard_tripped(tc, DAILY_LOSS_STOP_PCT, MAX_DAILY_LOSS_USD)
                if tripped:
                    log.warning("risk_guard | blocked new entries | %s", reason)
                    time.sleep(POLL_SECONDS)
                    continue

            pos = open_positions_count(tc)
            if MAX_OPEN_POSITIONS > 0 and pos >= MAX_OPEN_POSITIONS:
                log.warning("risk_guard | blocked new entries | max_open_positions_reached:%s/%s", pos, MAX_OPEN_POSITIONS)
                time.sleep(POLL_SECONDS)
                continue

            oo = open_orders_count(tc)
            if MAX_OPEN_ORDERS > 0 and oo >= MAX_OPEN_ORDERS:
                log.warning("risk_guard | blocked new entries | max_open_orders_reached:%s/%s", oo, MAX_OPEN_ORDERS)
                time.sleep(POLL_SECONDS)
                continue

            rows = fetch_unprocessed_signals(engine, PORTFOLIO_ID, SYMBOLS, MIN_STRENGTH, FETCH_LIMIT)
            log.info("fetch_new_signals | fetched %s rows", len(rows))
            if not rows:
                time.sleep(POLL_SECONDS)
                continue

            selected = select_signals(rows, SELECT_MAX)
            log.info(
                "select_signals | fetched=%s | selected=%s | unique_symbol_side=%s",
                len(rows),
                len(selected),
                len({(r["symbol"], r["side"]) for r in selected}),
            )
            if not selected:
                time.sleep(POLL_SECONDS)
                continue

            picked_ids = [r["id"] for r in selected]
            mark_picked(engine, picked_ids)

            remaining = picked_ids.copy()
            for r in selected:
                remaining.remove(r["id"])

                pos = open_positions_count(tc)
                if MAX_OPEN_POSITIONS > 0 and pos >= MAX_OPEN_POSITIONS:
                    unpick(engine, [r["id"]] + remaining, "risk_guard_max_open_positions")
                    log.warning(
                        "risk_guard | blocked mid-batch | max_open_positions_reached:%s/%s | unpicked_remaining=%s",
                        pos,
                        MAX_OPEN_POSITIONS,
                        len(remaining) + 1,
                    )
                    break

                oo = open_orders_count(tc)
                if MAX_OPEN_ORDERS > 0 and oo >= MAX_OPEN_ORDERS:
                    unpick(engine, [r["id"]] + remaining, "risk_guard_max_open_orders")
                    log.warning(
                        "risk_guard | blocked mid-batch | max_open_orders_reached:%s/%s | unpicked_remaining=%s",
                        oo,
                        MAX_OPEN_ORDERS,
                        len(remaining) + 1,
                    )
                    break

                sym, side = r["symbol"], r["side"]
                if symbol_in_cooldown(engine, sym, SYMBOL_COOLDOWN_SECONDS):
                    mark_processed(engine, r["id"], "skipped", f"symbol_cooldown_{SYMBOL_COOLDOWN_SECONDS}s")
                    log.info("skip | sid=%s %s %s | symbol_cooldown_%ss", r["id"], sym, side, SYMBOL_COOLDOWN_SECONDS)
                    continue

                if side not in ("buy", "sell"):
                    mark_processed(engine, r["id"], "skipped", "bad_side")
                    log.info("skip | sid=%s %s %s | bad_side", r["id"], sym, side)
                    continue

                desired = OrderSide.BUY if side == "buy" else OrderSide.SELL
                if LONG_ONLY and desired == OrderSide.SELL:
                    mark_processed(engine, r["id"], "skipped", "long_only")
                    log.info("skip | sid=%s %s %s | long_only", r["id"], sym, side)
                    continue
                if (not ALLOW_SHORT) and desired == OrderSide.SELL:
                    mark_processed(engine, r["id"], "skipped", "short_disabled")
                    log.info("skip | sid=%s %s %s | short_disabled", r["id"], sym, side)
                    continue

                if alpaca_dedupe(tc, sym, desired, ALPACA_DEDUPE_MINUTES):
                    mark_processed(engine, r["id"], "skipped", f"dedupe_alpaca_{ALPACA_DEDUPE_MINUTES}m")
                    log.info("skip | sid=%s %s %s | dedupe_alpaca_%sm", r["id"], sym, side, ALPACA_DEDUPE_MINUTES)
                    continue

                px = get_latest_price(sym)
                if px is None or px <= 0:
                    mark_processed(engine, r["id"], "skipped", "no_price")
                    log.info("skip | sid=%s %s %s | no_price", r["id"], sym, side)
                    continue

                qty = max(1, int(MAX_NOTIONAL // px)) if MAX_NOTIONAL > 0 else MAX_QTY
                qty = min(qty, MAX_QTY)
                if qty < 1:
                    mark_processed(engine, r["id"], "skipped", "qty_lt_1")
                    log.info("skip | sid=%s %s %s | qty_lt_1", r["id"], sym, side)
                    continue

                limit_px = round(px, 2)

                if CANCEL_OPPOSITE_OPEN_ORDERS:
                    cancel_opposite_open_orders(tc, sym, desired)

                try:
                    req = LimitOrderRequest(symbol=sym, qty=qty, side=desired, time_in_force=TimeInForce.DAY, limit_price=limit_px)
                    o = tc.submit_order(req)
                    oid = getattr(o, "id", None)
                    st = str(getattr(o, "status", "")).lower()
                    if st == "filled":
                        mark_processed(engine, r["id"], "filled", "order filled")
                    else:
                        mark_processed(engine, r["id"], "submitted", f"alpaca_id={oid}")
                    log.info("submitted | sid=%s | %s %s qty=%s limit=%.2f | alpaca_id=%s", r["id"], sym, side, qty, limit_px, oid)
                except Exception as e:
                    mark_processed(engine, r["id"], "error", f"{type(e).__name__}: {e}")
                    log.exception("submit_failed | sid=%s %s %s", r["id"], sym, side)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            log.exception("loop_error: %s", e)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
