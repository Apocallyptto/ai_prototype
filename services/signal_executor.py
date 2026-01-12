import os
import time
import math
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Dict, Any, Tuple, Set

import requests
from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.models import Order

from services.market_gate import should_trade_now


logger = logging.getLogger("signal_executor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


# -----------------------------
# env helpers
# -----------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


def env_int(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(float(str(v).strip()))


def env_float(name: str, default: float = 0.0) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip() not in ("0", "false", "False", "no", "NO")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------
# Alpaca price helper (Data API)
# -----------------------------
def get_latest_price(symbol: str) -> Optional[float]:
    """
    Pull latest price from Alpaca Data API v2.
    Uses ALPACA_DATA_URL + ALPACA_DATA_FEED (iex/sip).
    Falls back to quote mid if trade isn't available.
    """
    data_url = env_str("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
    data_feed = env_str("ALPACA_DATA_FEED", "").strip() or None

    api_key = env_str("ALPACA_API_KEY")
    api_secret = env_str("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.warning("get_latest_price | missing ALPACA_API_KEY/SECRET")
        return None

    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"feed": data_feed} if data_feed else None

    # 1) latest trade
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/trades/latest",
            headers=headers,
            params=params,
            timeout=10,
        )
        if r.ok:
            j = r.json() or {}
            t = j.get("trade") or {}
            p = t.get("p")
            if p is not None:
                return float(p)
        else:
            logger.debug("get_latest_price | trade latest failed | %s | %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.debug("get_latest_price | trade latest exception | %s | %s", symbol, e)

    # 2) latest quote mid
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=headers,
            params=params,
            timeout=10,
        )
        if r.ok:
            j = r.json() or {}
            q = j.get("quote") or {}
            bp = q.get("bp")
            ap = q.get("ap")
            if bp is not None and ap is not None:
                return (float(bp) + float(ap)) / 2.0
            if ap is not None:
                return float(ap)
            if bp is not None:
                return float(bp)
        else:
            logger.debug("get_latest_price | quote latest failed | %s | %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.debug("get_latest_price | quote latest exception | %s | %s", symbol, e)

    return None


# -----------------------------
# DB: signals (source of truth = processed_at)
# -----------------------------
def fetch_unprocessed_signals(engine, portfolio_id: int, symbols: List[str], min_strength: float, limit: int = 50):
    sql = text(
        """
        SELECT id, created_at, symbol, side, strength
        FROM signals
        WHERE processed_at IS NULL
          AND portfolio_id = :pid
          AND symbol = ANY(:symbols)
          AND strength >= :min_strength
        ORDER BY created_at ASC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"pid": portfolio_id, "symbols": symbols, "min_strength": min_strength, "limit": limit},
        ).mappings().all()
    logger.info("fetch_new_signals | fetched %d rows", len(rows))
    return rows


def mark_signal(engine, signal_id: int, status: str, note: str, alpaca_order_id=None) -> None:
    if alpaca_order_id is not None:
        alpaca_order_id = str(alpaca_order_id)

    sql = text(
        """
        UPDATE signals
        SET processed_status = :status,
            processed_note   = :note,
            processed_at     = NOW(),
            alpaca_order_id  = COALESCE(CAST(:oid AS text), alpaca_order_id)
        WHERE id = :id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"status": status, "note": note, "oid": alpaca_order_id, "id": signal_id})


def mark_signals_picked(engine, ids: List[int]) -> Set[int]:
    """
    Atomically mark selected signals as picked (only if not yet processed).
    Returns IDs that were successfully picked.
    """
    if not ids:
        return set()

    sql = text(
        """
        UPDATE signals
        SET processed_status = 'picked',
            processed_note   = 'picked',
            processed_at     = NOW()
        WHERE id = ANY(:ids)
          AND processed_at IS NULL
        RETURNING id
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"ids": ids}).fetchall()
    return {int(r[0]) for r in rows}


def select_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep at most one signal per (symbol, side) in this batch. (If duplicates, keep the strongest.)
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in signals:
        key = (s["symbol"], s["side"])
        if key not in best or float(s["strength"]) > float(best[key]["strength"]):
            best[key] = s
    selected = list(best.values())
    logger.info(
        "select_signals | fetched=%d | selected=%d | unique_symbol_side=%d",
        len(signals),
        len(selected),
        len(best),
    )
    return selected


# -----------------------------
# Alpaca-side dedupe + hygiene
# -----------------------------
def alpaca_dedupe_exists(tc: TradingClient, symbol: str, alpaca_side: OrderSide, lookback_minutes: int, limit: int = 200) -> bool:
    """
    Checks if Alpaca has a recent order for (symbol, side) within lookback window.
    Prevents repeated submits even if DB state lags.
    """
    try:
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
    except Exception as e:
        logger.debug("alpaca_dedupe_exists | get_orders failed | %s", e)
        return False

    cutoff = now_utc() - timedelta(minutes=lookback_minutes)
    for o in orders:
        try:
            if str(o.symbol).upper() != symbol:
                continue
            if o.side != alpaca_side:
                continue
            if o.submitted_at is None:
                continue
            ts = o.submitted_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                return True
        except Exception:
            continue
    return False


def cancel_opposite_open_orders(tc: TradingClient, symbol: str, alpaca_side: OrderSide, limit: int = 500) -> int:
    """
    Cancels OPEN orders on the same symbol but opposite side.
    Helps avoid accidental self-cross / conflicts.
    """
    try:
        open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=limit))
    except Exception as e:
        logger.debug("cancel_opposite_open_orders | get_orders failed | %s", e)
        return 0

    canceled = 0
    for o in open_orders:
        try:
            if str(o.symbol).upper() != symbol:
                continue
            if o.side == alpaca_side:
                continue
            tc.cancel_order_by_id(o.id)
            canceled += 1
        except Exception:
            continue
    return canceled


# -----------------------------
# Risk guard: persistent daily baseline in Postgres
# -----------------------------
def ensure_bot_state_table(engine) -> None:
    sql = text(
        """
        CREATE TABLE IF NOT EXISTS bot_state (
            key text PRIMARY KEY,
            value text NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)


def get_state(engine, key: str) -> Optional[str]:
    sql = text("SELECT value FROM bot_state WHERE key = :k")
    with engine.connect() as conn:
        row = conn.execute(sql, {"k": key}).fetchone()
    return row[0] if row else None


def set_state(engine, key: str, value: str) -> None:
    sql = text(
        """
        INSERT INTO bot_state(key, value, updated_at)
        VALUES(:k, :v, now())
        ON CONFLICT (key)
        DO UPDATE SET value = EXCLUDED.value, updated_at = now();
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"k": key, "v": value})


def get_today_equity_baseline(engine, tc: TradingClient) -> float:
    """
    Returns baseline equity for *today* (UTC date), stored in DB.
    If missing, initializes it from current account equity.
    """
    d = date.today().isoformat()
    key = f"equity_start_{d}"
    v = get_state(engine, key)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass

    a = tc.get_account()
    baseline = float(a.equity)
    set_state(engine, key, str(baseline))
    return baseline


def risk_guard_allows_new_entries(
    engine,
    tc: TradingClient,
    max_open_positions: int,
    max_open_orders: int,
    daily_loss_stop_pct: float,
    max_daily_loss_usd: float,
) -> Tuple[bool, str]:
    """
    Global guard. If false -> do not trade this cycle.
    """
    # Positions guard
    try:
        pos = tc.get_all_positions()
        npos = len(pos)
        if max_open_positions >= 0 and npos >= max_open_positions:
            return False, f"max_open_positions_reached:{npos}/{max_open_positions}"
    except Exception as e:
        logger.warning("risk_guard | positions check failed | %s", e)

    # Open orders guard
    try:
        o = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        nopen = len(o)
        if max_open_orders >= 0 and nopen >= max_open_orders:
            return False, f"max_open_orders_reached:{nopen}/{max_open_orders}"
    except Exception as e:
        logger.warning("risk_guard | open orders check failed | %s", e)

    # Daily loss guard
    if (daily_loss_stop_pct and daily_loss_stop_pct > 0) or (max_daily_loss_usd and max_daily_loss_usd > 0):
        try:
            baseline = get_today_equity_baseline(engine, tc)
            cur = float(tc.get_account().equity)
            loss_usd = max(0.0, baseline - cur)
            loss_pct = (loss_usd / baseline) * 100.0 if baseline > 0 else 0.0

            if max_daily_loss_usd and max_daily_loss_usd > 0 and loss_usd >= max_daily_loss_usd:
                return False, f"daily_loss_usd_stop:{loss_usd:.2f}>={max_daily_loss_usd:.2f}"

            if daily_loss_stop_pct and daily_loss_stop_pct > 0 and loss_pct >= daily_loss_stop_pct:
                return False, f"daily_loss_pct_stop:{loss_pct:.3f}%>={daily_loss_stop_pct:.3f}%"

        except Exception as e:
            logger.warning("risk_guard | daily loss check failed | %s", e)

    return True, "ok"

def in_symbol_cooldown(engine, portfolio_id: int, symbol: str, cooldown_seconds: int) -> bool:
    if cooldown_seconds <= 0:
        return False
    sql = text("""
        SELECT 1
        FROM signals
        WHERE portfolio_id = :pid
          AND symbol = :sym
          AND processed_at IS NOT NULL
          AND processed_at >= NOW() - (:cd || ' seconds')::interval
          AND processed_status IN ('submitted','filled')
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"pid": portfolio_id, "sym": symbol, "cd": int(cooldown_seconds)}).fetchone()
    return row is not None


def cooldown_reason(cooldown_seconds: int) -> str:
    return f"symbol_cooldown_{int(cooldown_seconds)}s"


# -----------------------------
# main loop
# -----------------------------
def main():
    # ---- config ----
    api_key = env_str("ALPACA_API_KEY")
    api_secret = env_str("ALPACA_API_SECRET")

    # Your env uses TRADING_MODE=paper. Keep compatibility with ALPACA_PAPER too.
    trading_mode = env_str("TRADING_MODE", "").lower()
    paper = (trading_mode == "paper") or (env_str("ALPACA_PAPER", "1") != "0")

    symbols = [s.strip().upper() for s in env_str("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
    portfolio_id = env_int("PORTFOLIO_ID", 1)

    # Your env uses EXEC_MIN_STRENGTH / EXEC_POLL_SECONDS
    min_strength = env_float("EXEC_MIN_STRENGTH", env_float("MIN_STRENGTH", 0.60))
    poll_seconds = env_int("EXEC_POLL_SECONDS", env_int("POLL_SECONDS", 20))

    stop_new_entries_min_before_close = env_int("STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE", 10)

    # sizing
    max_qty = env_int("MAX_QTY", 5)
    max_notional = env_float("MAX_NOTIONAL", 1200.0)

    allow_short = env_str("ALLOW_SHORT", "1") != "0"
    long_only = env_str("LONG_ONLY", "0") != "0"

    # dedupe + hygiene
    alpaca_dedupe_minutes = env_int("ALPACA_DEDUPE_MINUTES", 10)
    cancel_opposite = env_bool("CANCEL_OPPOSITE_OPEN_ORDERS", True)

    # ---- Risk guard env ----
    max_open_positions = env_int("MAX_OPEN_POSITIONS", 999999)  # set 0/1/2...; -1 disables
    max_open_orders = env_int("MAX_OPEN_ORDERS", 999999)        # set 0/1/2...; -1 disables
    daily_loss_stop_pct = env_float("DAILY_LOSS_STOP_PCT", 0.0) # percent, e.g. 1.0
    max_daily_loss_usd = env_float("MAX_DAILY_LOSS_USD", 0.0)   # optional absolute USD stop
    enable_daily_guard = env_bool("ENABLE_DAILY_RISK_GUARD", True)  # if you keep 0 in env, it disables

    # DB URL + engine
    db_url = env_str("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
    db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
    engine = create_engine(db_url, pool_pre_ping=True)

    # Alpaca client
    tc = TradingClient(api_key, api_secret, paper=paper)

    # Ensure state table exists (for daily baseline)
    ensure_bot_state_table(engine)

    logger.info(
        "signal_executor starting | MIN_STRENGTH=%.4f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | "
        "ALLOW_SHORT=%s | LONG_ONLY=%s | MAX_NOTIONAL=%.2f | MAX_QTY=%s | "
        "ALPACA_DEDUPE_MINUTES=%s | CANCEL_OPPOSITE_OPEN_ORDERS=%s | "
        "MAX_OPEN_POSITIONS=%s | MAX_OPEN_ORDERS=%s | DAILY_LOSS_STOP_PCT=%s | MAX_DAILY_LOSS_USD=%s | ENABLE_DAILY_RISK_GUARD=%s",
        min_strength,
        symbols,
        portfolio_id,
        poll_seconds,
        allow_short,
        long_only,
        max_notional,
        max_qty,
        alpaca_dedupe_minutes,
        cancel_opposite,
        max_open_positions,
        max_open_orders,
        daily_loss_stop_pct,
        max_daily_loss_usd,
        enable_daily_guard,
    )

    # ---- loop ----
    while True:
        try:
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

            # GLOBAL RISK GUARD (do not even fetch signals if capacity/stop hit)
            if enable_daily_guard:
                allowed, why = risk_guard_allows_new_entries(
                    engine=engine,
                    tc=tc,
                    max_open_positions=max_open_positions,
                    max_open_orders=max_open_orders,
                    daily_loss_stop_pct=daily_loss_stop_pct,
                    max_daily_loss_usd=max_daily_loss_usd,
                )
                if not allowed:
                    logger.warning("risk_guard | blocked new entries | %s", why)
                    time.sleep(poll_seconds)
                    continue

            # 1) fetch signals
            signals = fetch_unprocessed_signals(engine, portfolio_id, symbols, min_strength, limit=50)
            if not signals:
                time.sleep(poll_seconds)
                continue

            # 2) dedupe batch
            signals = select_signals(signals)

            # 2.5) pre-mark as picked (atomic) so they cannot loop
            ids = [int(s["id"]) for s in signals]
            picked_ids = mark_signals_picked(engine, ids)
            signals = [s for s in signals if int(s["id"]) in picked_ids]
            if not signals:
                logger.info("picked | none (already processed/picked)")
                time.sleep(poll_seconds)
                continue

            # 3) execute each
            for s in signals:
                sid = int(s["id"])
                symbol = str(s["symbol"]).upper()
                side = str(s["side"]).lower().strip()

                # Re-check risk guard per-signal (so we stop mid-batch if we hit cap)
                if enable_daily_guard:
                    allowed, why = risk_guard_allows_new_entries(
                        engine=engine,
                        tc=tc,
                        max_open_positions=max_open_positions,
                        max_open_orders=max_open_orders,
                        daily_loss_stop_pct=daily_loss_stop_pct,
                        max_daily_loss_usd=max_daily_loss_usd,
                    )
                    if not allowed:
                        # do NOT skip/mark signals; keep them unprocessed for later
                        logger.warning("risk_guard | blocked mid-batch | %s", why)
                        break

                if long_only and side == "sell":
                    mark_signal(engine, sid, "skipped", "long_only")
                    logger.info("skip | sid=%s %s %s | long_only", sid, symbol, side)
                    continue
                if (not allow_short) and side == "sell":
                    mark_signal(engine, sid, "skipped", "short_disabled")
                    logger.info("skip | sid=%s %s %s | short_disabled", sid, symbol, side)
                    continue

                # latest price for limit calc / sanity
                px = get_latest_price(symbol)
                if px is None or not math.isfinite(px) or px <= 0:
                    mark_signal(engine, sid, "skipped", "no_limit_price")
                    logger.info("skip | sid=%s %s %s | no_limit_price", sid, symbol, side)
                    continue

                # sizing (simple fixed caps)
                qty = max(1, min(max_qty, int(max_notional // px)))
                if qty <= 0:
                    mark_signal(engine, sid, "skipped", "qty_zero")
                    logger.info("skip | sid=%s %s %s | qty_zero", sid, symbol, side)
                    continue

                alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
                limit_price = round(px, 2)

                # Alpaca-side dedupe
                if alpaca_dedupe_minutes > 0 and alpaca_dedupe_exists(tc, symbol, alpaca_side, alpaca_dedupe_minutes):
                    mark_signal(engine, sid, "skipped", f"dedupe_alpaca_{alpaca_dedupe_minutes}m")
                    logger.info("skip | sid=%s %s %s | dedupe_alpaca_%sm", sid, symbol, side, alpaca_dedupe_minutes)
                    continue

                # cancel opposite open orders (optional)
                if cancel_opposite:
                    canceled = cancel_opposite_open_orders(tc, symbol, alpaca_side)
                    if canceled:
                        logger.info("canceled_opposite_open_orders | %s | canceled=%s", symbol, canceled)

                try:
                    req = LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=limit_price,
                    )
                    o: Order = tc.submit_order(req)
                    mark_signal(engine, sid, "submitted", f"limit={limit_price} qty={qty}", alpaca_order_id=o.id)
                    logger.info(
                        "submitted | sid=%s | %s %s qty=%s limit=%s | alpaca_id=%s",
                        sid, symbol, side, qty, limit_price, o.id
                    )

                except Exception as e:
                    logger.exception(
                        "submit_order failed | sid=%s | %s %s qty=%s px=%s | %s",
                        sid, symbol, side, qty, limit_price, e
                    )
                    mark_signal(engine, sid, "error", f"submit_failed:{type(e).__name__}")

            time.sleep(poll_seconds)

        except Exception:
            logger.exception("signal_executor loop error")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
