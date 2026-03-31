import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, text

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
except Exception:
    TradingClient = None
    QueryOrderStatus = None
    GetOrdersRequest = None


NON_BLOCKING_AUDIT_REASONS = {"no_signal"}
BENIGN_POST_TRADE_BLOCK_REASONS = {"market_closed", "max_open_positions"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _get_db_url() -> str:
    return (
        os.getenv("DB_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg2://postgres:postgres@postgres:5432/trader"
    )


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _fmt_num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _fmt_pct(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return str(x)


def _csv_env(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _normalize_side(side: Any) -> str:
    s = str(side or "").strip().lower()
    if s.endswith(".buy") or s == "buy":
        return "buy"
    if s.endswith(".sell") or s == "sell":
        return "sell"
    return s


def _normalize_status(status: Any) -> str:
    s = str(status or "").strip().lower()
    if s.endswith(".filled") or s == "filled":
        return "filled"
    if s.endswith(".canceled") or s == "canceled":
        return "canceled"
    if s.endswith(".cancelled") or s == "cancelled":
        return "canceled"
    if s.endswith(".new") or s == "new":
        return "new"
    if s.endswith(".open") or s == "open":
        return "open"
    return s


def _make_trading_client():
    if TradingClient is None:
        return None

    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        return None

    mode = (os.getenv("TRADING_MODE") or "").strip().lower()
    if mode in ("paper", "live"):
        paper = (mode == "paper")
    else:
        paper = _env_bool("ALPACA_PAPER", True)

    try:
        return TradingClient(key, sec, paper=paper)
    except Exception:
        return None


def _fetch_live_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "account": None,
        "positions": [],
        "open_orders": [],
        "error": None,
    }

    tc = _make_trading_client()
    if tc is None:
        out["error"] = "TradingClient unavailable or missing Alpaca credentials"
        return out

    try:
        out["account"] = tc.get_account()
    except Exception as e:
        out["error"] = f"account_error: {e!r}"
        return out

    try:
        positions = tc.get_all_positions() or []
        out["positions"] = [p for p in positions if _safe_float(getattr(p, "qty", 0), 0.0)]
    except Exception as e:
        out["error"] = f"positions_error: {e!r}"

    try:
        if GetOrdersRequest is not None and QueryOrderStatus is not None:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200, nested=True)
            out["open_orders"] = tc.get_orders(filter=req) or []
    except Exception as e:
        if out["error"]:
            out["error"] += f" | open_orders_error: {e!r}"
        else:
            out["error"] = f"open_orders_error: {e!r}"

    return out


def _table_exists(con, table_name: str) -> bool:
    q = text("""
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema = 'public'
        AND table_name = :table_name
    )
    """)
    return bool(con.execute(q, {"table_name": table_name}).scalar())


def _table_columns(con, table_name: str) -> List[str]:
    q = text("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = :table_name
    ORDER BY ordinal_position
    """)
    return [str(r[0]) for r in con.execute(q, {"table_name": table_name}).fetchall()]


def _pick_first(existing: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    s = set(existing)
    for c in candidates:
        if c in s:
            return c
    return None


def _fetch_latest_snapshot(con) -> Optional[Tuple]:
    q = text("""
    SELECT
      ts, equity, cash, buying_power, portfolio_value,
      long_market_value, short_market_value,
      daytrade_count, daytrading_buying_power, account_status
    FROM equity_snapshots
    ORDER BY ts DESC
    LIMIT 1
    """)
    return con.execute(q).fetchone()


def _fetch_daily_equity(con, days: int) -> List[Tuple]:
    q = text("""
    WITH x AS (
      SELECT
        date_trunc('day', ts) AS day_utc,
        ts,
        equity
      FROM equity_snapshots
      WHERE ts >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
        AND equity IS NOT NULL
    ),
    agg AS (
      SELECT DISTINCT ON (day_utc)
        day_utc,
        FIRST_VALUE(equity) OVER (PARTITION BY day_utc ORDER BY ts ASC)  AS open_equity,
        FIRST_VALUE(equity) OVER (PARTITION BY day_utc ORDER BY ts DESC) AS close_equity,
        MIN(ts) OVER (PARTITION BY day_utc) AS first_ts,
        MAX(ts) OVER (PARTITION BY day_utc) AS last_ts
      FROM x
      ORDER BY day_utc, ts DESC
    )
    SELECT
      day_utc::date AS day,
      open_equity,
      close_equity,
      (close_equity - open_equity) AS pnl,
      CASE
        WHEN open_equity > 0 THEN ((close_equity - open_equity) / open_equity * 100.0)
        ELSE NULL
      END AS pnl_pct,
      first_ts,
      last_ts
    FROM agg
    ORDER BY day DESC
    """)
    return list(con.execute(q, {"days": days}).fetchall())


def _fetch_today_order_rows(con) -> List[Tuple]:
    q = text("""
    SELECT
      recorded_at,
      symbol,
      side,
      status,
      type,
      qty,
      filled_qty,
      filled_avg_price,
      notional,
      client_order_id
    FROM alpaca_orders
    WHERE recorded_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
      AND recorded_at <  date_trunc('day', NOW() AT TIME ZONE 'UTC') + interval '1 day'
    ORDER BY recorded_at ASC
    """)
    return list(con.execute(q).fetchall())


def _summarize_order_rows(rows: List[Tuple]) -> Dict[str, Any]:
    filled_total = 0
    filled_buys = 0
    filled_sells = 0
    canceled_total = 0
    symbols = set()
    buy_notional = 0.0
    sell_notional = 0.0

    for r in rows:
        _, symbol, side, status, _, _, _, _, notional, _ = r
        st = _normalize_status(status)
        sd = _normalize_side(side)

        if symbol:
            symbols.add(str(symbol))

        if st == "filled":
            filled_total += 1
            if sd == "buy":
                filled_buys += 1
                buy_notional += _safe_float(notional, 0.0) or 0.0
            elif sd == "sell":
                filled_sells += 1
                sell_notional += _safe_float(notional, 0.0) or 0.0

        if st == "canceled":
            canceled_total += 1

    return {
        "filled_total": filled_total,
        "filled_buys": filled_buys,
        "filled_sells": filled_sells,
        "canceled_total": canceled_total,
        "symbols_total": len(symbols),
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "net_cashflow": sell_notional - buy_notional,
    }


def _fetch_recent_order_rows(con, days: int, limit_rows: int = 20) -> List[Tuple]:
    q = text("""
    SELECT
      recorded_at,
      symbol,
      side,
      status,
      type,
      qty,
      filled_qty,
      filled_avg_price,
      notional,
      client_order_id
    FROM alpaca_orders
    WHERE recorded_at >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
    ORDER BY recorded_at DESC
    LIMIT :limit_rows
    """)
    return list(con.execute(q, {"days": days, "limit_rows": limit_rows}).fetchall())


def _filled_row_sort_key(r: Tuple) -> Tuple[Any, int, str, str]:
    ts = r[0]
    side_s = _normalize_side(r[2])
    side_rank = 0 if side_s == "buy" else 1 if side_s == "sell" else 2
    cid = str(r[9] or "")
    typ = str(r[4] or "")
    return (ts, side_rank, cid, typ)


def _fetch_filled_order_rows(con, days: int) -> List[Tuple]:
    q = text("""
    SELECT
      recorded_at,
      symbol,
      side,
      status,
      type,
      qty,
      filled_qty,
      filled_avg_price,
      notional,
      client_order_id
    FROM alpaca_orders
    WHERE recorded_at >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
    ORDER BY recorded_at ASC
    """)
    rows = list(con.execute(q, {"days": days}).fetchall())
    rows = [r for r in rows if _normalize_status(r[3]) == "filled"]
    rows.sort(key=_filled_row_sort_key)
    return rows


def _fetch_system_flags(con) -> Dict[str, str]:
    q = text("""
    SELECT key, value
    FROM system_flags
    WHERE key LIKE 'GUARD_%' OR key LIKE 'TRADING_%'
    """)
    rows = con.execute(q).fetchall()
    return {str(r[0]): "" if r[1] is None else str(r[1]) for r in rows}


def _fetch_signal_summary(con) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "reason": "",
        "time_col": None,
        "strength_col": None,
        "symbol_col": None,
        "side_col": None,
        "source_col": None,
        "portfolio_col": None,
        "fresh_window_seconds": _env_int("PICK_TTL_SECONDS", 0),
        "total_signals_today": 0,
        "eligible_signals_today": 0,
        "eligible_fresh_signals_now": 0,
        "symbols_today": 0,
        "buy_signals_today": 0,
        "sell_signals_today": 0,
        "eligible_buy_signals_today": 0,
        "eligible_sell_signals_today": 0,
        "eligible_fresh_buy_signals_now": 0,
        "eligible_fresh_sell_signals_now": 0,
        "recent_rows": [],
    }

    if not _table_exists(con, "signals"):
        result["reason"] = "signals table not found"
        return result

    cols = _table_columns(con, "signals")
    time_col = _pick_first(cols, ["created_at", "ts", "inserted_at", "recorded_at", "updated_at"])
    strength_col = _pick_first(cols, ["strength", "score", "confidence"])
    symbol_col = _pick_first(cols, ["symbol", "ticker"])
    side_col = _pick_first(cols, ["side", "signal_side"])
    source_col = _pick_first(cols, ["source", "model_name", "strategy"])
    portfolio_col = _pick_first(cols, ["portfolio_id"])

    if time_col is None:
        result["reason"] = "signals table exists but no usable time column found"
        return result

    result["available"] = True
    result["time_col"] = time_col
    result["strength_col"] = strength_col
    result["symbol_col"] = symbol_col
    result["side_col"] = side_col
    result["source_col"] = source_col
    result["portfolio_col"] = portfolio_col

    min_strength = _env_float("MIN_STRENGTH", 0.0)
    wanted_portfolio = (os.getenv("PORTFOLIO_ID") or "").strip()
    wanted_symbols = set(_csv_env("SYMBOLS"))
    fresh_window_seconds = int(result["fresh_window_seconds"] or 0)

    where_parts = [
        f"{time_col} >= date_trunc('day', NOW() AT TIME ZONE 'UTC')",
        f"{time_col} < date_trunc('day', NOW() AT TIME ZONE 'UTC') + interval '1 day'",
    ]
    params: Dict[str, Any] = {}

    if portfolio_col and wanted_portfolio:
        where_parts.append(f"CAST({portfolio_col} AS TEXT) = :portfolio_id")
        params["portfolio_id"] = wanted_portfolio

    if symbol_col and wanted_symbols:
        where_parts.append(f"{symbol_col} = ANY(:symbols)")
        params["symbols"] = list(wanted_symbols)

    where_sql = " AND ".join(where_parts)

    q_total = text(f"""
    SELECT COUNT(*) AS n
    FROM signals
    WHERE {where_sql}
    """)
    result["total_signals_today"] = int(con.execute(q_total, params).scalar() or 0)

    if symbol_col:
        q_symbols = text(f"""
        SELECT COUNT(DISTINCT {symbol_col}) AS n
        FROM signals
        WHERE {where_sql}
        """)
        result["symbols_today"] = int(con.execute(q_symbols, params).scalar() or 0)

    if side_col:
        q_side = text(f"""
        SELECT
          COUNT(*) FILTER (WHERE LOWER(CAST({side_col} AS TEXT)) = 'buy')  AS buy_n,
          COUNT(*) FILTER (WHERE LOWER(CAST({side_col} AS TEXT)) = 'sell') AS sell_n
        FROM signals
        WHERE {where_sql}
        """)
        side_row = con.execute(q_side, params).fetchone()
        if side_row is not None:
            result["buy_signals_today"] = int(side_row[0] or 0)
            result["sell_signals_today"] = int(side_row[1] or 0)

    if strength_col:
        q_eligible = text(f"""
        SELECT COUNT(*) AS n
        FROM signals
        WHERE {where_sql}
          AND COALESCE({strength_col}, 0) >= :min_strength
        """)
        p2 = dict(params)
        p2["min_strength"] = min_strength
        result["eligible_signals_today"] = int(con.execute(q_eligible, p2).scalar() or 0)

        if side_col:
            q_eligible_side = text(f"""
            SELECT
              COUNT(*) FILTER (
                WHERE LOWER(CAST({side_col} AS TEXT)) = 'buy'
                  AND COALESCE({strength_col}, 0) >= :min_strength
              ) AS buy_n,
              COUNT(*) FILTER (
                WHERE LOWER(CAST({side_col} AS TEXT)) = 'sell'
                  AND COALESCE({strength_col}, 0) >= :min_strength
              ) AS sell_n
            FROM signals
            WHERE {where_sql}
            """)
            side_row = con.execute(q_eligible_side, p2).fetchone()
            if side_row is not None:
                result["eligible_buy_signals_today"] = int(side_row[0] or 0)
                result["eligible_sell_signals_today"] = int(side_row[1] or 0)

        if fresh_window_seconds > 0:
            q_fresh = text(f"""
            SELECT COUNT(*) AS n
            FROM signals
            WHERE {where_sql}
              AND COALESCE({strength_col}, 0) >= :min_strength
              AND {time_col} >= (NOW() AT TIME ZONE 'UTC') - (:fresh_window_seconds || ' seconds')::interval
            """)
            p3 = dict(p2)
            p3["fresh_window_seconds"] = fresh_window_seconds
            result["eligible_fresh_signals_now"] = int(con.execute(q_fresh, p3).scalar() or 0)

            if side_col:
                q_fresh_side = text(f"""
                SELECT
                  COUNT(*) FILTER (
                    WHERE LOWER(CAST({side_col} AS TEXT)) = 'buy'
                      AND COALESCE({strength_col}, 0) >= :min_strength
                      AND {time_col} >= (NOW() AT TIME ZONE 'UTC') - (:fresh_window_seconds || ' seconds')::interval
                  ) AS buy_n,
                  COUNT(*) FILTER (
                    WHERE LOWER(CAST({side_col} AS TEXT)) = 'sell'
                      AND COALESCE({strength_col}, 0) >= :min_strength
                      AND {time_col} >= (NOW() AT TIME ZONE 'UTC') - (:fresh_window_seconds || ' seconds')::interval
                  ) AS sell_n
                FROM signals
                WHERE {where_sql}
                """)
                side_row = con.execute(q_fresh_side, p3).fetchone()
                if side_row is not None:
                    result["eligible_fresh_buy_signals_now"] = int(side_row[0] or 0)
                    result["eligible_fresh_sell_signals_now"] = int(side_row[1] or 0)
    else:
        result["eligible_signals_today"] = result["total_signals_today"]
        result["eligible_buy_signals_today"] = result["buy_signals_today"]
        result["eligible_sell_signals_today"] = result["sell_signals_today"]

    select_cols: List[str] = [time_col]
    for c in [symbol_col, side_col, strength_col, source_col, portfolio_col]:
        if c and c not in select_cols:
            select_cols.append(c)

    q_recent = text(f"""
    SELECT {", ".join(select_cols)}
    FROM signals
    WHERE {where_sql}
    ORDER BY {time_col} DESC
    LIMIT 20
    """)
    result["recent_rows"] = con.execute(q_recent, params).fetchall()

    return result


def _build_execution_funnel(
    signal_summary: Dict[str, Any],
    today_order_summary: Dict[str, Any],
) -> Dict[str, Any]:
    filled_total = int(today_order_summary.get("filled_total", 0) or 0)
    filled_buys = int(today_order_summary.get("filled_buys", 0) or 0)
    filled_sells = int(today_order_summary.get("filled_sells", 0) or 0)

    total_signals = int(signal_summary.get("total_signals_today") or 0)
    eligible_signals = int(signal_summary.get("eligible_signals_today") or 0)
    eligible_fresh_signals_now = int(signal_summary.get("eligible_fresh_signals_now") or 0)
    buy_signals = int(signal_summary.get("buy_signals_today") or 0)
    sell_signals = int(signal_summary.get("sell_signals_today") or 0)
    eligible_buy_signals = int(signal_summary.get("eligible_buy_signals_today") or 0)
    eligible_sell_signals = int(signal_summary.get("eligible_sell_signals_today") or 0)
    eligible_fresh_buy_signals_now = int(signal_summary.get("eligible_fresh_buy_signals_now") or 0)
    eligible_fresh_sell_signals_now = int(signal_summary.get("eligible_fresh_sell_signals_now") or 0)

    long_only = _env_bool("LONG_ONLY", True)
    allow_short = _env_bool("ALLOW_SHORT", False)

    if long_only or not allow_short:
        entry_mode = "buy_entries_only"
        eligible_entry_signals_today = eligible_buy_signals
        eligible_fresh_entry_signals_now = eligible_fresh_buy_signals_now
        executed_entry_fills = filled_buys
    else:
        entry_mode = "all_eligible_signals"
        eligible_entry_signals_today = eligible_signals
        eligible_fresh_entry_signals_now = eligible_fresh_signals_now
        executed_entry_fills = filled_total

    blocked_eligible = max(eligible_entry_signals_today - executed_entry_fills, 0)
    blocked_eligible_buys = max(eligible_buy_signals - filled_buys, 0)

    return {
        "entry_mode": entry_mode,
        "total_signals": total_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "eligible_signals": eligible_signals,
        "eligible_fresh_signals_now": eligible_fresh_signals_now,
        "eligible_buy_signals": eligible_buy_signals,
        "eligible_sell_signals": eligible_sell_signals,
        "eligible_fresh_buy_signals_now": eligible_fresh_buy_signals_now,
        "eligible_fresh_sell_signals_now": eligible_fresh_sell_signals_now,
        "eligible_entry_signals_today": eligible_entry_signals_today,
        "eligible_fresh_entry_signals_now": eligible_fresh_entry_signals_now,
        "executed_entry_fills": executed_entry_fills,
        "executed_buy_fills": filled_buys,
        "executed_sell_fills": filled_sells,
        "filled_total": filled_total,
        "blocked_eligible": blocked_eligible,
        "blocked_eligible_buys": blocked_eligible_buys,
    }


def _infer_block_summary(
    signal_summary: Dict[str, Any],
    today_order_summary: Dict[str, Any],
    latest_snapshot: Optional[Tuple],
    live_positions: List[Any],
    live_open_orders: List[Any],
    flags: Dict[str, str],
) -> Dict[str, Any]:
    funnel = _build_execution_funnel(signal_summary, today_order_summary)

    max_open_positions = _env_int("MAX_OPEN_POSITIONS", 1)
    max_open_orders = _env_int("MAX_OPEN_ORDERS", 1)
    pause_on_daytrade_ge = _env_int("PAUSE_ON_DAYTRADE_COUNT_GE", 999999)
    min_strength = _env_float("MIN_STRENGTH", 0.0)

    daytrade_count = None
    if latest_snapshot is not None:
        daytrade_count = int((latest_snapshot[7] or 0))

    guard_status = flags.get("GUARD_STATUS", "")
    trading_paused = flags.get("TRADING_PAUSED", "")
    paused_reason = flags.get("TRADING_PAUSED_REASON", "")

    eligible_entry_signals_today = funnel["eligible_entry_signals_today"]
    executed_entry_fills = funnel["executed_entry_fills"]

    reason_counts: List[Tuple[str, int]] = []

    if trading_paused == "1":
        reason_counts.append((f"trading_paused ({paused_reason or 'no reason'})", max(eligible_entry_signals_today, 1)))
    elif daytrade_count is not None and daytrade_count >= pause_on_daytrade_ge:
        reason_counts.append((f"PDT/daytrade gate (daytrade_count={daytrade_count}, threshold={pause_on_daytrade_ge})", max(eligible_entry_signals_today - executed_entry_fills, 1)))
    elif funnel["total_signals"] == 0:
        reason_counts.append(("no signals generated today", 1))
    elif eligible_entry_signals_today == 0:
        reason_counts.append((f"no eligible entry signals (MIN_STRENGTH={min_strength:.4f}, mode={funnel['entry_mode']})", 1))
    else:
        if len(live_positions) >= max_open_positions and eligible_entry_signals_today > executed_entry_fills:
            reason_counts.append((f"max_open_positions gate (positions={len(live_positions)}, limit={max_open_positions})", eligible_entry_signals_today - executed_entry_fills))
        if len(live_open_orders) >= max_open_orders and eligible_entry_signals_today > executed_entry_fills:
            reason_counts.append((f"max_open_orders gate (open_orders={len(live_open_orders)}, limit={max_open_orders})", eligible_entry_signals_today - executed_entry_fills))
        if guard_status.startswith("NO_DATA"):
            reason_counts.append((f"guard not ready at day start ({guard_status})", 1))
        if executed_entry_fills > 0 and eligible_entry_signals_today > executed_entry_fills:
            reason_counts.append(("partial execution; some eligible entry signals likely blocked by other gates/cooldowns", eligible_entry_signals_today - executed_entry_fills))
        if not reason_counts:
            if executed_entry_fills > 0:
                reason_counts.append(("all eligible entry signals executed", executed_entry_fills))
            else:
                reason_counts.append(("eligible entry signals existed earlier today but were not executed", max(eligible_entry_signals_today, 1)))

    top_reason = reason_counts[0][0] if reason_counts else "unknown"
    return {
        "funnel": funnel,
        "reason_counts": reason_counts,
        "top_reason": top_reason,
        "source": "heuristic",
        "total_blocked": max(int(funnel.get("blocked_eligible", 0) or 0), 0),
        "state_reason_counts": [],
        "state_top_reason": None,
        "state_total": 0,
    }


def _fetch_execution_audit_block_summary(con) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "reason_counts": [],
        "top_reason": None,
        "total_blocked": 0,
        "source": "execution_audit",
        "state_reason_counts": [],
        "state_top_reason": None,
        "state_total": 0,
        "raw_total_rows": 0,
    }

    if not _table_exists(con, "execution_audit"):
        return result

    q = text("""
    SELECT
      COALESCE(NULLIF(reason, ''), 'unknown') AS reason,
      COUNT(*) AS cnt
    FROM execution_audit
    WHERE event_type = 'blocked_signal'
      AND (ts AT TIME ZONE 'UTC') >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
      AND (ts AT TIME ZONE 'UTC') <  date_trunc('day', NOW() AT TIME ZONE 'UTC') + interval '1 day'
    GROUP BY COALESCE(NULLIF(reason, ''), 'unknown')
    ORDER BY cnt DESC, reason ASC
    """)

    rows = list(con.execute(q).fetchall())
    result["available"] = True
    if not rows:
        return result

    blocking_reason_counts: List[Tuple[str, int]] = []
    state_reason_counts: List[Tuple[str, int]] = []
    total_blocked = 0
    state_total = 0
    raw_total_rows = 0

    for r in rows:
        reason = str(r[0])
        cnt = int(r[1] or 0)
        raw_total_rows += cnt
        if reason in NON_BLOCKING_AUDIT_REASONS:
            state_reason_counts.append((reason, cnt))
            state_total += cnt
        else:
            blocking_reason_counts.append((reason, cnt))
            total_blocked += cnt

    result["reason_counts"] = blocking_reason_counts
    result["top_reason"] = blocking_reason_counts[0][0] if blocking_reason_counts else None
    result["total_blocked"] = total_blocked
    result["state_reason_counts"] = state_reason_counts
    result["state_top_reason"] = state_reason_counts[0][0] if state_reason_counts else None
    result["state_total"] = state_total
    result["raw_total_rows"] = raw_total_rows
    return result


def _merge_block_summaries(
    heuristic_summary: Dict[str, Any],
    audit_summary: Dict[str, Any],
) -> Dict[str, Any]:
    funnel = heuristic_summary["funnel"]

    if audit_summary.get("available") and audit_summary.get("reason_counts"):
        return {
            "funnel": funnel,
            "reason_counts": audit_summary["reason_counts"],
            "top_reason": audit_summary["top_reason"],
            "total_blocked": audit_summary["total_blocked"],
            "source": "execution_audit",
            "fallback_top_reason": heuristic_summary.get("top_reason"),
            "state_reason_counts": audit_summary.get("state_reason_counts", []),
            "state_top_reason": audit_summary.get("state_top_reason"),
            "state_total": audit_summary.get("state_total", 0),
            "raw_total_rows": audit_summary.get("raw_total_rows", audit_summary.get("total_blocked", 0)),
        }

    merged = {
        "funnel": funnel,
        "reason_counts": heuristic_summary.get("reason_counts", []),
        "top_reason": heuristic_summary.get("top_reason"),
        "total_blocked": heuristic_summary.get("total_blocked", 0),
        "source": "heuristic",
        "fallback_top_reason": heuristic_summary.get("top_reason"),
        "state_reason_counts": audit_summary.get("state_reason_counts", []) if audit_summary.get("available") else [],
        "state_top_reason": audit_summary.get("state_top_reason") if audit_summary.get("available") else None,
        "state_total": audit_summary.get("state_total", 0) if audit_summary.get("available") else 0,
        "raw_total_rows": audit_summary.get("raw_total_rows", 0) if audit_summary.get("available") else 0,
    }
    return merged


def _build_realized_trade_summary(filled_rows: List[Tuple]) -> Dict[str, Any]:
    lots_by_symbol: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
    closed_trades: List[Dict[str, Any]] = []

    for r in filled_rows:
        ts, symbol, side, status, order_type, qty, filled_qty, fill_px, notional, client_order_id = r

        sym = str(symbol or "")
        side_s = _normalize_side(side)
        qty_f = _safe_float(filled_qty, 0.0)
        px_f = _safe_float(fill_px, None)

        if not sym or qty_f is None or qty_f <= 0 or px_f is None or px_f <= 0:
            continue

        if side_s == "buy":
            lots_by_symbol[sym].append({
                "qty": qty_f,
                "px": px_f,
                "ts": ts,
                "cid": client_order_id,
                "type": order_type,
            })
            continue

        if side_s != "sell":
            continue

        qty_to_close = qty_f
        while qty_to_close > 1e-12 and lots_by_symbol[sym]:
            lot = lots_by_symbol[sym][0]
            close_qty = min(qty_to_close, lot["qty"])

            pnl_usd = (px_f - lot["px"]) * close_qty
            pnl_pct = ((px_f - lot["px"]) / lot["px"] * 100.0) if lot["px"] else None

            closed_trades.append({
                "symbol": sym,
                "entry_ts": lot["ts"],
                "exit_ts": ts,
                "qty": close_qty,
                "entry_px": lot["px"],
                "exit_px": px_f,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "entry_cid": lot["cid"],
                "exit_cid": client_order_id,
                "exit_type": order_type,
            })

            lot["qty"] -= close_qty
            qty_to_close -= close_qty

            if lot["qty"] <= 1e-12:
                lots_by_symbol[sym].popleft()

    today_utc = datetime.now(timezone.utc).date()
    closed_today = [t for t in closed_trades if getattr(t["exit_ts"], "date", lambda: None)() == today_utc]

    wins_today = [t for t in closed_today if (t["pnl_usd"] or 0) > 0]
    losses_today = [t for t in closed_today if (t["pnl_usd"] or 0) < 0]
    flats_today = [t for t in closed_today if abs((t["pnl_usd"] or 0)) <= 1e-12]

    realized_pnl_today = sum((t["pnl_usd"] or 0) for t in closed_today)
    avg_win_usd = (sum(t["pnl_usd"] for t in wins_today) / len(wins_today)) if wins_today else None
    avg_loss_usd = (sum(t["pnl_usd"] for t in losses_today) / len(losses_today)) if losses_today else None

    last_closed_trade = closed_trades[-1] if closed_trades else None

    return {
        "closed_trades": closed_trades,
        "closed_today": closed_today,
        "completed_round_trips_today": len(closed_today),
        "wins_today": len(wins_today),
        "losses_today": len(losses_today),
        "flats_today": len(flats_today),
        "realized_pnl_today": realized_pnl_today,
        "avg_win_usd_today": avg_win_usd,
        "avg_loss_usd_today": avg_loss_usd,
        "last_closed_trade": last_closed_trade,
    }


def _build_aggregate_performance_summary(realized_summary: Dict[str, Any]) -> Dict[str, Any]:
    closed_trades = realized_summary.get("closed_trades", []) or []

    total_closed = len(closed_trades)
    wins = [t for t in closed_trades if (t["pnl_usd"] or 0) > 0]
    losses = [t for t in closed_trades if (t["pnl_usd"] or 0) < 0]
    flats = [t for t in closed_trades if abs((t["pnl_usd"] or 0)) <= 1e-12]

    total_realized_pnl = sum((t["pnl_usd"] or 0) for t in closed_trades)
    avg_pnl_per_trade = (total_realized_pnl / total_closed) if total_closed else None
    avg_win_usd = (sum(t["pnl_usd"] for t in wins) / len(wins)) if wins else None
    avg_loss_usd = (sum(t["pnl_usd"] for t in losses) / len(losses)) if losses else None
    win_rate = (len(wins) / total_closed * 100.0) if total_closed else None

    best_trade = max(closed_trades, key=lambda t: t["pnl_usd"]) if closed_trades else None
    worst_trade = min(closed_trades, key=lambda t: t["pnl_usd"]) if closed_trades else None

    return {
        "total_closed_trades": total_closed,
        "wins": len(wins),
        "losses": len(losses),
        "flats": len(flats),
        "win_rate": win_rate,
        "total_realized_pnl": total_realized_pnl,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }


def _classify_day(
    signal_summary: Dict[str, Any],
    today_order_summary: Dict[str, Any],
    flags: Dict[str, str],
    live_positions: List[Any],
    live_open_orders: List[Any],
) -> str:
    funnel = _build_execution_funnel(signal_summary, today_order_summary)
    guard_status = flags.get("GUARD_STATUS", "")
    trading_paused = flags.get("TRADING_PAUSED", "")

    if trading_paused == "1":
        return "paused_day"
    if live_positions and not live_open_orders:
        return "unprotected_position_day"
    if funnel["filled_total"] > 0:
        return "trade_day"
    if funnel["total_signals"] > 0 and funnel["eligible_entry_signals_today"] == 0:
        return "no_eligible_entry_signal_day"
    if funnel["eligible_entry_signals_today"] > 0 and funnel["executed_entry_fills"] == 0:
        return "blocked_entry_signal_day"
    if funnel["total_signals"] == 0:
        return "no_signal_day"
    if guard_status.startswith("NO_DATA"):
        return "guard_warmup_day"
    return "quiet_day"


def _build_verdict(
    flags: Dict[str, str],
    live_positions: List[Any],
    live_open_orders: List[Any],
    latest_snapshot: Optional[Tuple],
    block_summary: Dict[str, Any],
    signal_summary: Dict[str, Any],
) -> Tuple[str, str]:
    guard_status = flags.get("GUARD_STATUS", "")
    trading_paused = flags.get("TRADING_PAUSED", "")
    paused_reason = flags.get("TRADING_PAUSED_REASON", "")
    daytrade_count = None
    if latest_snapshot is not None:
        daytrade_count = latest_snapshot[7]

    funnel = block_summary["funnel"]
    fresh_window_seconds = int(signal_summary.get("fresh_window_seconds") or 0)

    if trading_paused == "1":
        return "ACTION NEEDED", f"TRADING_PAUSED=1 reason={paused_reason or '-'}"

    if live_positions and not live_open_orders:
        syms = ",".join(sorted({str(getattr(p, 'symbol', '?')) for p in live_positions}))
        return "ACTION NEEDED", f"open position(s) without open exit order(s): {syms}"

    if isinstance(guard_status, str) and guard_status.startswith("ERROR"):
        return "WARNING", f"guard reports ERROR: {guard_status}"

    if isinstance(guard_status, str) and guard_status.startswith("NO_DATA"):
        return "WARNING", f"guard reports NO_DATA: {guard_status}"

    if daytrade_count is not None and int(daytrade_count or 0) >= 3:
        return "WARNING", f"PDT/daytrade guard threshold reached: daytrade_count={daytrade_count}"

    blocking_reasons = [reason for reason, _count in (block_summary.get("reason_counts") or []) if reason]
    only_market_closed = bool(blocking_reasons) and set(blocking_reasons) == {"market_closed"}

    no_live_exposure = (not live_positions) and (not live_open_orders)
    no_entry_action_required = (
        int(funnel.get("eligible_entry_signals_today", 0) or 0) == 0
        and int(funnel.get("eligible_fresh_entry_signals_now", 0) or 0) == 0
        and int(funnel.get("executed_entry_fills", 0) or 0) == 0
    )

    successful_flat_trade_day = (
        no_live_exposure
        and int(funnel.get("executed_entry_fills", 0) or 0) > 0
        and int(funnel.get("executed_sell_fills", 0) or 0) > 0
    )

    if successful_flat_trade_day:
        if not blocking_reasons:
            return "OK", "trade executed and position closed successfully"
        if set(blocking_reasons).issubset(BENIGN_POST_TRADE_BLOCK_REASONS):
            return "OK", "trade executed and position closed; remaining block reasons are expected post-trade/market-state gates"

    if only_market_closed and no_live_exposure and no_entry_action_required:
        return "OK", "market closed; no eligible entry action required"

    if block_summary.get("top_reason"):
        return "WARNING", f"eligible entry signals blocked: {block_summary.get('top_reason')}"

    if funnel.get("blocked_eligible", 0) > 0 and block_summary.get("state_top_reason") == "no_signal":
        if funnel.get("eligible_fresh_entry_signals_now", 0) == 0:
            return (
                "WARNING",
                f"no fresh eligible entry signal now (PICK_TTL_SECONDS={fresh_window_seconds}); earlier eligible entry signal(s) today were not filled",
            )
        return "WARNING", "executor idle/no_signal while entry-capable signals remain"

    if block_summary.get("state_top_reason") == "no_signal":
        return "OK", f"executor idle: no fresh eligible entry signal now (PICK_TTL_SECONDS={fresh_window_seconds})"

    return "OK", "system looks healthy"


def _build_operator_summary(
    latest_snapshot: Optional[Tuple],
    flags: Dict[str, str],
    block_summary: Dict[str, Any],
    realized_summary: Dict[str, Any],
    verdict: str,
    verdict_detail: str,
) -> List[str]:
    lines: List[str] = []
    funnel = block_summary["funnel"]

    daytrade_count = latest_snapshot[7] if latest_snapshot is not None else None
    lines.append(
        f"daytrade_count={daytrade_count} | "
        f"signals={funnel['total_signals']} | eligible_entry_today={funnel['eligible_entry_signals_today']} | "
        f"eligible_fresh_entry_now={funnel['eligible_fresh_entry_signals_now']} | "
        f"entry_fills={funnel['executed_entry_fills']} | blocked_eligible={funnel['blocked_eligible']}"
    )
    lines.append(
        f"round_trips_today={realized_summary['completed_round_trips_today']} | "
        f"realized_pnl_today={_fmt_num(realized_summary['realized_pnl_today'])} | "
        f"wins={realized_summary['wins_today']} | losses={realized_summary['losses_today']}"
    )
    lines.append(f"guard_status={flags.get('GUARD_STATUS', '')} | trading_paused={flags.get('TRADING_PAUSED', '')}")
    lines.append(f"status={verdict} | detail={verdict_detail}")
    return lines


def main() -> None:
    days = _env_int("REPORT_DAYS", 14)
    engine = create_engine(_get_db_url())

    with engine.connect() as con:
        latest = _fetch_latest_snapshot(con)
        daily = _fetch_daily_equity(con, days)
        today_order_rows = _fetch_today_order_rows(con)
        today_order_summary = _summarize_order_rows(today_order_rows)
        recent_orders = _fetch_recent_order_rows(con, days, limit_rows=20)
        filled_rows = _fetch_filled_order_rows(con, days)
        flags = _fetch_system_flags(con)
        signal_summary = _fetch_signal_summary(con)
        audit_block_summary = _fetch_execution_audit_block_summary(con)

    live = _fetch_live_state()
    live_account = live["account"]
    live_positions = live["positions"]
    live_open_orders = live["open_orders"]
    live_error = live["error"]

    heuristic_block_summary = _infer_block_summary(
        signal_summary=signal_summary,
        today_order_summary=today_order_summary,
        latest_snapshot=latest,
        live_positions=live_positions,
        live_open_orders=live_open_orders,
        flags=flags,
    )
    block_summary = _merge_block_summaries(
        heuristic_summary=heuristic_block_summary,
        audit_summary=audit_block_summary,
    )

    funnel = block_summary["funnel"]
    realized_summary = _build_realized_trade_summary(filled_rows)
    aggregate_summary = _build_aggregate_performance_summary(realized_summary)
    verdict, verdict_detail = _build_verdict(
        flags=flags,
        live_positions=live_positions,
        live_open_orders=live_open_orders,
        latest_snapshot=latest,
        block_summary=block_summary,
        signal_summary=signal_summary,
    )
    day_class = _classify_day(
        signal_summary=signal_summary,
        today_order_summary=today_order_summary,
        flags=flags,
        live_positions=live_positions,
        live_open_orders=live_open_orders,
    )
    operator_summary = _build_operator_summary(
        latest_snapshot=latest,
        flags=flags,
        block_summary=block_summary,
        realized_summary=realized_summary,
        verdict=verdict,
        verdict_detail=verdict_detail,
    )

    now_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 100)
    print(f"DAILY REPORT V2.8 (UTC) | last {days} days | generated_at_utc={now_utc}")
    print("=" * 100)

    print("\n[0] OPERATOR SUMMARY")
    for line in operator_summary:
        print(f"  {line}")

    print("\n[1] ACCOUNT SNAPSHOT")
    if latest is None:
        print("  latest_db_snapshot: (none)")
    else:
        print(f"  latest_db_ts={latest[0]}")
        print(f"  equity={_fmt_num(latest[1])} | cash={_fmt_num(latest[2])} | buying_power={_fmt_num(latest[3])}")
        print(f"  portfolio_value={_fmt_num(latest[4])} | long_mv={_fmt_num(latest[5])} | short_mv={_fmt_num(latest[6])}")
        print(f"  daytrade_count={latest[7]} | dtbp={_fmt_num(latest[8])} | account_status={latest[9]}")

    if live_account is not None:
        print("  live_account:")
        print(
            f"    status={getattr(live_account, 'status', None)}"
            f" | trading_blocked={getattr(live_account, 'trading_blocked', None)}"
            f" | account_blocked={getattr(live_account, 'account_blocked', None)}"
            f" | pattern_day_trader={getattr(live_account, 'pattern_day_trader', None)}"
            f" | daytrade_count={getattr(live_account, 'daytrade_count', None)}"
            f" | daytrading_buying_power={getattr(live_account, 'daytrading_buying_power', None)}"
        )
    else:
        print(f"  live_account: unavailable ({live_error or 'unknown'})")

    print("\n[2] GUARD / SYSTEM FLAGS")
    if not flags:
        print("  (no flags)")
    else:
        wanted = [
            "GUARD_STATUS",
            "GUARD_LAST_DAY",
            "GUARD_LAST_EQUITY_OPEN",
            "GUARD_LAST_EQUITY_LAST",
            "GUARD_LAST_PNL_USD",
            "GUARD_LAST_PNL_PCT",
            "GUARD_HEARTBEAT_TS",
            "TRADING_PAUSED",
            "TRADING_PAUSED_REASON",
        ]
        for k in wanted:
            print(f"  {k}={flags.get(k, '')}")

    print("\n[3] DAILY EQUITY SUMMARY")
    if not daily:
        print("  (no rows)")
    else:
        for r in daily:
            print(
                f"  {r[0]} | open={_fmt_num(r[1])} | close={_fmt_num(r[2])} | "
                f"pnl={_fmt_num(r[3])} | pnl_pct={_fmt_pct(r[4])} | samples=({r[5]} .. {r[6]})"
            )

    print("\n[4] TODAY ORDER SUMMARY (from alpaca_orders by recorded_at UTC day)")
    print(
        f"  filled_total={today_order_summary['filled_total']} | "
        f"filled_buys={today_order_summary['filled_buys']} | "
        f"filled_sells={today_order_summary['filled_sells']} | "
        f"canceled_total={today_order_summary['canceled_total']} | "
        f"symbols_total={today_order_summary['symbols_total']}"
    )
    print(
        f"  buy_notional={_fmt_num(today_order_summary['buy_notional'])} | "
        f"sell_notional={_fmt_num(today_order_summary['sell_notional'])} | "
        f"net_cashflow={_fmt_num(today_order_summary['net_cashflow'])}"
    )

    print("\n[5] SIGNAL SUMMARY")
    if not signal_summary.get("available"):
        print(f"  unavailable: {signal_summary.get('reason', 'unknown')}")
    else:
        print(
            f"  total_signals_today={signal_summary.get('total_signals_today')} | "
            f"buy_signals_today={signal_summary.get('buy_signals_today')} | "
            f"sell_signals_today={signal_summary.get('sell_signals_today')} | "
            f"symbols_today={signal_summary.get('symbols_today')}"
        )
        print(
            f"  eligible_signals_today={signal_summary.get('eligible_signals_today')} | "
            f"eligible_buy_signals_today={signal_summary.get('eligible_buy_signals_today')} | "
            f"eligible_sell_signals_today={signal_summary.get('eligible_sell_signals_today')}"
        )
        print(
            f"  eligible_fresh_signals_now={signal_summary.get('eligible_fresh_signals_now')} | "
            f"eligible_fresh_buy_signals_now={signal_summary.get('eligible_fresh_buy_signals_now')} | "
            f"eligible_fresh_sell_signals_now={signal_summary.get('eligible_fresh_sell_signals_now')} | "
            f"PICK_TTL_SECONDS={signal_summary.get('fresh_window_seconds')}"
        )
        print(
            f"  detected_columns: time={signal_summary.get('time_col')} | "
            f"symbol={signal_summary.get('symbol_col')} | side={signal_summary.get('side_col')} | "
            f"strength={signal_summary.get('strength_col')} | source={signal_summary.get('source_col')} | "
            f"portfolio={signal_summary.get('portfolio_col')}"
        )

    print("\n[6] EXECUTION FUNNEL")
    print(
        f"  entry_mode={funnel['entry_mode']} | total_signals={funnel['total_signals']} | "
        f"eligible_signals_today={funnel['eligible_signals']} | eligible_entry_signals_today={funnel['eligible_entry_signals_today']}"
    )
    print(
        f"  eligible_fresh_signals_now={funnel['eligible_fresh_signals_now']} | "
        f"eligible_fresh_entry_signals_now={funnel['eligible_fresh_entry_signals_now']} | "
        f"executed_entry_fills={funnel['executed_entry_fills']}"
    )
    print(
        f"  buy_signals={funnel['buy_signals']} | sell_signals={funnel['sell_signals']} | "
        f"eligible_buy_signals={funnel['eligible_buy_signals']} | eligible_sell_signals={funnel['eligible_sell_signals']}"
    )
    print(
        f"  executed_buy_fills={funnel['executed_buy_fills']} | executed_sell_fills={funnel['executed_sell_fills']} | "
        f"blocked_eligible={funnel['blocked_eligible']} | blocked_eligible_buys={funnel['blocked_eligible_buys']}"
    )

    print("\n[7] BLOCK SUMMARY")
    print(f"  source={block_summary.get('source')}")
    print(f"  blocked_reason_rows_today={block_summary.get('total_blocked', 0)}")
    if block_summary.get("reason_counts"):
        print(f"  top_reason={block_summary.get('top_reason')}")
        for reason, count in block_summary.get("reason_counts", []):
            print(f"  - {reason}: {count}")
    else:
        print("  (no blocking reasons recorded today)")

    if block_summary.get("state_reason_counts"):
        print("  executor_state_rows_today=" + str(block_summary.get("state_total", 0)))
        print(f"  executor_state_top_reason={block_summary.get('state_top_reason')}")
        for reason, count in block_summary.get("state_reason_counts", []):
            print(f"  - state:{reason}: {count}")

    print("\n[8] REALIZED TRADE SUMMARY")
    print(
        f"  completed_round_trips_today={realized_summary['completed_round_trips_today']} | "
        f"realized_pnl_today={_fmt_num(realized_summary['realized_pnl_today'])} | "
        f"wins_today={realized_summary['wins_today']} | "
        f"losses_today={realized_summary['losses_today']} | "
        f"flats_today={realized_summary['flats_today']}"
    )
    print(
        f"  avg_win_usd_today={_fmt_num(realized_summary['avg_win_usd_today'])} | "
        f"avg_loss_usd_today={_fmt_num(realized_summary['avg_loss_usd_today'])}"
    )
    last_closed = realized_summary["last_closed_trade"]
    if last_closed:
        print(
            f"  last_closed_trade={last_closed['symbol']} | qty={_fmt_num(last_closed['qty'], 6)} | "
            f"entry_px={_fmt_num(last_closed['entry_px'], 4)} | exit_px={_fmt_num(last_closed['exit_px'], 4)} | "
            f"pnl_usd={_fmt_num(last_closed['pnl_usd'], 4)} | pnl_pct={_fmt_pct(last_closed['pnl_pct'], 4)} | "
            f"exit_ts={last_closed['exit_ts']}"
        )
    else:
        print("  last_closed_trade=(none)")

    print("\n[9] AGGREGATE PERFORMANCE SUMMARY")
    print(
        f"  total_closed_trades={aggregate_summary['total_closed_trades']} | "
        f"wins={aggregate_summary['wins']} | losses={aggregate_summary['losses']} | flats={aggregate_summary['flats']}"
    )
    print(
        f"  win_rate={_fmt_pct(aggregate_summary['win_rate'])} | "
        f"total_realized_pnl={_fmt_num(aggregate_summary['total_realized_pnl'], 4)} | "
        f"avg_pnl_per_trade={_fmt_num(aggregate_summary['avg_pnl_per_trade'], 4)}"
    )
    print(
        f"  avg_win_usd={_fmt_num(aggregate_summary['avg_win_usd'], 4)} | "
        f"avg_loss_usd={_fmt_num(aggregate_summary['avg_loss_usd'], 4)}"
    )
    best_trade = aggregate_summary["best_trade"]
    worst_trade = aggregate_summary["worst_trade"]
    if best_trade:
        print(
            f"  best_trade={best_trade['symbol']} | exit_ts={best_trade['exit_ts']} | "
            f"pnl_usd={_fmt_num(best_trade['pnl_usd'], 4)} | pnl_pct={_fmt_pct(best_trade['pnl_pct'], 4)}"
        )
    else:
        print("  best_trade=(none)")
    if worst_trade:
        print(
            f"  worst_trade={worst_trade['symbol']} | exit_ts={worst_trade['exit_ts']} | "
            f"pnl_usd={_fmt_num(worst_trade['pnl_usd'], 4)} | pnl_pct={_fmt_pct(worst_trade['pnl_pct'], 4)}"
        )
    else:
        print("  worst_trade=(none)")

    if signal_summary.get("available"):
        recent_signal_rows = signal_summary.get("recent_rows") or []
        print("\n[10] RECENT SIGNAL ROWS (last 20 today)")
        if not recent_signal_rows:
            print("  (no recent signal rows today)")
        else:
            for r in recent_signal_rows:
                print("  " + " | ".join(str(x) for x in r))

    print("\n[11] LIVE POSITIONS")
    if not live_positions:
        print("  (no live positions)")
    else:
        for p in live_positions:
            print(
                f"  {getattr(p, 'symbol', '?')} | qty={getattr(p, 'qty', '?')} | "
                f"avg_entry={getattr(p, 'avg_entry_price', '?')} | current={getattr(p, 'current_price', '?')} | "
                f"uPnL={getattr(p, 'unrealized_pl', '?')} | uPnL%={getattr(p, 'unrealized_plpc', '?')}"
            )

    print("\n[12] LIVE OPEN ORDERS")
    if not live_open_orders:
        print("  (no live open orders)")
    else:
        for o in live_open_orders:
            print(
                f"  {getattr(o, 'symbol', '?')} | side={getattr(o, 'side', '?')} | "
                f"type={getattr(o, 'type', '?')} | status={getattr(o, 'status', '?')} | "
                f"tif={getattr(o, 'time_in_force', '?')} | stop={getattr(o, 'stop_price', None)} | "
                f"limit={getattr(o, 'limit_price', None)} | cid={getattr(o, 'client_order_id', '')}"
            )

    print("\n[13] RECENT CLOSED TRADES (last 20)")
    closed_trades = realized_summary["closed_trades"][-20:]
    if not closed_trades:
        print("  (no recent closed trades)")
    else:
        for t in reversed(closed_trades):
            print(
                f"  {t['exit_ts']} | {t['symbol']} | qty={_fmt_num(t['qty'], 6)} | "
                f"entry_px={_fmt_num(t['entry_px'], 4)} | exit_px={_fmt_num(t['exit_px'], 4)} | "
                f"pnl_usd={_fmt_num(t['pnl_usd'], 4)} | pnl_pct={_fmt_pct(t['pnl_pct'], 4)} | "
                f"exit_type={t['exit_type']}"
            )

    print("\n[14] RECENT ORDER ROWS (last 20 from alpaca_orders)")
    if not recent_orders:
        print("  (no recent orders)")
    else:
        for r in recent_orders:
            print(
                f"  {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
                f"qty={r[5]} | filled_qty={r[6]} | fill_px={r[7]} | notional={r[8]} | cid={r[9]}"
            )

    print("\n[15] DAY CLASSIFICATION")
    print(f"  class={day_class}")

    print("\n[16] FINAL VERDICT")
    print(f"  status={verdict}")
    print(f"  detail={verdict_detail}")

    print("\nNotes:")
    print("  - Equity by day is based on equity_snapshots.")
    print("  - Today order summary uses Python-side normalization for OrderStatus.* / OrderSide.* values.")
    print("  - BLOCK SUMMARY uses blocking audit reasons when available and keeps non-blocking executor state separately.")
    print("  - In LONG_ONLY mode, entry funnel is based on eligible buy-entry signals, not sell signals.")
    print("  - Fresh signal counts use PICK_TTL_SECONDS to reflect what the picker could act on right now.")
    print("  - FINAL VERDICT treats market_closed as OK when there is no live exposure and no eligible entry action required.")
    print("  - FINAL VERDICT also treats successful flat trade days as OK when remaining block reasons are only benign market-state/post-trade gates.")
    print("  - Realized trade summary sorts same-timestamp filled rows deterministically so BUY lots are matched before SELL exits.")
    print("  - Realized trade summary is FIFO-based approximation from filled alpaca_orders rows.")
    print("  - Aggregate performance summary is computed from the same FIFO-matched closed trades.")
    print("  - Live positions/open orders come from Alpaca API when credentials are available.")
    print("  - If status=WARNING or ACTION NEEDED, inspect positions/open orders/flags first.")
    print("=" * 100)


if __name__ == "__main__":
    main()