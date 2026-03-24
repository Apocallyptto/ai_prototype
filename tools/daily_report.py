import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, text

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
except Exception:
    TradingClient = None
    QueryOrderStatus = None
    GetOrdersRequest = None


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


def _fetch_today_order_summary(con) -> Optional[Tuple]:
    q = text("""
    SELECT
      COUNT(*) FILTER (WHERE LOWER(status) = 'filled') AS filled_total,
      COUNT(*) FILTER (WHERE LOWER(status) = 'filled' AND LOWER(side) = 'buy') AS filled_buys,
      COUNT(*) FILTER (WHERE LOWER(status) = 'filled' AND LOWER(side) = 'sell') AS filled_sells,
      COUNT(*) FILTER (WHERE LOWER(status) = 'canceled') AS canceled_total,
      COUNT(DISTINCT symbol) FILTER (WHERE symbol IS NOT NULL AND symbol <> '') AS symbols_total,
      COALESCE(SUM(CASE WHEN LOWER(status) = 'filled' AND LOWER(side) = 'buy'  THEN COALESCE(notional, 0) ELSE 0 END), 0) AS buy_notional,
      COALESCE(SUM(CASE WHEN LOWER(status) = 'filled' AND LOWER(side) = 'sell' THEN COALESCE(notional, 0) ELSE 0 END), 0) AS sell_notional
    FROM alpaca_orders
    WHERE recorded_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC')
      AND recorded_at <  date_trunc('day', NOW() AT TIME ZONE 'UTC') + interval '1 day'
    """)
    return con.execute(q).fetchone()


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
        "total_signals_today": None,
        "eligible_signals_today": None,
        "symbols_today": None,
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
    else:
        result["eligible_signals_today"] = result["total_signals_today"]

    if symbol_col:
        q_symbols = text(f"""
        SELECT COUNT(DISTINCT {symbol_col}) AS n
        FROM signals
        WHERE {where_sql}
        """)
        result["symbols_today"] = int(con.execute(q_symbols, params).scalar() or 0)

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
    rows = con.execute(q_recent, params).fetchall()
    result["recent_rows"] = rows

    return result


def _infer_block_summary(
    signal_summary: Dict[str, Any],
    today_orders: Optional[Tuple],
    latest_snapshot: Optional[Tuple],
    live_positions: List[Any],
    live_open_orders: List[Any],
    flags: Dict[str, str],
) -> Dict[str, Any]:
    filled_buys = int((today_orders[1] if today_orders is not None and today_orders[1] is not None else 0) or 0)
    total_signals = signal_summary.get("total_signals_today")
    eligible_signals = signal_summary.get("eligible_signals_today")

    if total_signals is None:
        total_signals = 0
    if eligible_signals is None:
        eligible_signals = 0

    blocked_estimate = max(int(eligible_signals) - int(filled_buys), 0)

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

    if trading_paused == "1":
        top_reason = f"trading_paused ({paused_reason or 'no reason'})"
    elif daytrade_count is not None and daytrade_count >= pause_on_daytrade_ge:
        top_reason = f"PDT/daytrade gate (daytrade_count={daytrade_count}, threshold={pause_on_daytrade_ge})"
    elif total_signals == 0:
        top_reason = "no signals generated today"
    elif total_signals > 0 and eligible_signals == 0:
        top_reason = f"signals below threshold (MIN_STRENGTH={min_strength:.4f})"
    elif len(live_positions) >= max_open_positions and eligible_signals > filled_buys:
        top_reason = f"max_open_positions gate (positions={len(live_positions)}, limit={max_open_positions})"
    elif len(live_open_orders) >= max_open_orders and eligible_signals > filled_buys:
        top_reason = f"max_open_orders gate (open_orders={len(live_open_orders)}, limit={max_open_orders})"
    elif filled_buys > 0 and blocked_estimate == 0:
        top_reason = "all eligible signals executed"
    elif filled_buys > 0 and blocked_estimate > 0:
        top_reason = "partial execution; some eligible signals likely blocked by other gates/cooldowns"
    elif guard_status.startswith("NO_DATA"):
        top_reason = f"guard not ready at day start ({guard_status})"
    else:
        top_reason = "other executor gate / cooldown / market-state block"

    return {
        "filled_buys_today": filled_buys,
        "blocked_estimate": blocked_estimate,
        "top_reason": top_reason,
    }


def _build_verdict(
    flags: Dict[str, str],
    live_positions: List[Any],
    live_open_orders: List[Any],
    latest_snapshot: Optional[Tuple],
    block_summary: Dict[str, Any],
) -> Tuple[str, str]:
    guard_status = flags.get("GUARD_STATUS", "")
    trading_paused = flags.get("TRADING_PAUSED", "")
    paused_reason = flags.get("TRADING_PAUSED_REASON", "")
    daytrade_count = None
    if latest_snapshot is not None:
        daytrade_count = latest_snapshot[7]

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

    if block_summary.get("blocked_estimate", 0) > 0:
        return "WARNING", f"signals likely blocked: {block_summary.get('top_reason')}"

    return "OK", "system looks healthy"


def main() -> None:
    days = _env_int("REPORT_DAYS", 14)
    engine = create_engine(_get_db_url())

    with engine.connect() as con:
        latest = _fetch_latest_snapshot(con)
        daily = _fetch_daily_equity(con, days)
        today_orders = _fetch_today_order_summary(con)
        recent_orders = _fetch_recent_order_rows(con, days, limit_rows=20)
        flags = _fetch_system_flags(con)
        signal_summary = _fetch_signal_summary(con)

    live = _fetch_live_state()
    live_account = live["account"]
    live_positions = live["positions"]
    live_open_orders = live["open_orders"]
    live_error = live["error"]

    block_summary = _infer_block_summary(
        signal_summary=signal_summary,
        today_orders=today_orders,
        latest_snapshot=latest,
        live_positions=live_positions,
        live_open_orders=live_open_orders,
        flags=flags,
    )
    verdict, verdict_detail = _build_verdict(
        flags=flags,
        live_positions=live_positions,
        live_open_orders=live_open_orders,
        latest_snapshot=latest,
        block_summary=block_summary,
    )

    now_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 100)
    print(f"DAILY REPORT V2.1 (UTC) | last {days} days | generated_at_utc={now_utc}")
    print("=" * 100)

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
    if today_orders is None:
        print("  (no summary)")
    else:
        print(
            f"  filled_total={today_orders[0]} | filled_buys={today_orders[1]} | "
            f"filled_sells={today_orders[2]} | canceled_total={today_orders[3]} | "
            f"symbols_total={today_orders[4]}"
        )
        print(
            f"  buy_notional={_fmt_num(today_orders[5])} | "
            f"sell_notional={_fmt_num(today_orders[6])} | "
            f"net_cashflow={_fmt_num((today_orders[6] or 0) - (today_orders[5] or 0))}"
        )

    print("\n[5] SIGNAL SUMMARY")
    if not signal_summary.get("available"):
        print(f"  unavailable: {signal_summary.get('reason', 'unknown')}")
    else:
        print(
            f"  total_signals_today={signal_summary.get('total_signals_today')} | "
            f"eligible_signals_today={signal_summary.get('eligible_signals_today')} | "
            f"symbols_today={signal_summary.get('symbols_today')}"
        )
        print(
            f"  detected_columns: time={signal_summary.get('time_col')} | "
            f"symbol={signal_summary.get('symbol_col')} | side={signal_summary.get('side_col')} | "
            f"strength={signal_summary.get('strength_col')} | source={signal_summary.get('source_col')} | "
            f"portfolio={signal_summary.get('portfolio_col')}"
        )

    print("\n[6] BLOCK SUMMARY (heuristic)")
    print(
        f"  filled_buys_today={block_summary.get('filled_buys_today')} | "
        f"blocked_estimate={block_summary.get('blocked_estimate')}"
    )
    print(f"  top_reason={block_summary.get('top_reason')}")

    if signal_summary.get("available"):
        recent_signal_rows = signal_summary.get("recent_rows") or []
        print("\n[7] RECENT SIGNAL ROWS (last 20 today)")
        if not recent_signal_rows:
            print("  (no recent signal rows today)")
        else:
            for r in recent_signal_rows:
                print("  " + " | ".join(str(x) for x in r))

    print("\n[8] LIVE POSITIONS")
    if not live_positions:
        print("  (no live positions)")
    else:
        for p in live_positions:
            print(
                f"  {getattr(p, 'symbol', '?')} | qty={getattr(p, 'qty', '?')} | "
                f"avg_entry={getattr(p, 'avg_entry_price', '?')} | current={getattr(p, 'current_price', '?')} | "
                f"uPnL={getattr(p, 'unrealized_pl', '?')} | uPnL%={getattr(p, 'unrealized_plpc', '?')}"
            )

    print("\n[9] LIVE OPEN ORDERS")
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

    print("\n[10] RECENT ORDER ROWS (last 20 from alpaca_orders)")
    if not recent_orders:
        print("  (no recent orders)")
    else:
        for r in recent_orders:
            print(
                f"  {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
                f"qty={r[5]} | filled_qty={r[6]} | fill_px={r[7]} | notional={r[8]} | cid={r[9]}"
            )

    print("\n[11] FINAL VERDICT")
    print(f"  status={verdict}")
    print(f"  detail={verdict_detail}")

    print("\nNotes:")
    print("  - Equity by day is based on equity_snapshots.")
    print("  - Today order summary is based on alpaca_orders.recorded_at in UTC day.")
    print("  - Signal/block summary is heuristic unless executor block reasons are explicitly persisted.")
    print("  - Live positions/open orders come from Alpaca API when credentials are available.")
    print("  - If status=WARNING or ACTION NEEDED, inspect positions/open orders/flags first.")
    print("=" * 100)


if __name__ == "__main__":
    main()