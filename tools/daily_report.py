import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    return int(v)


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
        account = tc.get_account()
        out["account"] = account
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


def _build_verdict(
    flags: Dict[str, str],
    live_positions: List[Any],
    live_open_orders: List[Any],
    latest_snapshot: Optional[Tuple],
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

    live = _fetch_live_state()
    live_account = live["account"]
    live_positions = live["positions"]
    live_open_orders = live["open_orders"]
    live_error = live["error"]

    verdict, verdict_detail = _build_verdict(flags, live_positions, live_open_orders, latest)

    now_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 100)
    print(f"DAILY REPORT V2 (UTC) | last {days} days | generated_at_utc={now_utc}")
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

    print("\n[5] LIVE POSITIONS")
    if not live_positions:
        print("  (no live positions)")
    else:
        for p in live_positions:
            print(
                f"  {getattr(p, 'symbol', '?')} | qty={getattr(p, 'qty', '?')} | "
                f"avg_entry={getattr(p, 'avg_entry_price', '?')} | current={getattr(p, 'current_price', '?')} | "
                f"uPnL={getattr(p, 'unrealized_pl', '?')} | uPnL%={getattr(p, 'unrealized_plpc', '?')}"
            )

    print("\n[6] LIVE OPEN ORDERS")
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

    print("\n[7] RECENT ORDER ROWS (last 20 from alpaca_orders)")
    if not recent_orders:
        print("  (no recent orders)")
    else:
        for r in recent_orders:
            print(
                f"  {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
                f"qty={r[5]} | filled_qty={r[6]} | fill_px={r[7]} | notional={r[8]} | cid={r[9]}"
            )

    print("\n[8] FINAL VERDICT")
    print(f"  status={verdict}")
    print(f"  detail={verdict_detail}")

    print("\nNotes:")
    print("  - Equity by day is based on equity_snapshots.")
    print("  - Today order summary is based on alpaca_orders.recorded_at in UTC day.")
    print("  - Live positions/open orders come from Alpaca API when credentials are available.")
    print("  - If status=WARNING or ACTION NEEDED, inspect positions/open orders/flags first.")
    print("=" * 100)


if __name__ == "__main__":
    main()