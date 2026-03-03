import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    return int(v)


def _get_db_url() -> str:
    return os.getenv("DB_URL") or os.getenv("DATABASE_URL") or ""


def main() -> None:
    db = _get_db_url()
    if not db:
        raise RuntimeError("Missing DB_URL / DATABASE_URL")

    days = _env_int("REPORT_DAYS", 14)
    engine = create_engine(db)

    # daily equity: first and last equity per UTC day
    q_daily = text("""
    WITH x AS (
      SELECT
        date_trunc('day', ts) AS day_utc,
        ts,
        equity
      FROM equity_snapshots
      WHERE ts >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
        AND equity IS NOT NULL
    ),
    first_last AS (
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
      round(open_equity::numeric, 2)  AS open_equity,
      round(close_equity::numeric, 2) AS close_equity,
      round((close_equity - open_equity)::numeric, 2) AS pnl,
      CASE WHEN open_equity > 0 THEN round(((close_equity - open_equity)/open_equity*100)::numeric, 2) ELSE NULL END AS pnl_pct,
      first_ts,
      last_ts
    FROM first_last
    ORDER BY day DESC;
    """)

    # today snapshot
    q_latest = text("""
    SELECT ts, equity, cash, daytrade_count
    FROM equity_snapshots
    ORDER BY ts DESC
    LIMIT 1;
    """)

    # current position (latest snapshot per symbol)
    q_pos = text("""
    WITH x AS (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) rn
      FROM position_snapshots
      WHERE ts >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
    )
    SELECT ts, symbol, qty, current_price, unrealized_pl, unrealized_plpc
    FROM x
    WHERE rn=1
    ORDER BY symbol;
    """)

    # closed orders summary last N days (from alpaca_orders)
    q_orders = text("""
    SELECT
      symbol,
      side,
      status,
      count(*) AS n,
      round(sum(COALESCE(notional, 0))::numeric, 2) AS notional_sum
    FROM alpaca_orders
    WHERE recorded_at >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
    GROUP BY symbol, side, status
    ORDER BY symbol, side, status;
    """)

    with engine.connect() as con:
        latest = con.execute(q_latest).fetchone()
        daily = con.execute(q_daily, {"days": days}).fetchall()
        pos = con.execute(q_pos, {"days": days}).fetchall()
        orders = con.execute(q_orders, {"days": days}).fetchall()

    print("=" * 80)
    print(f"DAILY REPORT (UTC) | last {days} days | generated_at_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    if latest:
        print("\nLatest snapshot:")
        print(f"  ts={latest[0]} equity={latest[1]} cash={latest[2]} daytrade_count={latest[3]}")
    else:
        print("\nLatest snapshot: (none)")

    print("\nEquity by day (UTC):")
    if not daily:
        print("  (no rows)")
    else:
        for r in daily:
            # (day, open_equity, close_equity, pnl, pnl_pct, first_ts, last_ts)
            print(f"  {r[0]}  open={r[1]}  close={r[2]}  pnl={r[3]}  pnl%={r[4]}  samples=({r[5]}..{r[6]})")

    print("\nCurrent positions (latest snapshots):")
    if not pos:
        print("  (no positions in snapshots)")
    else:
        for r in pos:
            print(f"  {r[1]} qty={r[2]} px={r[3]} uPnL={r[4]} uPnL%={r[5]} ts={r[0]}")

    print("\nOrders summary (alpaca_orders):")
    if not orders:
        print("  (no orders)")
    else:
        for r in orders:
            print(f"  {r[0]} | {r[1]} | {r[2]} | n={r[3]} | notional_sum={r[4]}")

    print("\nNotes:")
    print("  - Equity PnL is based on snapshots; for more accurate daily close, increase snapshot frequency or align to market close.")
    print("  - With small account + PDT limits, prefer swing holds (>1 day).")
    print("=" * 80)


if __name__ == "__main__":
    main()