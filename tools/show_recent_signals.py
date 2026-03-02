import os
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import RealDictCursor


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    return float(v)


def _get_dsn() -> str:
    return os.getenv("DATABASE_URL") or os.getenv("DB_URL") or ""


def _get_symbols() -> list[str]:
    raw = os.getenv("SYMBOLS") or os.getenv("TICKERS") or "AAPL,MSFT"
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _table_cols(cur, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def main() -> None:
    dsn = _get_dsn()
    if not dsn:
        raise RuntimeError("Missing DATABASE_URL / DB_URL env")

    symbols = _get_symbols()
    min_strength = _env_float("MIN_STRENGTH", 0.60)
    window_min = _env_int("WINDOW_MIN", 180)
    limit = _env_int("LIMIT", 200)
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    since_dt = datetime.now(timezone.utc) - timedelta(minutes=window_min)

    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cols = _table_cols(cur, "signals")
        wanted = ["id", "symbol", "side", "strength", "price", "created_at", "portfolio_id", "source"]
        sel = [c for c in wanted if c in cols]
        if not sel:
            raise RuntimeError(f"signals table has unexpected columns: {cols}")

        sql = f"""
            SELECT {", ".join(sel)}
            FROM signals
            WHERE portfolio_id = %s
              AND symbol = ANY(%s)
              AND created_at >= %s
              AND strength >= %s
            ORDER BY created_at DESC
            LIMIT %s
        """

        params = (portfolio_id, symbols, since_dt, min_strength, limit)
        cur.execute(sql, params)
        rows = cur.fetchall()

        print(f"Window: last {window_min}m | portfolio_id={portfolio_id} | symbols={symbols} | min_strength={min_strength}")
        print(f"Columns: {sel}")
        print(f"Rows: {len(rows)}")
        for r in rows[:50]:
            print(r)

        # Show last side per symbol (debug for crossover)
        cur.execute(
            """
            SELECT DISTINCT ON (symbol) symbol, side, strength, created_at
            FROM signals
            WHERE portfolio_id = %s AND symbol = ANY(%s)
            ORDER BY symbol, created_at DESC
            """,
            (portfolio_id, symbols),
        )
        print("\nLast per symbol:")
        for r in cur.fetchall():
            print(r)

        cur.close()
    finally:
        conn.close()


if __name__ == "__main__":
    main()