"""
tools/show_recent_trades.py

Rýchly prehľad posledných signálov zo signals tabuľky.
Spúšťanie:

  # lokálne:
  PYTHONPATH=. python -m tools.show_recent_trades

  # v kontajneri:
  docker compose exec signal_executor bash -lc "python -m tools.show_recent_trades --limit 30"
"""

import os
import sys
import argparse
from datetime import timezone

import psycopg2
import psycopg2.extras


def normalize_db_url_for_psycopg2(url: str) -> str:
    """
    Ak je URL vo forme SQLAlchemy (postgresql+psycopg2://),
    pre psycopg2 ju musíme zmeniť na postgresql://
    """
    if not url:
        return url

    # Najčastejší prípad: postgresql+psycopg2://...
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql://", 1)

    # Generalizovane: stripni čokoľvek po "+"
    # napr. "postgresql+asyncpg://..."
    if url.startswith("postgresql+") and "://" in url:
        scheme, rest = url.split("://", 1)
        base_scheme = scheme.split("+", 1)[0]  # "postgresql"
        return f"{base_scheme}://{rest}"

    return url


def get_db_url() -> str:
    """
    Vráti správne DB URL podľa projektu:
    1) Preferuje DATABASE_URL (náš globálny štandard v projekte)
    2) Potom DB_URL, ak existuje
    3) Potom fallback PGHOST / PGUSER / PGPASSWORD / PGDATABASE
    4) Nakoniec úplný docker fallback
    """

    # HLAVNÁ PREMENNÁ – používame v celom projekte (SQLAlchemy štýl)
    url = os.getenv("DATABASE_URL")
    if url:
        return normalize_db_url_for_psycopg2(url)

    # sekundárna možnosť
    url2 = os.getenv("DB_URL")
    if url2:
        return normalize_db_url_for_psycopg2(url2)

    # fallback pre docker compose
    host = os.getenv("PGHOST", "postgres")
    user = os.getenv("PGUSER", "postgres")
    pwd = os.getenv("PGPASSWORD", "postgres")
    db = os.getenv("PGDATABASE", "trader")

    fallback = f"postgresql://{user}:{pwd}@{host}:5432/{db}"
    return fallback


def fetch_recent_signals(limit: int = 20):
    db_url = get_db_url()
    # pre debug môžeš si odkomentovať:
    # print(f"[DEBUG] Using DB URL for psycopg2: {db_url}", file=sys.stderr)

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    created_at AT TIME ZONE 'UTC' AS created_at_utc,
                    symbol,
                    side,
                    strength,
                    status,
                    error,
                    order_id,
                    client_order_id,
                    portfolio_id
                FROM signals
                ORDER BY id DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return rows


def print_table(rows):
    if not rows:
        print("No signals found.")
        return

    headers = [
        "id",
        "created_at_utc",
        "symbol",
        "side",
        "strength",
        "status",
        "order_id",
        "client_order_id",
        "error_short",
    ]

    data = []
    for r in rows:
        error = r["error"] or ""
        if len(error) > 60:
            error_short = error[:57] + "..."
        else:
            error_short = error

        created = r["created_at_utc"]
        if created is not None:
            created_str = created.replace(tzinfo=timezone.utc).isoformat()
        else:
            created_str = ""

        status = r["status"] if r["status"] is not None else ""

        data.append(
            [
                r["id"],
                created_str,
                r["symbol"],
                r["side"],
                float(r["strength"]) if r["strength"] is not None else None,
                status,
                str(r["order_id"]) if r["order_id"] else "",
                str(r["client_order_id"]) if r["client_order_id"] else "",
                error_short,
            ]
        )

    col_widths = [
        max(len(str(x)) for x in [h] + [row[i] for row in data])
        for i, h in enumerate(headers)
    ]

    def fmt_row(row_vals):
        return " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row_vals))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in data:
        print(fmt_row(row))
    print()


def print_summary(rows):
    total = len(rows)
    by_status = {}

    for r in rows:
        st_raw = r["status"]
        # None -> "None" kvôli prehľadnosti
        st = st_raw if st_raw is not None else "None"
        by_status[st] = by_status.get(st, 0) + 1

    print("Summary:")
    print(f"  Total signals in this view: {total}")
    for status, cnt in sorted(by_status.items(), key=lambda x: x[0]):
        status_str = str(status)
        # formátuj ako text (šírka 10)
        print(f"  {status_str:10s}: {cnt}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show recent signals/trades from DB.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent signals to show (default: 20)",
    )
    args = parser.parse_args(argv)

    print(f"[show_recent_trades] Fetching last {args.limit} signals...")
    rows = fetch_recent_signals(limit=args.limit)
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
