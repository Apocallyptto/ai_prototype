#!/usr/bin/env python3
"""
Sync local 'signals' table with Alpaca orders.

Schema expected:
  signals(..., created_at, processed_status, processed_note, processed_at, alpaca_order_id, ...)

What it does:
- Repeats every SYNC_POLL seconds (default 30).
- Finds signals where processed_status='submitted'.
- Looks up the Alpaca order by alpaca_order_id (exact).
- If Alpaca order is terminal (filled/canceled/expired/rejected/replaced),
  updates processed_status/processed_note/processed_at.

Env:
  DB_URL (optional; default postgresql://postgres:postgres@postgres:5432/trader)
  ALPACA_API_KEY / ALPACA_API_SECRET (required)
  TRADING_MODE ("paper" or "live"; default "paper")
  ALPACA_PAPER (optional override; "1"/"0")
  LOOKBACK_DAYS (default "30")   # just for order fetching, not for DB selection
  SYNC_POLL (default "30")
"""

import datetime as dt
import os
import time

from sqlalchemy import create_engine, text

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrderByIdRequest
from alpaca.trading.enums import QueryOrderStatus


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _resolve_mode() -> str:
    return (os.getenv("TRADING_MODE") or "paper").strip().lower()


def _resolve_paper() -> bool:
    # Explicit override wins (useful for emergency forcing paper)
    if os.getenv("ALPACA_PAPER") is not None:
        return _env_bool("ALPACA_PAPER", True)
    # Default comes from TRADING_MODE
    return _resolve_mode() != "live"


def _sval(v) -> str:
    if v is None:
        return ""
    return str(v)


def _normalize_db_url() -> str:
    url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or ""
    if not url:
        # default local in compose
        return "postgresql://postgres:postgres@postgres:5432/trader"
    return url


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _is_terminal(status: str) -> bool:
    s = (status or "").lower()
    return s in ("filled", "canceled", "expired", "rejected", "replaced")


def main_loop():
    db_url = _normalize_db_url()
    poll = int(os.getenv("SYNC_POLL", "30"))
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "30"))

    mode = _resolve_mode()
    paper = _resolve_paper()

    api_key = os.getenv("ALPACA_API_KEY") or ""
    api_secret = os.getenv("ALPACA_API_SECRET") or ""
    if not api_key or not api_secret:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    print(f"sync_orders starting | poll={poll}s | lookback_days={lookback_days} | mode={mode} | paper={paper}")

    engine = create_engine(db_url, future=True)
    tc = TradingClient(api_key, api_secret, paper=paper)

    while True:
        try:
            # fetch recent submitted signals
            with engine.begin() as con:
                rows = con.execute(
                    text(
                        """
                        SELECT id, alpaca_order_id, processed_status
                        FROM signals
                        WHERE processed_status = 'submitted'
                        ORDER BY created_at DESC
                        LIMIT 200
                        """
                    )
                ).fetchall()

            if not rows:
                time.sleep(poll)
                continue

            # for each signal, check Alpaca order status
            for (sid, oid, pstatus) in rows:
                oid = _sval(oid).strip()
                if not oid:
                    continue

                try:
                    o = tc.get_order_by_id(GetOrderByIdRequest(order_id=oid))
                except Exception as e:
                    # order might not exist (or auth issue)
                    note = f"alpaca_get_error: {type(e).__name__}: {_sval(e)[:200]}"
                    with engine.begin() as con:
                        con.execute(
                            text(
                                """
                                UPDATE signals
                                SET processed_note = :note,
                                    processed_at = :ts
                                WHERE id = :id
                                """
                            ),
                            {"note": note, "ts": _now_utc(), "id": sid},
                        )
                    continue

                status = _sval(getattr(o, "status", "")).lower()
                if not _is_terminal(status):
                    continue

                # terminal: write it
                note = f"alpaca_terminal:{status}"
                with engine.begin() as con:
                    con.execute(
                        text(
                            """
                            UPDATE signals
                            SET processed_status = :status,
                                processed_note = :note,
                                processed_at = :ts
                            WHERE id = :id
                            """
                        ),
                        {"status": status, "note": note, "ts": _now_utc(), "id": sid},
                    )

        except Exception as e:
            print("sync_orders ERROR:", repr(e))

        time.sleep(poll)


if __name__ == "__main__":
    main_loop()
