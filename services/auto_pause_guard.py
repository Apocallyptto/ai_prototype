import os
import time
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from tools.db import get_engine
from tools.system_flags import set_flag

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
LOG = logging.getLogger("auto_pause_guard")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    return float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    return int(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _today_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def main():
    poll = _env_int("GUARD_POLL_SECONDS", 60)
    max_daily_loss_usd = _env_float("GUARD_MAX_DAILY_LOSS_USD", 2.0)
    max_daily_loss_pct = _env_float("GUARD_MAX_DAILY_LOSS_PCT", 2.0)
    enabled = _env_bool("GUARD_ENABLED", True)

    engine = get_engine()

    LOG.info(
        "auto_pause_guard starting | enabled=%s poll=%ss max_loss_usd=%.2f max_loss_pct=%.2f",
        enabled, poll, max_daily_loss_usd, max_daily_loss_pct
    )

    while True:
        try:
            if not enabled:
                time.sleep(poll)
                continue

            day = _today_utc_date()

            # open equity = first snapshot of day, close equity = latest snapshot
            q = text("""
            WITH x AS (
              SELECT ts, equity
              FROM equity_snapshots
              WHERE ts >= date_trunc('day', (NOW() AT TIME ZONE 'UTC'))
                AND equity IS NOT NULL
              ORDER BY ts ASC
            )
            SELECT
              (SELECT equity FROM x ORDER BY ts ASC LIMIT 1) AS open_eq,
              (SELECT equity FROM x ORDER BY ts DESC LIMIT 1) AS last_eq,
              (SELECT ts FROM x ORDER BY ts ASC LIMIT 1) AS open_ts,
              (SELECT ts FROM x ORDER BY ts DESC LIMIT 1) AS last_ts;
            """)

            with engine.connect() as con:
                r = con.execute(q).fetchone()

            if not r or r[0] is None or r[1] is None:
                # no data yet today
                time.sleep(poll)
                continue

            open_eq = float(r[0])
            last_eq = float(r[1])
            open_ts = r[2]
            last_ts = r[3]

            pnl = last_eq - open_eq
            pnl_pct = (pnl / open_eq * 100.0) if open_eq > 0 else 0.0

            if pnl <= -abs(max_daily_loss_usd) or pnl_pct <= -abs(max_daily_loss_pct):
                set_flag("TRADING_PAUSED", "1")
                set_flag("TRADING_PAUSED_REASON", f"daily_loss day={day} pnl={pnl:.2f} pnl_pct={pnl_pct:.2f}% open={open_eq:.2f} last={last_eq:.2f}")
                LOG.warning(
                    "PAUSED | day=%s pnl=%.2f pnl_pct=%.2f open=%.2f last=%.2f open_ts=%s last_ts=%s",
                    day, pnl, pnl_pct, open_eq, last_eq, open_ts, last_ts
                )

            time.sleep(poll)

        except Exception:
            LOG.exception("loop_error")
            time.sleep(poll)


if __name__ == "__main__":
    main()