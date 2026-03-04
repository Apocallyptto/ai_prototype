import os
import time
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from tools.db import get_engine
from tools.system_flags import set_flag, get_flag


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

    # NEW: automatically clear pause at start of a new day (if pause was caused by this guard)
    auto_unpause_new_day = _env_bool("GUARD_AUTO_UNPAUSE_NEW_DAY", True)

    engine = get_engine()

    LOG.info(
        "auto_pause_guard starting | enabled=%s poll=%ss max_loss_usd=%.2f max_loss_pct=%.2f auto_unpause_new_day=%s",
        enabled, poll, max_daily_loss_usd, max_daily_loss_pct, auto_unpause_new_day
    )

    while True:
        try:
            # Heartbeat so we can see guard is alive (even when it doesn't pause)
            set_flag("GUARD_HEARTBEAT_TS", datetime.now(timezone.utc).isoformat())
            set_flag("GUARD_STATUS", "OK")

            day = _today_utc_date()

            # NEW: auto-unpause on new day if guard paused yesterday
            if auto_unpause_new_day:
                last_day = get_flag("GUARD_LAST_DAY", "")
                was_paused = (get_flag("TRADING_PAUSED", "0") == "1")
                reason = get_flag("TRADING_PAUSED_REASON", "") or ""
                # only unpause if our guard set the pause
                paused_by_guard = ("daily_loss" in reason) or (get_flag("GUARD_STATUS", "") == "PAUSED_TRIGGERED")

                if was_paused and paused_by_guard and last_day and last_day != day:
                    set_flag("TRADING_PAUSED", "0")
                    set_flag("TRADING_PAUSED_REASON", f"auto_unpause_new_day {day}")
                    LOG.warning("AUTO_UNPAUSE | last_day=%s new_day=%s", last_day, day)

            if not enabled:
                set_flag("GUARD_STATUS", f"DISABLED day={day}")
                time.sleep(poll)
                continue

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
                set_flag("GUARD_LAST_DAY", day)
                set_flag("GUARD_STATUS", f"NO_DATA day={day}")
                time.sleep(poll)
                continue

            open_eq = float(r[0])
            last_eq = float(r[1])
            open_ts = r[2]
            last_ts = r[3]

            pnl = last_eq - open_eq
            pnl_pct = (pnl / open_eq * 100.0) if open_eq > 0 else 0.0

            # store last computed stats for visibility
            set_flag("GUARD_LAST_PNL_USD", f"{pnl:.4f}")
            set_flag("GUARD_LAST_PNL_PCT", f"{pnl_pct:.4f}")
            set_flag("GUARD_LAST_EQUITY_OPEN", f"{open_eq:.4f}")
            set_flag("GUARD_LAST_EQUITY_LAST", f"{last_eq:.4f}")
            set_flag("GUARD_LAST_DAY", day)

            if pnl <= -abs(max_daily_loss_usd) or pnl_pct <= -abs(max_daily_loss_pct):
                reason = (
                    f"daily_loss day={day} pnl={pnl:.2f} pnl_pct={pnl_pct:.2f}% "
                    f"open={open_eq:.2f} last={last_eq:.2f}"
                )
                set_flag("TRADING_PAUSED", "1")
                set_flag("TRADING_PAUSED_REASON", reason)
                set_flag("GUARD_STATUS", "PAUSED_TRIGGERED")

                LOG.warning(
                    "PAUSED | day=%s pnl=%.2f pnl_pct=%.2f open=%.2f last=%.2f open_ts=%s last_ts=%s",
                    day, pnl, pnl_pct, open_eq, last_eq, open_ts, last_ts
                )
            else:
                set_flag("GUARD_STATUS", "OK")

            time.sleep(poll)

        except Exception:
            set_flag("GUARD_STATUS", "ERROR")
            LOG.exception("loop_error")
            time.sleep(poll)


if __name__ == "__main__":
    main()