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


def _db_ping(engine) -> None:
    with engine.connect() as con:
        con.execute(text("SELECT 1"))


def _wait_for_db_ready(max_attempts: int, sleep_seconds: float) -> None:
    """
    Wait until Postgres is reachable before the guard enters its main loop.
    This avoids noisy startup crashes when postgres/DNS is not ready yet.
    """
    engine = get_engine()
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            _db_ping(engine)
            LOG.info("db_ready | attempts=%s", attempt)
            return
        except Exception as exc:
            last_err = exc
            if attempt == 1 or attempt % 5 == 0 or attempt == max_attempts:
                LOG.warning(
                    "db_wait_retry | attempt=%s/%s | sleep=%.1fs | err=%r",
                    attempt,
                    max_attempts,
                    sleep_seconds,
                    exc,
                )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"DB not ready after {max_attempts} attempts; last_err={last_err!r}"
    )


def _safe_set_flag(key: str, value: str) -> bool:
    """
    Best-effort flag setter.
    Returns True on success, False on failure.
    """
    try:
        set_flag(key, value)
        return True
    except Exception:
        LOG.exception("set_flag_failed | key=%s value=%s", key, value)
        return False


def _safe_get_flag(key: str, default: str = "") -> str:
    """
    Best-effort flag getter.
    Returns default on failure.
    """
    try:
        return get_flag(key, default)
    except Exception:
        LOG.exception("get_flag_failed | key=%s", key)
        return default


def main():
    poll = _env_int("GUARD_POLL_SECONDS", 60)
    max_daily_loss_usd = _env_float("GUARD_MAX_DAILY_LOSS_USD", 2.0)
    max_daily_loss_pct = _env_float("GUARD_MAX_DAILY_LOSS_PCT", 2.0)
    enabled = _env_bool("GUARD_ENABLED", True)
    auto_unpause_new_day = _env_bool("GUARD_AUTO_UNPAUSE_NEW_DAY", True)

    db_warmup_max_attempts = _env_int("DB_WARMUP_MAX_ATTEMPTS", 30)
    db_warmup_sleep_seconds = _env_float("DB_WARMUP_SLEEP_SECONDS", 2.0)

    engine = get_engine()

    LOG.info(
        "auto_pause_guard starting | enabled=%s poll=%ss max_loss_usd=%.2f max_loss_pct=%.2f auto_unpause_new_day=%s",
        enabled,
        poll,
        max_daily_loss_usd,
        max_daily_loss_pct,
        auto_unpause_new_day,
    )

    try:
        _wait_for_db_ready(
            max_attempts=db_warmup_max_attempts,
            sleep_seconds=db_warmup_sleep_seconds,
        )
    except Exception:
        LOG.exception("startup_db_not_ready")
        while True:
            time.sleep(poll)
            try:
                _wait_for_db_ready(
                    max_attempts=db_warmup_max_attempts,
                    sleep_seconds=db_warmup_sleep_seconds,
                )
                break
            except Exception:
                LOG.exception("startup_db_retry_failed")

    while True:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()

            # Heartbeat first
            _safe_set_flag("GUARD_HEARTBEAT_TS", now_iso)

            day = _today_utc_date()

            if auto_unpause_new_day:
                last_day = _safe_get_flag("GUARD_LAST_DAY", "")
                was_paused = (_safe_get_flag("TRADING_PAUSED", "0") == "1")
                reason = _safe_get_flag("TRADING_PAUSED_REASON", "") or ""
                guard_status = _safe_get_flag("GUARD_STATUS", "")

                paused_by_guard = ("daily_loss" in reason) or (guard_status == "PAUSED_TRIGGERED")

                if was_paused and paused_by_guard and last_day and last_day != day:
                    _safe_set_flag("TRADING_PAUSED", "0")
                    _safe_set_flag("TRADING_PAUSED_REASON", f"auto_unpause_new_day {day}")
                    LOG.warning("AUTO_UNPAUSE | last_day=%s new_day=%s", last_day, day)

            if not enabled:
                _safe_set_flag("GUARD_LAST_DAY", day)
                _safe_set_flag("GUARD_STATUS", f"DISABLED day={day}")
                time.sleep(poll)
                continue

            q = text(
                """
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
                """
            )

            with engine.connect() as con:
                r = con.execute(q).fetchone()

            if not r or r[0] is None or r[1] is None:
                _safe_set_flag("GUARD_LAST_DAY", day)
                _safe_set_flag("GUARD_STATUS", f"NO_DATA day={day}")
                time.sleep(poll)
                continue

            open_eq = float(r[0])
            last_eq = float(r[1])
            open_ts = r[2]
            last_ts = r[3]

            pnl = last_eq - open_eq
            pnl_pct = (pnl / open_eq * 100.0) if open_eq > 0 else 0.0

            _safe_set_flag("GUARD_LAST_PNL_USD", f"{pnl:.4f}")
            _safe_set_flag("GUARD_LAST_PNL_PCT", f"{pnl_pct:.4f}")
            _safe_set_flag("GUARD_LAST_EQUITY_OPEN", f"{open_eq:.4f}")
            _safe_set_flag("GUARD_LAST_EQUITY_LAST", f"{last_eq:.4f}")
            _safe_set_flag("GUARD_LAST_DAY", day)

            if pnl <= -abs(max_daily_loss_usd) or pnl_pct <= -abs(max_daily_loss_pct):
                reason = (
                    f"daily_loss day={day} pnl={pnl:.2f} pnl_pct={pnl_pct:.2f}% "
                    f"open={open_eq:.2f} last={last_eq:.2f}"
                )
                _safe_set_flag("TRADING_PAUSED", "1")
                _safe_set_flag("TRADING_PAUSED_REASON", reason)
                _safe_set_flag("GUARD_STATUS", "PAUSED_TRIGGERED")

                LOG.warning(
                    "PAUSED | day=%s pnl=%.2f pnl_pct=%.2f open=%.2f last=%.2f open_ts=%s last_ts=%s",
                    day,
                    pnl,
                    pnl_pct,
                    open_eq,
                    last_eq,
                    open_ts,
                    last_ts,
                )
            else:
                _safe_set_flag("GUARD_STATUS", "OK")

            time.sleep(poll)

        except Exception:
            _safe_set_flag("GUARD_STATUS", "ERROR")
            LOG.exception("loop_error")
            time.sleep(poll)


if __name__ == "__main__":
    main()