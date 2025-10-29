# jobs/cron_v2.py
from __future__ import annotations

import os
import time
import shlex
import subprocess
import logging
import psycopg2

# ---------- config ----------
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
SLEEP_S = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
MIN_STRENGTH = os.getenv("MIN_STRENGTH", "0.60")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
# Keep NN off in container unless you’ve built the images with torch/joblib
DISABLE_NN = os.getenv("DISABLE_NN", "1") in ("1", "true", "True")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cron_v2")


# ---------- helpers ----------
def _db_ready(dsn: str) -> bool:
    try:
        with psycopg2.connect(dsn) as _:
            return True
    except Exception as e:
        log.info("DB not ready yet: %s", e)
        return False


def _run(cmd: str) -> int:
    """Run a module/command and stream output to our logs; return exit code."""
    log.info("run: %s", cmd)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            log.info("%s", line)
    proc.wait()
    return proc.returncode


def _init_db_once():
    """Create minimal schema so other jobs don’t crash on first boot."""
    # init core tables
    _run("python -m tools.init_db")
    _run("python -m tools.init_orders_db")

    # optional alembic-style migrations that expect DATABASE_URL; skip if not set
    if os.getenv("DATABASE_URL"):
        _run("python -m tools.db_migrate_equity")
        _run("python -m tools.db_migrate_pnl")


# ---------- main loop ----------
def main():
    log.info("cron_v2 | start loop symbols=%s sleep=%ss", SYMBOLS, SLEEP_S)
    log.info("cron_v2 | DB_URL=%s", DB_URL)

    # wait for DB
    deadline = time.time() + 60
    while time.time() < deadline and not _db_ready(DB_URL):
        time.sleep(2)
    if not _db_ready(DB_URL):
        log.error("cron_v2 | database is not reachable, exiting.")
        return

    # make sure required tables exist
    _init_db_once()

    while True:
        try:
            # 1) build signals (ensemble if available, otherwise rule/technicals)
            if DISABLE_NN:
                log.info("signals: DISABLE_NN=1 -> ensemble will skip NN path.")
            _run("python -m jobs.make_signals_ensemble")

            # 2) submit new brackets from recent strong signals
            _run(f"python -m services.executor_bracket --since-min 20 --min-strength {MIN_STRENGTH}")

            # 3) house-keeping: exits / stale orders
            _run(f"python -m jobs.manage_exits --symbols {SYMBOLS}")
            _run(f"python -m jobs.manage_stale_orders --symbols {SYMBOLS}")

            # 4) persist broker orders locally
            _run("python -m tools.sync_orders")

        except Exception as e:
            log.exception("cron_v2 | loop error: %s", e)

        log.info("cron_v2 | sleeping %ss …", SLEEP_S)
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
