# scripts/bootstrap.py
from __future__ import annotations

import os
import sys
import time
import logging
import subprocess

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bootstrap")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
WAIT_SECS = int(os.getenv("DB_WAIT_SECS", "60"))
DISABLE_NN = os.getenv("DISABLE_NN", "1") == "1"  # default OFF in containers

def _check_db() -> bool:
    try:
        import psycopg2  # installed in requirements.txt
        conn = psycopg2.connect(DB_URL)
        conn.close()
        return True
    except Exception as e:
        log.info("DB not ready yet: %s", e)
        return False

def _run_module(mod: str, *args: str) -> int:
    cmd = ["python", "-m", mod, *args]
    log.info("run: %s", " ".join(cmd))
    return subprocess.call(cmd)

def main() -> None:
    log.info("BOOT | DB_URL = %s", DB_URL)

    # 1) Wait for Postgres
    deadline = time.time() + WAIT_SECS
    while time.time() < deadline:
        if _check_db():
            log.info("DB is ready.")
            break
        time.sleep(2)
    else:
        log.error("DB did not become ready within %s seconds.", WAIT_SECS)
        sys.exit(1)

    # 2) Initialize / migrate schema (idempotent)
    # These scripts already exist in your repo’s tools/
    _run_module("tools.init_db")
    _run_module("tools.init_orders_db")           # if present; no-op otherwise
    _run_module("tools.db_migrate_equity")        # if present; safe if already ran
    _run_module("tools.db_migrate_pnl")           # if present; safe if already ran

    # 3) Kick one-time sync to ensure orders table exists/works
    _run_module("tools.sync_orders")

    # 4) Start the main loop process
    # Use the newer cron that calls: make_signals_ensemble -> executor_bracket -> manage_* -> sync_orders
    target = ["python", "-m", "jobs.cron_v2"]
    log.info("exec: %s", " ".join(target))

    # Replace current process (so Docker gets proper exit code & restarts on failure)
    os.execvp(target[0], target)

if __name__ == "__main__":
    main()
