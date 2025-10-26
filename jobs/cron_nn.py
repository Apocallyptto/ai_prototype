# jobs/cron_nn.py
from __future__ import annotations

import os
import time
import shlex
import subprocess
from datetime import datetime, timezone

PY = os.getenv("PYTHON", os.getenv("PYTHON_EXE", "python"))

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
SLEEP = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
MIN_STRENGTH = os.getenv("MIN_STRENGTH", "0.60")
SIGNAL_JOB = os.getenv("SIGNAL_JOB", "jobs.make_signals_ensemble")
EXEC_WINDOW_MIN = os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN", "20")  # <<< window in minutes

def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def _run(cmd: list[str]):
    print(f"{_utcnow()} cron_nn | run: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.call(cmd)

def main():
    print(f"{_utcnow()} cron_nn | start loop symbols={SYMBOLS} sleep={SLEEP}s", flush=True)
    print(f"{_utcnow()} cron_nn | signals via {SIGNAL_JOB} (MIN_STRENGTH={MIN_STRENGTH})", flush=True)

    while True:
        # 1) Make signals
        _run([PY, "-m", SIGNAL_JOB])

        # 2) Place orders from recent signals (use --since-min)
        _run([
            PY, "-m", "services.executor_bracket",
            "--since-min", str(EXEC_WINDOW_MIN),
            "--min-strength", str(MIN_STRENGTH),
        ])

        # 3) Manage exits
        _run([PY, "-m", "jobs.manage_exits", "--symbols", SYMBOLS])

        # 4) Reprice stale (if any)
        _run([PY, "-m", "jobs.manage_stale_orders", "--symbols", SYMBOLS])

        # 5) Sync orders table
        _run([PY, "-m", "tools.sync_orders"])

        time.sleep(SLEEP)

if __name__ == "__main__":
    main()
