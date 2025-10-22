# jobs/cron_nn.py
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PY = sys.executable

# cadence + universe
SLEEP_SEC = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
SYMBOLS   = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
MIN_STRENGTH = os.getenv("MIN_STRENGTH", "0.60")

# choose which signals job to run
# options: jobs.make_signals_ensemble | jobs.make_signals_nn | jobs.make_signals_ml
SIGNAL_JOB = os.getenv("SIGNAL_JOB", "jobs.make_signals_ensemble")

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def run_argv(argv: list[str]):
    print(f"{ts()} cron_nn | run: {' '.join(argv)}", flush=True)
    p = subprocess.Popen(argv)
    p.wait()
    return p.returncode

def main():
    print(f"{ts()} cron_nn | start loop symbols={SYMBOLS} sleep={SLEEP_SEC}s", flush=True)
    print(f"{ts()} cron_nn | signals via {SIGNAL_JOB} (MIN_STRENGTH={MIN_STRENGTH})", flush=True)

    while True:
        # 1) generate signals
        run_argv([PY, "-m", SIGNAL_JOB])

        # 2) place brackets from DB signals (last 1 day, min strength = env)
        run_argv([PY, "-m", "services.executor_bracket", "--since-days", "1", "--min-strength", MIN_STRENGTH])

        # 3) manage exits (idempotent)
        run_argv([PY, "-m", "jobs.manage_exits", "--symbols", SYMBOLS])

        # 4) refresh stale open limit parents
        run_argv([PY, "-m", "jobs.manage_stale_orders", "--symbols", SYMBOLS])

        # 5) sync order book -> Neon
        run_argv([PY, "-m", "tools.sync_orders"])

        # sleep to next tick
        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{ts()} cron_nn | stopped by user", flush=True)
