# jobs/cron_nn.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timezone
from typing import List

SLEEP_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "60"))
SYMBOLS = os.getenv("CRON_SYMBOLS", "AAPL,MSFT,SPY")
PY = sys.executable  # absolute path to the current python (may include spaces on Windows)

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    print(f"{ts} cron_nn | {msg}", flush=True)

def run_argv(argv: List[str]) -> int:
    """
    Run a command using argv (list form). This avoids quoting issues
    on Windows paths with spaces (e.g., Program Files).
    """
    log(f"run: {' '.join(argv)}")
    p = subprocess.run(argv, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout.rstrip())
    if p.stderr:
        # keep stderr visible; many libs print warnings here
        print(p.stderr.rstrip(), file=sys.stderr)
    return p.returncode

def main():
    log(f"start loop symbols={SYMBOLS} sleep={SLEEP_SECONDS}s")

    while True:
        # 1) Produce fresh NN signals (schema-aware insert)
        run_argv([PY, "-m", "jobs.make_signals_nn"])

        # 2) Place bracket orders from signals (dedupe/wash-guard already in your executor)
        min_strength = os.getenv("MIN_STRENGTH", "0.45")
        run_argv([PY, "-m", "services.executor_bracket", "--since-days", "1", "--min-strength", str(min_strength)])

        # 3) Ensure exits for any filled positions (OCO TP/SL)
        run_argv([PY, "-m", "jobs.manage_exits", "--symbols", SYMBOLS])

        # 4) Reprice/cancel stale working orders (RTH/ETH thresholds)
        run_argv([PY, "-m", "jobs.manage_stale_orders", "--symbols", SYMBOLS])

        # 5) Sync orders table for the dashboard/analytics
        run_argv([PY, "-m", "tools.sync_orders"])

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped by user")
