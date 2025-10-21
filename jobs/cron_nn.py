# jobs/cron_nn.py
from __future__ import annotations
import os, sys, time, subprocess, shlex
from datetime import datetime, timezone

SLEEP_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "60"))
SYMBOLS = os.getenv("CRON_SYMBOLS", "AAPL,MSFT,SPY")

PY = sys.executable

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    print(f"{ts} cron_nn | {msg}", flush=True)

def run(cmd: str):
    log(f"run: {cmd}")
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if p.stdout:
        print(p.stdout.strip())
    if p.stderr:
        print(p.stderr.strip(), file=sys.stderr)
    return p.returncode

def main():
    log(f"start loop symbols={SYMBOLS} sleep={SLEEP_SECONDS}s")
    while True:
        # 1) Produce fresh NN signals (schema-aware insert)
        run(f'{PY} -m jobs.make_signals_nn')

        # 2) Place bracket orders from signals (dedupe/wash-guard already in your executor)
        min_strength = os.getenv("MIN_STRENGTH", "0.45")
        run(f'{PY} -m services.executor_bracket --since-days 1 --min-strength {min_strength}')

        # 3) Ensure exits for any filled positions (OCO TP/SL)
        run(f'{PY} -m jobs.manage_exits --symbols {SYMBOLS}')

        # 4) Reprice/cancel stale working orders (RTH/ETH thresholds)
        run(f'{PY} -m jobs.manage_stale_orders --symbols {SYMBOLS}')

        # 5) Sync orders table for the dashboard/analytics
        run(f'{PY} -m tools.sync_orders')

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped by user")
