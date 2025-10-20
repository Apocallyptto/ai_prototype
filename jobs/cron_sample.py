# jobs/cron_sample.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timezone

SYMBOLS = os.getenv("CRON_SYMBOLS", "AAPL,MSFT,SPY")
SLEEP_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "60"))

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} cron | {msg}", flush=True)

def run(cmd: list[str]):
    log(f"run: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, end="")
    except Exception as e:
        log(f"error: {e}")

def main():
    log(f"start cron over symbols={SYMBOLS} sleep={SLEEP_SECONDS}s")
    while True:
        # 1) Ensure exits exist
        run([sys.executable, "-m", "jobs.manage_exits", "--symbols", SYMBOLS])
        # 2) Reprice stale open (non-OCO) orders
        run([sys.executable, "-m", "jobs.manage_stale_orders", "--symbols", SYMBOLS])
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
