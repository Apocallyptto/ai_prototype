# jobs/cron_v2.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timezone

SYMBOLS = os.getenv("CRON_SYMBOLS", "AAPL,MSFT,SPY")
SLEEP_SECONDS = int(os.getenv("CRON_SLEEP_SECONDS", "60"))
SINCE_DAYS = os.getenv("CRON_SINCE_DAYS", os.getenv("SINCE_DAYS","1"))
MIN_STRENGTH = os.getenv("CRON_MIN_STRENGTH", os.getenv("MIN_STRENGTH","0.45"))

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} cron_v2 | {msg}", flush=True)

def run(cmd: list[str]):
    log(f"run: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.stdout: print(p.stdout, end="")
        if p.stderr: print(p.stderr, end="")
    except Exception as e:
        log(f"error: {e}")

def main():
    log(f"start loop symbols={SYMBOLS} sleep={SLEEP_SECONDS}s")
    while True:
        # 1) Place new bracket entries from signals
        run([sys.executable, "-m", "services.executor_bracket", "--since-days", str(SINCE_DAYS), "--min-strength", str(MIN_STRENGTH)])

        # 2) Ensure exits for any legacy/manual positions (safety net)
        run([sys.executable, "-m", "jobs.manage_exits", "--symbols", SYMBOLS])

        # 3) Reprice stale standalone limits (ignores bracket legs)
        run([sys.executable, "-m", "jobs.manage_stale_orders", "--symbols", SYMBOLS])

        # 4) Sync orders into DB for UI/analytics
        run([sys.executable, "-m", "tools.sync_orders"])

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
