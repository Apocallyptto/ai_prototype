# jobs/cron_nn.py
from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime, timezone

# Always use the exact interpreter running this process
PY = sys.executable

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default

def _run(cmd: list[str]) -> int:
    # Log command (shortened)
    print(f"{_ts()} cron_nn | run:", " ".join(cmd))
    return subprocess.call(cmd)

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def main() -> None:
    symbols = _env("SYMBOLS", "AAPL,MSFT,SPY")
    min_strength = _env("MIN_STRENGTH", "0.60")
    sleep_s = int(_env("CRON_SLEEP_SECONDS", "180"))
    signal_job = _env("SIGNAL_JOB", "jobs.make_signals_ensemble")

    # executor window (minutes) – matches your newer CLI (--since-min)
    window_min = _env("EXECUTOR_SIGNAL_WINDOW_MIN", "20")
    portfolio_id = _env("PORTFOLIO_ID", "")  # optional; "" means any/ignore

    print(f"{_ts()} cron_nn | start loop symbols={symbols} sleep={sleep_s}s")
    print(f"{_ts()} cron_nn | signals via {signal_job} (MIN_STRENGTH={min_strength})")

    try:
        while True:
            # 1) Generate signals
            _run([PY, "-m", signal_job])

            # 2) Place new brackets from recent signals
            exec_cmd = [
                PY, "-m", "services.executor_bracket",
                "--since-min", window_min,
                "--min-strength", min_strength,
            ]
            if portfolio_id:
                exec_cmd += ["--portfolio-id", portfolio_id]
            _run(exec_cmd)

            # 3) Manage exits / stale orders / sync
            _run([PY, "-m", "jobs.manage_exits", "--symbols", symbols])
            _run([PY, "-m", "jobs.manage_stale_orders", "--symbols", symbols])
            _run([PY, "-m", "tools.sync_orders"])

            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print(f"{_ts()} cron_nn | stopped by user")

if __name__ == "__main__":
    main()
