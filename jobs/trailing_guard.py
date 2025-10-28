# jobs/trailing_guard.py
"""
Background job wrapper for services.trailing_stop
--------------------------------------------------
Runs the trailing stop profit protector continuously.
Safe to run in parallel with cron_nn or executor_bracket.

Usage:
    python -m jobs.trailing_guard
"""

import os
import time
import logging
from services.trailing_stop import trailing_logic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("trailing_guard")

INTERVAL = int(os.getenv("TRAIL_INTERVAL", "120"))

def main():
    logger.info("🚀 starting trailing_guard loop (interval=%ss)", INTERVAL)
    while True:
        try:
            trailing_logic()
        except Exception as e:
            logger.error("Error in trailing_guard: %s", e)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
