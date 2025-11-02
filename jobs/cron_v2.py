# services/cron_v2.py
import os, time, logging, subprocess
from datetime import datetime

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("cron_v2")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
SLEEP_SEC = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

def run(cmd: str):
    """Run a subprocess and log result."""
    log.info("run: %s", cmd)
    try:
        subprocess.run(cmd.split(), check=True)
    except subprocess.CalledProcessError as e:
        log.error("command failed: %s", e)

def main():
    log.info("cron_v2 | start loop symbols=%s sleep=%ss", SYMBOLS, SLEEP_SEC)
    log.info("cron_v2 | DB_URL=%s", DB_URL)

    while True:
        try:
            # --- ensure DB schema ---
            run("python -m tools.init_db")
            run("python -m tools.init_orders_db")
            run("python -m tools.db_migrate_equity")
            run("python -m tools.db_migrate_pnl")
            run("python -m tools.db_migrate_models")
            run("python -m tools.db_migrate_signals_px")

            # --- start of each cycle: visibility into BP ---
            run("python -m tools.bp_log")

            # --- signal generation (ensemble only; replaces make_signals) ---
            run("python -m jobs.make_signals_ensemble")

            # --- normalization / auto-retrain / execution pipeline ---
            run("python -m jobs.scale_strength")     # normalize strengths (z-score)
            run("python -m jobs.auto_retrain")       # weekly retrain check
            run("python -m services.executor_bracket")  # bracket orders (uses env thresholds)

            # --- risk management / maintenance ---
            run("python -m jobs.trailing_guard")
            run(f"python -m jobs.manage_exits --symbols {SYMBOLS}")
            run("python -m jobs.partial_fills")  # cancel lingering partials
            run("python -m jobs.reprice_stale")  # reprice orders older than REPRICE_AFTER_SEC
            run(f"python -m jobs.manage_stale_orders --symbols {SYMBOLS}")

            # --- housekeeping ---
            run("python -m tools.sync_orders")

            log.info("cron_v2 | sleeping %ds …", SLEEP_SEC)
            time.sleep(SLEEP_SEC)

        except KeyboardInterrupt:
            log.warning("manual stop")
            break
        except Exception as e:
            log.error("cron_v2 loop error: %s", e)
            time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()
