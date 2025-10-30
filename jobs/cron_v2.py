# cron_v2.py
# Loop runner for signals → scaling → (auto)retrain → execution → maintenance
import os, time, logging, subprocess, shlex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cron_v2")

SLEEP = int(os.getenv("CRON_SLEEP_SECONDS", "180"))
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")

def run(cmd: str):
    log.info("run: %s", cmd)
    try:
        subprocess.run(shlex.split(cmd), check=True)
    except subprocess.CalledProcessError as e:
        log.error("command failed: %s", e)

def main():
    log.info("cron_v2 | start loop symbols=%s sleep=%ss", SYMBOLS, SLEEP)
    log.info("cron_v2 | DB_URL=%s", os.getenv("DB_URL", "(missing)"))

    # one-time idempotent migrations each boot
    run("python -m tools.init_db")
    run("python -m tools.init_orders_db")
    run("python -m tools.db_migrate_equity")
    run("python -m tools.db_migrate_pnl")
    run("python -m tools.db_migrate_models")
    # keep signals schema in shape (px column etc.)
    try:
        run("python -m tools.db_migrate_signals_px")
    except Exception:
        pass  # tool may not exist in older images; ok to skip

    while True:
        try:
            # 1) make rule-based signals
            run("python -m jobs.make_signals")

            # 2) ensemble (adds NN opinions, logs final picks)
            #    keep your existing ensemble call; if you prefer, you can swap to make_signals_nn
            run("python -m jobs.make_signals_ensemble")

            # 3) normalize strengths per symbol (z-score → [0,1])  🧠
            run("python -m jobs.scale_strength")

            # 4) kick weekly retrain if in schedule window (idempotent)  🔁
            run("python -m jobs.auto_retrain")

            # 5) place ATR bracket orders from recent strong signals
            run("python -m services.executor_bracket --since-min 20 --min-strength 0.60")

            # 6) manage open positions & exits
            run(f"python -m jobs.manage_exits --symbols {SYMBOLS}")
            run(f"python -m jobs.manage_stale_orders --symbols {SYMBOLS}")

            # 7) sync broker orders back into DB
            run("python -m tools.sync_orders")

        except Exception as e:
            log.exception("loop error: %s", e)

        log.info("cron_v2 | sleeping %ss …", SLEEP)
        time.sleep(SLEEP)

if __name__ == "__main__":
    main()
