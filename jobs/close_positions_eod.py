# jobs/close_positions_eod.py

import os
import time
import logging
from datetime import datetime, timezone
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

ALPACA_BASE_URL = os.environ["ALPACA_BASE_URL"].rstrip("/")
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_API_SECRET = os.environ["ALPACA_API_SECRET"]

EOD_FLATTEN_ENABLE = os.getenv("EOD_FLATTEN_ENABLE", "1") == "1"
EOD_FLATTEN_MIN_BEFORE_CLOSE = int(os.getenv("EOD_FLATTEN_MIN_BEFORE_CLOSE", "5"))

SESSION = requests.Session()
SESSION.headers.update(
    {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }
)


def get_clock():
    resp = SESSION.get(f"{ALPACA_BASE_URL}/v2/clock", timeout=10)
    resp.raise_for_status()
    return resp.json()


def seconds_to_close(clock: dict) -> float:
    """
    clock["next_close"] je ISO string napr. '2025-12-05T21:00:00Z'
    """
    next_close_str = clock["next_close"].replace("Z", "+00:00")
    next_close = datetime.fromisoformat(next_close_str)
    now = datetime.now(timezone.utc)
    return (next_close - now).total_seconds()


def cancel_all_orders():
    logging.info("Canceling ALL open orders...")
    resp = SESSION.delete(f"{ALPACA_BASE_URL}/v2/orders", timeout=15)
    if resp.status_code not in (200, 204):
        logging.warning("Cancel orders returned %s: %s", resp.status_code, resp.text)
    else:
        logging.info("Cancel orders OK (%s)", resp.status_code)


def close_all_positions():
    logging.info("Closing ALL open positions (market)...")
    # cancel_orders=true pre istotu, aj keď už rušíme predtým
    resp = SESSION.delete(
        f"{ALPACA_BASE_URL}/v2/positions", params={"cancel_orders": "true"}, timeout=30
    )
    if resp.status_code not in (200, 204):
        logging.warning("Close positions returned %s: %s", resp.status_code, resp.text)
    else:
        logging.info("Close positions OK (%s)", resp.status_code)


def flatten_eod_if_needed(last_done_date: str | None):
    """
    Vráti nový last_done_date (aby sme flatten nerobili viackrát za jeden deň).
    """
    if not EOD_FLATTEN_ENABLE:
        logging.debug("EOD flatten disabled via EOD_FLATTEN_ENABLE")
        return last_done_date

    try:
        clock = get_clock()
    except Exception as e:
        logging.error("Failed to fetch Alpaca clock: %s", e)
        return last_done_date

    if not clock.get("is_open", False):
        # trh je zatvorený => nový deň ešte nezačal / skončil
        return None

    secs = seconds_to_close(clock)
    trading_day = clock["next_close"][:10]  # 'YYYY-MM-DD'

    logging.debug(
        "Clock: is_open=%s, secs_to_close=%.1f, trading_day=%s",
        clock.get("is_open"),
        secs,
        trading_day,
    )

    if secs <= EOD_FLATTEN_MIN_BEFORE_CLOSE * 60 and secs > 0:
        if last_done_date == trading_day:
            # už sme dnes flatten robili
            logging.info(
                "EOD flatten already done today (%s), skipping.", trading_day
            )
            return last_done_date

        logging.info(
            "EOD FLATTEN TRIGGERED: %.1f sec to close (<= %d min).",
            secs,
            EOD_FLATTEN_MIN_BEFORE_CLOSE,
        )

        try:
            cancel_all_orders()
        except Exception as e:
            logging.error("Error while canceling orders: %s", e)

        try:
            close_all_positions()
        except Exception as e:
            logging.error("Error while closing positions: %s", e)

        logging.info("EOD flatten finished for trading day %s", trading_day)
        return trading_day

    return last_done_date


def main():
    logging.info(
        "Starting close_positions_eod loop | ENABLE=%s | BUFFER_MIN=%d",
        EOD_FLATTEN_ENABLE,
        EOD_FLATTEN_MIN_BEFORE_CLOSE,
    )

    last_done_date = None

    while True:
        last_done_date = flatten_eod_if_needed(last_done_date)
        # stačí raz za minútu
        time.sleep(60)


if __name__ == "__main__":
    main()
