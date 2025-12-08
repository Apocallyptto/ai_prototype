#!/usr/bin/env python3
"""
End-of-day closer for Alpaca.

- Sleduje čas do konca US session (default 21:00 UTC).
- Keď sme v okne BUFFER_MIN minút pred close a dnes ešte neflattenovalo:
    - zruší všetky open orders
    - pošle market príkazy na zatvorenie všetkých pozícií
    - po krátkej pauze spraví DRUHÝ CHECK:
        - ak ešte ostali pozície, skúsi flatten ešte raz

Env vars:
    ALPACA_API_KEY / ALPACA_API_SECRET (required)
    ALPACA_PAPER           (default "1")
    EOD_BUFFER_MINUTES     (default "5")     # koľko minút pred close spustiť flatten
    EOD_CLOSE_HOUR_UTC     (default "21")    # hodina close v UTC
    EOD_CLOSE_MINUTE_UTC   (default "0")     # minúta close v UTC
    EOD_SLEEP_SECONDS      (default "60")    # sleep medzi jednotlivými loop cyklami
"""

import os
import time
import logging
from datetime import datetime, time as dtime, timezone

from alpaca.trading.client import TradingClient

# ---- logging ----

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eod_closer")


# ---- helpers ----

def seconds_to_close(now_utc: datetime, close_time_utc: dtime) -> float:
    """Return seconds from now_utc until today's close_time_utc (can be negative after close)."""
    close_dt = datetime.combine(now_utc.date(), close_time_utc)
    if close_dt.tzinfo is None:
        close_dt = close_dt.replace(tzinfo=timezone.utc)
    return (close_dt - now_utc).total_seconds()


def flatten_all(tc: TradingClient) -> None:
    """Cancel all open orders and close all positions with market orders."""
    # Cancel orders
    logger.info("Canceling ALL open orders...")
    try:
        resp = tc.cancel_orders()
        # Alpaca vracia 207 Multi-Status; zalogujeme si raw odpoveď
        logger.warning("Cancel orders returned 207: %s", resp)
    except Exception as exc:
        logger.error("Error canceling orders: %s", exc)

    # Close positions
    logger.info("Closing ALL open positions (market)...")
    try:
        resp = tc.close_all_positions()
        logger.warning("Close positions returned 207: %s", resp)
    except Exception as exc:
        logger.error("Error closing positions: %s", exc)


def log_open_positions(tc: TradingClient, prefix: str) -> None:
    """Helper na zalogovanie aktuálnych pozícií (symbol, qty)."""
    try:
        positions = tc.get_all_positions()
    except Exception as exc:
        logger.error("%s: error loading positions: %s", prefix, exc)
        return

    if not positions:
        logger.info("%s: no open positions.", prefix)
        return

    summary = [(p.symbol, p.qty) for p in positions]
    logger.warning("%s: still open positions: %s", prefix, summary)


# ---- main loop ----

def main() -> None:
    paper = os.getenv("ALPACA_PAPER", "1") != "0"
    api_key = os.environ["ALPACA_API_KEY"]
    api_secret = os.environ["ALPACA_API_SECRET"]

    buffer_min = int(os.getenv("EOD_BUFFER_MINUTES", os.getenv("BUFFER_MIN", "5")))
    close_hour = int(os.getenv("EOD_CLOSE_HOUR_UTC", "21"))
    close_minute = int(os.getenv("EOD_CLOSE_MINUTE_UTC", "0"))
    sleep_seconds = int(os.getenv("EOD_SLEEP_SECONDS", "60"))

    close_time_utc = dtime(hour=close_hour, minute=close_minute, tzinfo=timezone.utc)

    tc = TradingClient(api_key, api_secret, paper=paper)

    logger.info(
        "Starting close_positions_eod loop | ENABLE=True | BUFFER_MIN=%s | CLOSE=%02d:%02d UTC",
        buffer_min,
        close_hour,
        close_minute,
    )

    # aby sme flatten nerobili viackrát za jeden deň
    last_flatten_date = None

    while True:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        sec_to_close = seconds_to_close(now_utc, close_time_utc)

        # Logika:
        # - ak sme v okne [0, buffer_min] min pred close
        # - a ešte sme dnes flatten nerobili
        if sec_to_close <= buffer_min * 60 and sec_to_close >= -3600:
            if last_flatten_date != today:
                logger.info(
                    "EOD FLATTEN TRIGGERED: %.1f sec to close (<= %s min).",
                    sec_to_close,
                    buffer_min,
                )

                # 1) prvý pokus o flatten
                log_open_positions(tc, "Before first flatten")
                flatten_all(tc)

                # uložíme si, že dnešný deň už má flatten
                last_flatten_date = today

                # 2) DRUHÝ CHECK po krátkej pauze
                time.sleep(10)
                try:
                    positions = tc.get_all_positions()
                except Exception as exc:
                    logger.error("Error checking positions after first flatten: %s", exc)
                    positions = []

                if positions:
                    summary = [(p.symbol, p.qty) for p in positions]
                    logger.warning(
                        "Positions still open after first flatten: %s. Retrying flatten...",
                        summary,
                    )
                    flatten_all(tc)

                    # posledný check – už len zalogujeme stav
                    time.sleep(5)
                    log_open_positions(tc, "After second flatten")
                else:
                    logger.info("No positions remain after EOD flatten.")
            else:
                logger.info(
                    "EOD flatten already done today (%s), skipping.", today
                )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
