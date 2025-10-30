# jobs/auto_retrain.py
"""
Auto-retrain scheduler (weekly) for GradientBoosting model.

How it works (idempotent):
- Reads schedule from env (day/hour/minute, timezone optional).
- Checks a small KV table in Postgres to see when we last trained.
- If we're inside the schedule window AND last train was > 6 days ago,
  it runs: `python -m jobs.train_model_gbc`
- Records the run timestamp to prevent duplicate retrains the same week.

You can safely call this every loop; it'll only run at the scheduled time.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("auto_retrain")

# --- Config (env) ---
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

# Enable/disable
ENABLE = os.getenv("ENABLE_WEEKLY_RETRAIN", "1") == "1"

# Schedule (defaults: Sunday 23:00 Europe/Bratislava)
RETRAIN_DAY = os.getenv("RETRAIN_DAY", "SUN").upper()     # MON/TUE/WED/THU/FRI/SAT/SUN
RETRAIN_HOUR = int(os.getenv("RETRAIN_HOUR", "23"))       # 0..23
RETRAIN_MINUTE = int(os.getenv("RETRAIN_MINUTE", "0"))    # 0..59
RETRAIN_TZ = os.getenv("RETRAIN_TZ", "Europe/Bratislava") # any IANA TZ

# Window to allow retrain (in minutes). If loop misses exact minute, still ok.
WINDOW_MIN = int(os.getenv("RETRAIN_WINDOW_MINUTES", "20"))

# Minimum gap between retrains (days)
MIN_GAP_DAYS = int(os.getenv("RETRAIN_MIN_GAP_DAYS", "6"))

KV_TABLE = "app_kv"       # simple key-value store in DB
KV_KEY_LAST = "last_model_retrain_iso"

DAY_MAP = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}

def _ensure_kv_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_kv (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.commit()

def _get_last_retrain(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM app_kv WHERE key=%s", (KV_KEY_LAST,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            v = row[0]
            # expect: {"when": "2025-10-30T22:59:12Z"}
            iso = v.get("when")
            if not iso:
                return None
            # parse as UTC
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return None

def _set_last_retrain(conn, when_dt_utc: datetime):
    as_json = json.dumps({"when": when_dt_utc.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00","Z")})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app_kv (key, value)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW();
            """,
            (KV_KEY_LAST, as_json),
        )
        conn.commit()

def _now_tz():
    try:
        return datetime.now(ZoneInfo(RETRAIN_TZ))
    except Exception:
        # fallback UTC if TZ invalid
        return datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

def _should_run(now_tz: datetime, last_utc: datetime | None) -> bool:
    if not ENABLE:
        return False

    # Respect minimum gap between trainings
    if last_utc is not None:
        if (datetime.now(ZoneInfo("UTC")) - last_utc) < timedelta(days=MIN_GAP_DAYS):
            return False

    # Build scheduled datetime for this week in local tz
    target_dow = DAY_MAP.get(RETRAIN_DAY, 6)  # default SUN
    # current week Monday=0..Sunday=6
    curr_dow = now_tz.weekday()

    # find most recent occurrence (this week) of RETRAIN_DAY
    # make a candidate scheduled time in same week
    delta_days = (target_dow - curr_dow) % 7
    candidate = (now_tz + timedelta(days=delta_days)).replace(
        hour=RETRAIN_HOUR, minute=RETRAIN_MINUTE, second=0, microsecond=0
    )

    # if candidate is in the future by almost a week (because modulo wrapped),
    # and we're earlier in the week than target time, we need the candidate in the *current* week
    # Alternative approach: compute the candidate for *this* calendar week (from Monday)
    # We can instead construct from week start:
    week_start = (now_tz - timedelta(days=curr_dow)).replace(hour=0, minute=0, second=0, microsecond=0)
    candidate = week_start + timedelta(days=target_dow, hours=RETRAIN_HOUR, minutes=RETRAIN_MINUTE)

    # Decide window: [candidate - WINDOW/2, candidate + WINDOW/2]
    half = timedelta(minutes=WINDOW_MIN/2.0)
    left = candidate - half
    right = candidate + half

    return left <= now_tz <= right

def _run_training():
    log.info("Starting weekly retrain: running jobs.train_model_gbc ...")
    # Run as module so it uses your package paths
    cmd = [sys.executable, "-m", "jobs.train_model_gbc"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60*60)  # up to 60 min
        stdout = (proc.stdout or "")[-4000:]
        stderr = (proc.stderr or "")[-4000:]
        log.info("train stdout (tail):\n%s", stdout)
        if proc.returncode != 0:
            log.warning("train_model_gbc exit code %s; stderr tail:\n%s", proc.returncode, stderr)
            return False
        log.info("Weekly retrain finished OK.")
        return True
    except subprocess.TimeoutExpired:
        log.warning("train_model_gbc timed out.")
        return False
    except Exception as e:
        log.warning("train_model_gbc failed: %s", e)
        return False

def main():
    now_local = _now_tz()
    now_utc = datetime.now(ZoneInfo("UTC"))
    try:
        with psycopg2.connect(DB_URL) as conn:
            _ensure_kv_table(conn)
            last = _get_last_retrain(conn)
            if _should_run(now_local, last):
                ok = _run_training()
                if ok:
                    _set_last_retrain(conn, now_utc)
            else:
                log.debug("Not in retrain window or gap not met; skipping.")
    except Exception as e:
        log.warning("auto_retrain error: %s", e)

if __name__ == "__main__":
    main()
