import re
from pathlib import Path

path = Path("services") / "signal_executor.py"
if not path.exists():
    raise SystemExit(f"ERROR: file not found: {path}")

src = path.read_text(encoding="utf-8")
bak = path.with_suffix(".py.bak")
bak.write_text(src, encoding="utf-8")

# 1) Ensure import for market_gate
if "from services.market_gate import should_trade_now" not in src:
    # insert after the last import line near top
    m = re.search(r"^(?:import .*\n|from .* import .*\n)+", src, flags=re.M)
    if not m:
        raise SystemExit("ERROR: couldn't find import block")
    import_block = m.group(0)
    src = src.replace(import_block, import_block + "from services.market_gate import should_trade_now\n")

# 2) Replace mark_signal() entirely (fix UUID/TEXT mismatch robustly)
mark_re = re.compile(r"^def\s+mark_signal\s*\(.*?\n(?=^def\s|\Z)", flags=re.M | re.S)

new_mark = """def mark_signal(engine, signal_id: int, status: str, note: str, alpaca_order_id=None) -> None:
    # Alpaca SDK may return UUID objects; DB column is TEXT in our schema.
    if alpaca_order_id is not None:
        alpaca_order_id = str(alpaca_order_id)

    # Support both styles: some older code used param name 'oid' in SQL params
    sql = text(\"\"\"
        UPDATE signals
        SET processed_status = :status,
            processed_note = :note,
            processed_at = NOW(),
            alpaca_order_id = COALESCE(CAST(:oid AS text), alpaca_order_id)
        WHERE id = :id
    \"\"\")
    with engine.begin() as conn:
        conn.execute(sql, {"status": status, "note": note, "oid": alpaca_order_id, "id": signal_id})
"""

if mark_re.search(src):
    src = mark_re.sub(new_mark + "\n\n", src, count=1)
else:
    raise SystemExit("ERROR: mark_signal() not found to replace")

# 3) Add STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE env read in main() (once)
# We look for poll_seconds assignment and inject after it if missing
if "stop_new_entries_min_before_close" not in src:
    src = re.sub(
        r"(poll_seconds\s*=\s*env_int\([^\n]*\)\s*\n)",
        r"\\1    stop_new_entries_min_before_close = env_int(\"STOP_NEW_ENTRIES_MIN_BEFORE_CLOSE\", 10)\n",
        src,
        count=1
    )

# 4) Insert market gate block at start of each while loop (before try)
if "market_gate | skip trading" not in src:
    # Replace first occurrence of while True: ... try:
    src = re.sub(
        r"(^\s*while\s+True\s*:\s*\n)(\s*)try\s*:\s*\n",
        r"""\\1\\2ok, reason, clock = should_trade_now(stop_new_entries_min_before_close)
\\2if not ok:
\\2    logger.info(
\\2        "market_gate | skip trading | reason=%s | is_open=%s | ts=%s | next_open=%s | next_close=%s",
\\2        reason,
\\2        clock.get("is_open"),
\\2        clock.get("timestamp"),
\\2        clock.get("next_open"),
\\2        clock.get("next_close"),
\\2    )
\\2    time.sleep(poll_seconds)
\\2    continue
\\2try:
""",
        src,
        count=1,
        flags=re.M
    )

path.write_text(src, encoding="utf-8")
print(f"OK: patched {path}")
print(f"Backup saved as: {bak}")
