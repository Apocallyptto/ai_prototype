# jobs/make_signals_ensemble.py
import os, logging, subprocess, json, sys
from datetime import datetime, timezone
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
DISABLE_NN = os.getenv("DISABLE_NN", "0") == "1"

def _run(cmd: List[str]) -> int:
    logging.info("run: " + " ".join(cmd))
    return subprocess.call(cmd)

def _maybe_nn_predict(symbols_csv: str) -> Dict[str, Dict[str, float]]:
    if DISABLE_NN:
        logging.info("NN path is OFF (disabled via DISABLE_NN=1). Ensemble will use rule/technicals only.")
        return {}
    try:
        from jobs.make_signals_nn import nn_predict
        return nn_predict(symbols_csv)  # <-- accepts optional symbols_csv now
    except Exception as e:
        logging.warning(f"NN path failed: {e}; continuing without NN.")
        return {}

def _merge(rule_rows: Dict[str, Dict[str, float]],
           nn_rows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Simple weighted ensemble: 60% rule/technicals, 40% NN.
    Each row like {"side": "buy|sell|hold", "strength": float in [0,1]}.
    """
    weights = {"rule": 0.6, "nn": 0.4}
    out: Dict[str, Dict[str, float]] = {}
    syms = set(rule_rows.keys()) | set(nn_rows.keys())
    for s in syms:
        r = rule_rows.get(s, {"side": "hold", "strength": 0.0})
        n = nn_rows.get(s, {"side": "hold", "strength": 0.0})

        # map side -> signed score
        def score(row):
            if row["side"] == "buy":  return +row["strength"]
            if row["side"] == "sell": return -row["strength"]
            return 0.0

        blend = weights["rule"] * score(r) + weights["nn"] * score(n)
        if blend > +0.10:
            out[s] = {"side": "buy",  "strength": round(min(1.0, abs(blend)), 3)}
        elif blend < -0.10:
            out[s] = {"side": "sell", "strength": round(min(1.0, abs(blend)), 3)}
        else:
            out[s] = {"side": "hold", "strength": round(abs(blend), 3)}
    return out

def _insert_rows(rows: Dict[str, Dict[str, float]]) -> None:
    """
    Reuse your existing rules inserter by calling jobs.make_signals with env or
    (optionally) write a small DB inserter. For now we just log.
    """
    for s, r in rows.items():
        logging.info(f"Ensemble -> {s}: {r['side']} ({r['strength']})")

def main():
    logging.info(f"make_signals_ensemble | symbols={SYMBOLS}")
    # Always generate baseline rule/technical signals
    rc = _run([sys.executable, "-m", "jobs.make_signals"])
    if rc != 0:
        logging.error("jobs.make_signals failed")
        sys.exit(rc)

    # Load rule signals back from DB if you have an accessor; to keep this simple,
    # we’ll read the latest decision from jobs.make_signals_nn adapter as proxy.
    # In your current flow we just combine opinions at runtime:
    # assume rule recommends HOLD with strength 0.5 baseline -> you can replace with actual DB read.
    rule_rows = {s: {"side": "hold", "strength": 0.5} for s in SYMBOLS.split(",")}

    nn_rows = _maybe_nn_predict(SYMBOLS)

    merged = _merge(rule_rows, nn_rows)
    _insert_rows(merged)

if __name__ == "__main__":
    main()
