# jobs/make_signals_ensemble.py
from __future__ import annotations
import os
import logging
import subprocess
from typing import Dict, Any

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("make_signals_ensemble")

# ---- Optional libs (guarded) ----
_HAS_JOBLIB = False
_HAS_TORCH = False

try:
    import joblib  # noqa: F401
    _HAS_JOBLIB = True
except Exception as e:
    log.warning("NN ensemble: joblib not available (%s) → NN path will be skipped.", e)

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception as e:
    log.warning("NN ensemble: torch not available (%s) → NN path will be skipped.", e)

DISABLE_NN = os.getenv("DISABLE_NN", "0") == "1"
USE_NN = (not DISABLE_NN) and _HAS_JOBLIB and _HAS_TORCH


def _symbols_env() -> str:
    return os.getenv("SYMBOLS", "AAPL,MSFT,SPY")


def _run_module(mod: str, *args: str) -> int:
    """
    Run a Python module in a subprocess within this container image, inheriting env.
    Returns the process return code.
    """
    cmd = ["python", "-m", mod, *args]
    log.info("run: %s", " ".join(cmd))
    return subprocess.call(cmd)


def _maybe_nn_predict(symbols_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Optional NN predictor hook.
    Returns {} when NN is disabled or libs are missing.
    """
    if not USE_NN:
        why = "disabled via DISABLE_NN=1" if DISABLE_NN else "missing libs (torch/joblib)"
        log.info("NN path is OFF (%s). Ensemble will use rule/technicals only.", why)
        return {}
    # Import here (late) so missing libs don’t break the module import
    from jobs.make_signals_nn import nn_predict  # type: ignore
    return nn_predict(symbols_csv)


def _weighted_ensemble(
    rule_rows: Dict[str, Dict[str, Any]],
    nn_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Combine rule + NN strengths.
    Row format: {SYM: {"side": "buy|sell", "strength": float}}
    On side conflict, keep record with larger |strength|.
    """
    out: Dict[str, Dict[str, Any]] = {}
    w_rule = float(os.getenv("W_RULE", "0.7"))
    w_nn = float(os.getenv("W_NN", "0.3"))

    # seed with rule
    for sym, rec in rule_rows.items():
        out[sym] = dict(rec)

    # blend NN
    for sym, rec in nn_rows.items():
        if sym not in out:
            out[sym] = {"side": rec.get("side"), "strength": float(rec.get("strength", 0.0))}
            continue

        nn_side = rec.get("side")
        nn_str = float(rec.get("strength", 0.0))
        rule_side = out[sym].get("side")
        rule_str = float(out[sym].get("strength", 0.0))

        if nn_side == rule_side:
            out[sym]["strength"] = w_rule * rule_str + w_nn * nn_str
        else:
            if abs(nn_str) > abs(rule_str):
                out[sym] = {"side": nn_side, "strength": nn_str}

    return out


def main():
    symbols = _symbols_env()
    log.info("make_signals_ensemble | symbols=%s", symbols)

    # 1) Produce rule/technical signals into DB by running the existing module.
    #    (No import of jobs.make_signals needed.)
    rc = _run_module("jobs.make_signals")
    if rc != 0:
        log.error("jobs.make_signals failed (rc=%s). Aborting ensemble step.", rc)
        return

    # 2) Optionally compute NN opinions (in-memory)
    nn_rows = _maybe_nn_predict(symbols)

    if nn_rows:
        log.info("NN opinions present for %d symbols; ensemble blending ready.", len(nn_rows))
        # If you later want to persist blended values, fetch the just-written rule rows
        # from the DB and upsert a blended record. Keeping logging only for now.
    else:
        log.info("No NN opinions; using rule/technicals only.")


if __name__ == "__main__":
    main()
