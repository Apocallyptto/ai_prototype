# jobs/make_signals_ensemble.py
import os, logging, math
from typing import Dict, List
import psycopg2
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals_ensemble")

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "paper")

WEIGHT_RULE = float(os.getenv("ENSEMBLE_W_RULE", "0.6"))
WEIGHT_NN   = float(os.getenv("ENSEMBLE_W_NN", "0.4"))
HOLD_THRESH = float(os.getenv("ENSEMBLE_HOLD_THRESHOLD", "0.10"))  # |score| <= HOLD -> hold

# --- Rules side/strength (variable; not constant) ---
def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = (up / dn.replace(0, float("nan"))).fillna(0.0)
    return 100 - 100/(1+rs.replace(0, float("inf")))

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _atr(h, l, c, n=14):
    # robust ATR calc without to_frame()
    hl = (h - l).abs()
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rule_one_symbol(sym: str) -> Dict[str, float]:
    try:
        df = yf.download(sym, interval="5m", period="2d", progress=False, auto_adjust=False).rename(columns=str.lower)
        if df is None or df.empty or len(df) < 60:
            return {"side":"hold","strength":0.0}
        c = df["close"]; h = df["high"]; l = df["low"]
        rsi14 = _rsi(c, 14)
        ema20 = _ema(c, 20)
        ema50 = _ema(c, 50)
        atr14 = _atr(h, l, c, 14).fillna(0.0)
        px    = float(c.iloc[-1])

        up_trend   = px > float(ema50.iloc[-1])
        down_trend = px < float(ema50.iloc[-1])
        rsi_up     = float(rsi14.iloc[-1]) - 50.0
        dist_ema   = (px - float(ema20.iloc[-1]))

        atr = max(1e-6, float(atr14.iloc[-1]))
        score = 0.4*(rsi_up/20.0) + 0.6*(dist_ema/atr)
        score = max(-1.0, min(1.0, score))

        if score > +0.05 and up_trend:
            return {"side":"buy",  "strength": round(min(1.0, abs(score)), 3)}
        if score < -0.05 and down_trend:
            return {"side":"sell", "strength": round(min(1.0, abs(score)), 3)}
        return {"side":"hold","strength": round(abs(score), 3)}
    except Exception as e:
        log.warning("%s rule calc failed: %s", sym, e)
        return {"side":"hold","strength":0.0}

def _rules_all(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    return {s: _rule_one_symbol(s) for s in symbols}

def _nn_rows(symbols_csv: str) -> Dict[str, Dict[str, float]]:
    try:
        from jobs.make_signals_nn import nn_predict
        return nn_predict(symbols_csv)
    except Exception as e:
        log.warning("NN path failed: %s; using rules only.", e)
        return {}

def _score(row):
    side = row.get("side","hold")
    st   = float(row.get("strength",0.0))
    if side == "buy":  return +st
    if side == "sell": return -st
    return 0.0

def _merge(rule_rows: Dict[str, Dict[str, float]],
           nn_rows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in set(rule_rows.keys()) | set(nn_rows.keys()):
        r = rule_rows.get(s, {"side":"hold","strength":0.0})
        n = nn_rows.get(s,   {"side":"hold","strength":0.0})
        blend = WEIGHT_RULE*_score(r) + WEIGHT_NN*_score(n)
        if blend > +HOLD_THRESH:
            out[s] = {"side":"buy",  "strength": round(min(1.0, abs(blend)), 3)}
        elif blend < -HOLD_THRESH:
            out[s] = {"side":"sell", "strength": round(min(1.0, abs(blend)), 3)}
        else:
            out[s] = {"side":"hold", "strength": round(abs(blend), 3)}
    return out

def _last_px(sym: str) -> float:
    try:
        p = yf.Ticker(sym).fast_info.last_price
        return float(p) if p is not None else float("nan")
    except Exception:
        return float("nan")

def _insert_rows(rows: Dict[str, Dict[str, float]]):
    sql = """
        INSERT INTO public.signals (created_at, symbol, side, strength, px, portfolio_id)
        VALUES (NOW(), %s, %s, %s, %s, %s)
    """
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        for s, r in rows.items():
            if r["side"] == "hold":
                continue
            px = _last_px(s)
            cur.execute(sql, (s, r["side"], float(r["strength"]), (None if math.isnan(px) else px), PORTFOLIO_ID))
        conn.commit()

def main():
    log.info("make_signals_ensemble | symbols=%s", ",".join(SYMBOLS))
    rule_rows = _rules_all(SYMBOLS)
    nn_rows   = _nn_rows(",".join(SYMBOLS))
    merged    = _merge(rule_rows, nn_rows)

    for s, r in merged.items():
        log.info("Ensemble -> %s: %s (%.3f)", s, r["side"], r["strength"])

    _insert_rows(merged)

if __name__ == "__main__":
    main()
