import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature DataFrame from raw OHLCV bars.

    - Normalizes column names (handles tuples / MultiIndex from some data sources)
    - Ensures we have canonical columns: Open, High, Low, Close, Volume
    - Adds the engineered features that jobs.make_signals_ml expects.
    """

    if df is None or len(df) == 0:
        raise ValueError("make_features() received empty DataFrame")

    # ------------------------------------------------------------------
    # 1) Flatten columns and normalize names
    # ------------------------------------------------------------------
    # Some data sources (or concatenations) can give MultiIndex columns
    # like ('Open', 'AAPL') or ('AAPL', 'Open'). We want to pick the
    # part that looks like OHLCV (open/high/low/close/volume).
    raw_cols = list(df.columns)

    ohlcv_tokens = {
        "open",
        "high",
        "low",
        "close",
        "adj close",
        "adj_close",
        "volume",
        "vol",
        "v",
    }

    flat_names = []
    for c in raw_cols:
        if isinstance(c, tuple):
            # Normalize all tuple parts to lowercase strings
            parts = [str(p) for p in c if p is not None and str(p).strip() != ""]
            lower_parts = [p.strip().lower() for p in parts]

            chosen = None
            # 1) Try to find element that looks like OHLCV
            for lp, orig_p in zip(lower_parts, parts):
                if lp in ohlcv_tokens:
                    chosen = str(orig_p)
                    break

            # 2) Fallback: first non-empty part
            if chosen is None:
                chosen = parts[0] if parts else ""

            name = chosen
        else:
            name = str(c)

        flat_names.append(name)

    df = df.copy()
    df.columns = flat_names

    # Build mapping from lower-case -> original column key
    cols_lower = {}
    for orig in df.columns:
        key = str(orig).lower()
        cols_lower[key] = orig

    def pick(*candidates: str) -> str:
        """
        Pick the first column name that exists (case-insensitive).
        Raises KeyError if none found.
        """
        for cand in candidates:
            lc = cand.lower()
            if lc in cols_lower:
                return cols_lower[lc]
        raise KeyError(f"None of {candidates} found in columns={list(df.columns)}")

    # ------------------------------------------------------------------
    # 2) Canonical OHLCV columns
    # ------------------------------------------------------------------
    open_col = pick("open", "o")
    high_col = pick("high", "h")
    low_col = pick("low", "l")
    close_col = pick("close", "c", "adj close", "adj_close")
    volume_col = pick("volume", "vol", "v")

    out = pd.DataFrame(index=df.index)
    out["Open"] = df[open_col].astype(float)
    out["High"] = df[high_col].astype(float)
    out["Low"] = df[low_col].astype(float)
    out["Close"] = df[close_col].astype(float)
    out["Volume"] = df[volume_col].astype(float)

    # ------------------------------------------------------------------
    # 3) Engineered features (must match FEATURE_COLS in make_signals_ml)
    # ------------------------------------------------------------------
    # Simple returns
    out["return_1"] = out["Close"].pct_change(1)
    out["return_5"] = out["Close"].pct_change(5)
    out["return_10"] = out["Close"].pct_change(10)

    # Moving averages
    out["ma_10"] = out["Close"].rolling(window=10, min_periods=1).mean()
    out["ma_20"] = out["Close"].rolling(window=20, min_periods=1).mean()

    # Rolling std (volatility)
    out["std_10"] = out["Close"].rolling(window=10, min_periods=1).std()
    out["std_20"] = out["Close"].rolling(window=20, min_periods=1).std()

    # Ranges
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["oc_range"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)

    # Volume Z-score
    vol_mean_20 = out["Volume"].rolling(window=20, min_periods=1).mean()
    vol_std_20 = out["Volume"].rolling(window=20, min_periods=1).std()
    out["vol_zscore_20"] = (out["Volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)

    # RSI(14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    roll_up = gain.rolling(window=14, min_periods=1).mean()
    roll_down = loss.rolling(window=14, min_periods=1).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Drop initial NaNs where features are not defined, keep index aligned
    out = out.dropna()

    return out
