# ml/features.py
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = np.maximum.reduce([
        (h - l).values,
        np.abs(h - c.shift(1)).values,
        np.abs(l - c.shift(1)).values
    ])
    tr = pd.Series(tr, index=df.index)
    return tr.rolling(length).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"]   = out["close"].pct_change()
    out["ret_5"]   = out["close"].pct_change(5)
    out["ema_10"]  = ema(out["close"], 10)
    out["ema_20"]  = ema(out["close"], 20)
    out["rsi_14"]  = rsi(out["close"], 14)
    out["atr_14"]  = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / out["close"]
    out = out.dropna().reset_index(drop=True)
    return out

def make_label(df: pd.DataFrame, horizon: int = 3, thresh: float = 0.0005):
    """
    Binary label:
      1 = next 'horizon' bars up by > +thresh
      0 = next 'horizon' bars down by < -thresh
    Otherwise None (we drop).
    """
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    label = np.where(fwd >  thresh, 1,
             np.where(fwd < -thresh, 0, np.nan))
    return pd.Series(label, index=df.index)
