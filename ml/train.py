# ml/train.py
import os, glob, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from ml.features import make_features, make_label

def load_all_csv(path_pattern: str) -> pd.DataFrame:
    frames = []
    for fp in glob.glob(path_pattern):
        sym = os.path.basename(fp).split("_")[0]
        df = pd.read_csv(fp, parse_dates=["time"])
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def main():
    bars = load_all_csv(os.path.join("data","bars_5m","*_5m.csv"))
    feats = []
    for sym, g in bars.groupby("symbol"):
        gf = make_features(g)
        y  = make_label(gf, horizon=int(os.environ.get("ML_HORIZON","3")),
                           thresh=float(os.environ.get("ML_THRESH","0.0007")))
        gf = gf.assign(y=y).dropna()
        gf["symbol"] = sym
        feats.append(gf)
    df = pd.concat(feats, ignore_index=True)

    feature_cols = ["ret_1","ret_5","ema_10","ema_20","rsi_14","atr_14","atr_pct"]
    X = df[feature_cols]
    y = df["y"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, p)
    print(f"AUC: {auc:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": clf, "features": feature_cols}, "models/gbc_5m.pkl")
    print("saved models/gbc_5m.pkl")

if __name__ == "__main__":
    main()
