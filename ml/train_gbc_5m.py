# ml/train_gbc_5m.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ml.nn_train import make_features
from jobs.make_signals_ml import FEATURE_COLS  # reuse exactly the same features


def load_training_data() -> pd.DataFrame:
    """
    TODO: Replace this with your real historical data loader.

    For example, if you already have a CSV with OHLCV bars, load it here.
    The important thing: columns must be compatible with make_features()
    (Timestamp, Open, High, Low, Close, Volume).
    """
    raise RuntimeError("Please implement load_training_data() for your data.")


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = load_training_data()
    feats = make_features(df)

    # X = feature matrix with exactly the same 11 columns we use in Docker
    X = feats[FEATURE_COLS].values

    # TODO: adjust target creation to whatever you used before.
    # Example: predict next-bar direction:
    future_return = feats["return_1"].shift(-1)
    y = np.where(future_return > 0, 1, 0)  # 1=up, 0=down
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask].astype(int)

    return X, y


def main():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "gbc_5m.pkl")
    scaler_path = os.path.join(models_dir, "gbc_5m_scaler.pkl")

    print("Building dataset...")
    X, y = build_dataset()

    print("Fitting scaler...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("Training GradientBoosting model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(Xs, y)

    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print("Done.")


if __name__ == "__main__":
    main()
