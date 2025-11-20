import numpy as np
import pandas as pd

# ------------------------------------------------------------
# TRY IMPORTING TORCH (OPTIONAL)
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # running in lightweight environment (Docker)
    nn = None
    optim = None


# ------------------------------------------------------------
# FEATURE BUILDER USED BY make_signals_ml AND executor
# ------------------------------------------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build numeric ML features from OHLCV dataframe.

    This function MUST remain torch-free so Docker can use it.

    It is robust to different column namings, e.g.:
      - 'Open','High','Low','Close','Volume'
      - 'open','high','low','close','volume'
      - 'o','h','l','c','v' (Alpaca style)
    """

    if df is None or df.empty:
        raise ValueError("make_features(): received empty dataframe")

    out = df.copy()

    # --- Normalize & detect column names ---
    cols_lower = {c.lower(): c for c in out.columns}

    def find_col(candidates):
        """
        Return the actual column name from df for any of the
        given candidate names (case-insensitive).
        """
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        raise KeyError(
            f"None of columns {candidates} found in dataframe. "
            f"Available columns: {list(out.columns)}"
        )

    close_col = find_col(["close", "c"])
    open_col = find_col(["open", "o"])
    high_col = find_col(["high", "h"])
    low_col = find_col(["low", "l"])
    vol_col = find_col(["volume", "v"])

    # --- Use the detected columns to build features ---

    close = out[close_col]
    open_ = out[open_col]
    high = out[high_col]
    low = out[low_col]
    vol = out[vol_col]

    # Basic returns
    out["return_1"] = close.pct_change()
    out["return_5"] = close.pct_change(5)
    out["return_10"] = close.pct_change(10)

    # Rolling statistics
    out["ma_10"] = close.rolling(10).mean()
    out["ma_20"] = close.rolling(20).mean()
    out["std_10"] = close.rolling(10).std()
    out["std_20"] = close.rolling(20).std()

    # High/low range type features
    out["hl_range"] = (high - low) / close
    out["oc_range"] = (close - open_) / open_

    # Volume features
    out["vol_zscore_20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (-delta.where(delta < 0, 0)).abs()
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Drop the NaN rows (start of rolling windows)
    out = out.dropna()

    return out


# ------------------------------------------------------------
# OPTIONAL TORCH-BASED MODEL (only defined if torch is present)
# ------------------------------------------------------------
if torch is not None and nn is not None and optim is not None:

    class PriceDirectionModel(nn.Module):
        """
        Tiny neural network classifier.
        Only available when torch is installed.
        """
        def __init__(self, n_features: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)  # up vs down
            )

        def forward(self, x):
            return self.net(x)


    def train_model(X: np.ndarray, y: np.ndarray, epochs=50, lr=1e-3):
        """
        Training function — ONLY used in local/offline training.
        Requires torch to be installed.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        model = PriceDirectionModel(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

        return model


    def save_model(model, path: str):
        torch.save(model.state_dict(), path)


    def load_model(path: str, n_features: int):
        model = PriceDirectionModel(n_features)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

else:
    # Torch is NOT available => define safe stubs so imports still work,
    # but trying to train/load will clearly tell you what's wrong.
    class PriceDirectionModel:  # simple placeholder, not used in Docker
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PriceDirectionModel is not available because PyTorch is not "
                "installed in this environment. Use it only in a local "
                "environment with torch."
            )

    def train_model(*args, **kwargs):
        raise RuntimeError(
            "train_model() requires PyTorch. Install torch locally to train the model."
        )

    def save_model(*args, **kwargs):
        raise RuntimeError(
            "save_model() requires PyTorch. Install torch locally to train/save models."
        )

    def load_model(*args, **kwargs):
        raise RuntimeError(
            "load_model() requires PyTorch. Install torch locally to load models."
        )
