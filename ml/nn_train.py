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
    Expected columns: ['Open','High','Low','Close','Volume'] at least.
    """
    out = df.copy()

    # Basic returns
    out["return_1"] = out["Close"].pct_change()
    out["return_5"] = out["Close"].pct_change(5)
    out["return_10"] = out["Close"].pct_change(10)

    # Rolling statistics
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["std_10"] = out["Close"].rolling(10).std()
    out["std_20"] = out["Close"].rolling(20).std()

    # RSI
    delta = out["Close"].diff()
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (-delta.where(delta < 0, 0)).abs()
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
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
                "PriceDirectionModel is not available because PyTorch is not installed "
                "in this environment. Use it only in a local environment with torch."
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
