# ml/nn_model.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Small, fast MLP:
      in_dim -> 128 -> 64 -> 1 (sigmoid probability that next-return > 0)
    """
    def __init__(self, in_dim: int, dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.do = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        x = self.do(x)
        x = torch.sigmoid(self.out(x))
        return x  # shape (N, 1)


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    pos_weight: float | None = None  # class balance


def train_loop(model: nn.Module,
               train_xy: Tuple[torch.Tensor, torch.Tensor],
               valid_xy: Tuple[torch.Tensor, torch.Tensor],
               cfg: TrainConfig) -> None:
    Xtr, ytr = train_xy
    Xva, yva = valid_xy

    criterion = nn.BCEWithLogitsLoss(pos_weight=(torch.tensor([cfg.pos_weight]) if cfg.pos_weight else None))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    n = Xtr.size(0)
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i+cfg.batch_size]
            xb = Xtr[idx]
            yb = ytr[idx].view(-1, 1)
            opt.zero_grad()
            logits = model.out(F.relu(model.fc2(F.relu(model.fc1(xb)))))
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # quick valid AUC-ish proxy: accuracy around 0.5
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).view(-1)
            acc = ((pv > 0.5) == (yva > 0.5)).float().mean().item()
            tl = sum(losses) / max(len(losses), 1)
        model.train()
        print(f"epoch {epoch:02d} | train_loss={tl:.4f} valid_acc@0.5={acc:.3f}")
