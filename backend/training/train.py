"""
Offline training script — run once to generate app/weights.npz.
Extra deps (not in requirements.txt): torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Usage (from backend/):
    python -m training.train
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

ARCH = [784, 128, 64, 10]
EPOCHS = 10
LR = 1e-3
BATCH = 256
DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent.parent / "app" / "weights.npz"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))


def train():
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=2)

    model = MLP()
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()

        avg_loss = total_loss / len(train_ds)
        acc = correct / len(val_ds)
        print(f"Epoch {epoch:2d}  loss={avg_loss:.4f}  val_acc={acc:.4f}")

    params = {name: p.detach().numpy() for name, p in model.named_parameters()}
    np.savez(
        OUT_PATH,
        W1=params["net.0.weight"],
        b1=params["net.0.bias"],
        W2=params["net.2.weight"],
        b2=params["net.2.bias"],
        W3=params["net.4.weight"],
        b3=params["net.4.bias"],
    )
    print(f"Saved weights → {OUT_PATH}")


if __name__ == "__main__":
    train()
