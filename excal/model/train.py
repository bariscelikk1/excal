"""Train the exercise classifier.

Usage:
  python -m excal.model.train [--data-root PATH] [--epochs 40]

Saves weights + feature normalization stats to weights/exercise_net.pt.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from excal.model.dataset import load_dataset, split_by_video
from excal.model.net import CLASSES, ExerciseNet

DEFAULT_DATA = (
    Path.home()
    / ".cache/kagglehub/datasets/muhannadtuameh/exercise-recognition-time-series/versions/1"
)
WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "exercise_net.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print("loading dataset...")
    X, y, groups = load_dataset(args.data_root)
    train_m, val_m = split_by_video(y, groups)
    print(f"windows: {len(X)} (train {train_m.sum()}, val {val_m.sum()}), dim {X.shape[1:]}")

    mean = X[train_m].reshape(-1, X.shape[-1]).mean(0)
    std = X[train_m].reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = (X - mean) / std

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xn[train_m]).to(device)
    yt = torch.tensor(y[train_m]).to(device)
    Xv = torch.tensor(Xn[val_m]).to(device)
    yv = torch.tensor(y[val_m]).to(device)

    model = ExerciseNet(X.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(1)
            acc = (pred == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(f"epoch {epoch + 1:3d}  loss {total / len(Xt):.4f}  val_acc {acc:.3f}")

    # per-class accuracy of the best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).cpu().numpy()
    yv_np = yv.cpu().numpy()
    for c, name in enumerate(CLASSES):
        m = yv_np == c
        if m.any():
            print(f"  {name:14s} acc {(pred[m] == c).mean():.3f}  (n={m.sum()})")

    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": best_state, "mean": mean, "std": std, "classes": CLASSES},
        WEIGHTS,
    )
    print(f"best val_acc {best_acc:.3f} -> {WEIGHTS}")


if __name__ == "__main__":
    main()
