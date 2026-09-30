"""
model/train.py
──────────────
Training loop for the LSTM + MC Dropout forecaster.

Features
────────
  • AdamW optimiser with configurable LR + weight decay
  • MSELoss
  • Early stopping on validation loss (patience configurable in config.yaml)
  • Saves the best checkpoint to checkpoints/best_model.pt
  • Prints epoch-level train / val loss

Usage
─────
  python -m model.train
  python -m model.train --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from data.loader import load_trace
from model.lstm_model import build_model
from preprocessing.pipeline import PreprocessingPipeline


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── main training function ────────────────────────────────────────────────────

def train(cfg: dict, verbose: bool = True) -> dict:
    """Train the LSTM model.

    Parameters
    ----------
    cfg     : parsed config dict (from config.yaml).
    verbose : if True, print per-epoch progress.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'best_epoch'.
    """
    t_cfg  = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device : {device}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if verbose:
        print("Loading and preprocessing data …")
    df = load_trace(cfg=cfg)
    pipeline = PreprocessingPipeline(cfg)
    splits, scaler = pipeline.fit_transform(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    if verbose:
        print(f"  Train : {X_tr.shape}  Val : {X_val.shape}  Test : {X_te.shape}")

    train_loader = _to_loader(X_tr, y_tr, t_cfg["batch_size"], shuffle=True)
    val_loader   = _to_loader(X_val, y_val, t_cfg["batch_size"], shuffle=False)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg.get("weight_decay", 1e-4),
    )

    # ── 3. Training loop with early stopping ──────────────────────────────────
    patience   = t_cfg["early_stopping_patience"]
    best_val   = float("inf")
    no_improve = 0
    best_epoch = 0

    ckpt_dir = Path(t_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    history: dict[str, list] = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, t_cfg["epochs"] + 1):
        # ── train ──
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(criterion(pred, yb).item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:4d}/{t_cfg['epochs']}  "
                f"train={t_loss:.6f}  val={v_loss:.6f}  "
                f"elapsed={elapsed:.1f}s"
            )

        # ── early stopping ──
        if v_loss < best_val:
            best_val   = v_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    best_val,
                    "cfg":         cfg,
                    # Save scaler params so inference can reconstruct it
                    "scaler_min":  scaler.data_min_.tolist(),
                    "scaler_max":  scaler.data_max_.tolist(),
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch} (patience={patience}).")
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val

    if verbose:
        print(f"\n[OK] Training complete.")
        print(f"  Best val loss : {best_val:.6f}  (epoch {best_epoch})")
        print(f"  Checkpoint    : {ckpt_path.resolve()}")

    return history


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM forecaster.")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path  = project_root / args.config

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    history = train(cfg, verbose=True)
