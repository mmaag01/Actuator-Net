"""Training script for the actuator torque prediction model.

Usage
-----
    python train.py --model gru
    python train.py --model mlp --epochs 100 --batch_size 128
"""

import argparse
import csv
import time

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore

import config
from dataset import get_dataloaders
from models import ActuatorGRU, WindowedMLP


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train actuator torque model")
    parser.add_argument("--model", choices=["mlp", "gru"], required=True,
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str, n_features: int) -> nn.Module:
    if model_type == "mlp":
        return WindowedMLP(
            seq_len=config.SEQ_LEN,
            n_features=n_features,
            hidden_size=config.MLP_HIDDEN_SIZE,
            n_layers=config.MLP_N_LAYERS,
        )
    return ActuatorGRU(
        n_features=n_features,
        hidden_size=config.GRU_HIDDEN_SIZE,
        n_layers=config.GRU_N_LAYERS,
        dropout=config.GRU_DROPOUT,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training / validation steps ───────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()   # OneCycleLR steps per batch
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(model(X), y).item() * len(X)
    return total_loss / len(loader.dataset)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, model_type, n_features):
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    hidden_size = config.MLP_HIDDEN_SIZE if model_type == "mlp" else config.GRU_HIDDEN_SIZE
    n_layers    = config.MLP_N_LAYERS    if model_type == "mlp" else config.GRU_N_LAYERS
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_type": model_type,
            "config": {
                "seq_len":     config.SEQ_LEN,
                "n_features":  n_features,
                "hidden_size": hidden_size,
                "n_layers":    n_layers,
            },
        },
        config.CHECKPOINT_DIR / f"best_model_{model_type}.pt",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    args = parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = get_device()
    print(f"Device : {device}")

    print("Loading and preprocessing data …")
    train_loader, val_loader, test_loader, scaler_X, scaler_y = get_dataloaders(
        batch_size=args.batch_size, save_scalers=True
    )
    print(f"Elapsed Time: {(time.perf_counter()-t_start):.3f} s")
    print(f"\nTrain Samples: {len(train_loader)}")
    print(f"\nTest Samples: {len(test_loader)}")
    print(f"\nValidation Samples: {len(val_loader)}")

    n_features = config.N_FEATURES
    model = build_model(args.model, n_features).to(device)
    print(f"\nModel  : {args.model.upper()} | Parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    # OneCycleLR needs total_steps upfront; early stopping may shorten the run
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.ONE_CYCLE_MAX_LR,
        total_steps=total_steps,
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\nTraining up to {args.epochs} epochs  "
          f"(early-stop patience = {config.PATIENCE})\n")

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)
        history.append((epoch, train_loss, val_loss))

        improved = val_loss < best_val_loss
        marker   = " [saved]" if improved else f" (patience {patience_counter + 1}/{config.PATIENCE})"
        print(f"Epoch {epoch:4d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}{marker} | duration {(time.perf_counter()-t_epoch_start):.3f} s")

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, args.model, n_features)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # Save loss history
    config.RESULTS_DIR.mkdir(exist_ok=True)
    history_path = config.RESULTS_DIR / f"loss_history_{args.model}.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse"])
        writer.writerows(history)

    print(f"\nBest val MSE : {best_val_loss:.6f}")
    print(f"Loss history : {history_path}")
    print(f"Checkpoint   : {config.CHECKPOINT_DIR / f'best_model_{args.model}.pt'}")


if __name__ == "__main__":
    main()
