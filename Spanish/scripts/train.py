"""
Training script for Spanish MNIST character recognition.

Usage:
    # Train CNN (default)
    python scripts/train.py

    # Train LeNet-5
    python scripts/train.py --model lenet

    # Custom settings
    python scripts/train.py --model cnn --epochs 50 --batch_size 128 --lr 1e-3

    # Evaluate a saved checkpoint
    python scripts/train.py --evaluate --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.spanish_mnist_dataset import SpanishMNISTDataset, get_train_transform, get_eval_transform
from models.cnn_model import CNNModel
from models.lenet_model import LeNet5Spanish
from utils.metrics import accuracy, print_summary


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "csv_file":    "data/raw/spanish_mnist.csv",
    "img_dir":     "data/raw",
    "checkpoint_dir": "checkpoints",
    "num_classes": 80,
    "epochs":      30,
    "batch_size":  64,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "patience":    7,
    "seed":        42,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "cnn":
        return CNNModel(num_classes=num_classes)
    elif name == "lenet":
        return LeNet5Spanish(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'cnn' or 'lenet'.")


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total * 100.0, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    set_seed(cfg["seed"])
    device = get_device()
    print(f"\n🖥️  Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_ds = SpanishMNISTDataset(cfg["csv_file"], cfg["img_dir"], split="train",
                                   transform=get_train_transform())
    val_ds   = SpanishMNISTDataset(cfg["csv_file"], cfg["img_dir"], split="val",
                                   transform=get_eval_transform())
    test_ds  = SpanishMNISTDataset(cfg["csv_file"], cfg["img_dir"], split="test",
                                   transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print(f"📂 Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────
    model = get_model(args.model, cfg["num_classes"]).to(device)
    print(f"🧠 {model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # ── Training loop ─────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n🚀 Training for {cfg['epochs']} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>12} {'Val Loss':>10} {'Val Acc':>10} {'LR':>10}")
    print("─" * 65)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0
        print(f"{epoch:>6} {tr_loss:>12.4f} {tr_acc:>11.2f}% {vl_loss:>10.4f} {vl_acc:>9.2f}%  {lr:.2e}  ({dt:.1f}s)")

        # Save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": vl_acc,
                "config": cfg,
                "model_name": args.model,
            }, ckpt_path)
            print(f"         ✅ New best: {best_val_acc:.2f}% → saved {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\n⏹  Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    # ── Final test evaluation ─────────────────────────────────
    print(f"\n📊 Loading best checkpoint for test evaluation…")
    ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    id_to_char = test_ds.id_to_char
    print_summary(labels, preds, class_names=id_to_char)
    print(f"🏁 Final Test Accuracy: {test_acc:.2f}%")

    return history


def run_evaluate(args):
    cfg = DEFAULT_CONFIG.copy()
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "cnn")
    model = get_model(model_name, cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = SpanishMNISTDataset(cfg["csv_file"], cfg["img_dir"], split="test",
                                   transform=get_eval_transform())
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    _, acc, preds, labels = evaluate(model, loader, criterion, device)
    print_summary(labels, preds, class_names=test_ds.id_to_char)
    print(f"Test Accuracy: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Spanish MNIST classifier")
    parser.add_argument("--model",      type=str, default="cnn", choices=["cnn", "lenet"])
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--csv_file",   type=str, default="data/raw/spanish_mnist.csv")
    parser.add_argument("--img_dir",    type=str, default="data/raw")
    parser.add_argument("--evaluate",   action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()

    if args.evaluate:
        run_evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
