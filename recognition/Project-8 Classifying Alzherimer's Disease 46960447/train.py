import os
import argparse
import random
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import ADNIConvNeXt
from dataset import find_nii_files, find_image_files, ADNISliceDataset, ADNIImageDataset, CLASSES

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_train_val(items, val_ratio=0.2, seed=1337):
    # stratified split by labels
    labels = [lbl for _, lbl in items]
    idxs = np.arange(len(items))
    skf = StratifiedKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(idxs, labels))
    return idxs[train_idx].tolist(), idxs[val_idx].tolist()

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits.detach(), labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    for imgs, labels, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits, labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def plot_curves(train_hist, val_hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot([x["loss"] for x in train_hist], label="train")
    plt.plot([x["loss"] for x in val_hist], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()
    # Acc
    plt.figure()
    plt.plot([x["acc"] for x in train_hist], label="train")
    plt.plot([x["acc"] for x in val_hist], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(outdir, "acc_curve.png")); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Project 8: ADNI AD vs CN with ConvNeXt (Colab-ready)")
    parser.add_argument("--data_root", type=str, required=True, help="Root with CN/ and AD/ subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--slice_strategy", type=str, default="center", choices=["center","three_slice_rgb"])
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="./runs_adni_convnext")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Try to find image files first (JPEG/PNG), fall back to NIfTI
    items = find_image_files(args.data_root)
    use_images = len(items) > 0

    if not use_images:
        items = find_nii_files(args.data_root)
        if len(items) == 0:
            raise SystemExit(f"No image or NIfTI files found under {args.data_root}. Expected folders: {CLASSES}.")

    print(f"Found {len(items)} {'image' if use_images else 'NIfTI'} files")
    print(f"Using device: {device}")

    train_idx, val_idx = split_train_val(items, val_ratio=args.val_ratio, seed=args.seed)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    if use_images:
        ds_train = ADNIImageDataset(train_items, train=True, augment=True)
        ds_val   = ADNIImageDataset(val_items, train=False, augment=False)
    else:
        ds_train = ADNISliceDataset(train_items, slice_strategy=args.slice_strategy, train=True, augment=True)
        ds_val   = ADNISliceDataset(val_items, slice_strategy=args.slice_strategy, train=False, augment=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = ADNIConvNeXt(num_classes=2, freeze_backbone=args.freeze_backbone, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    train_hist, val_hist = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)

        train_hist.append({"loss": tr_loss, "acc": tr_acc})
        val_hist.append({"loss": va_loss, "acc": va_acc})

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join(args.outdir, "best.ckpt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "args": vars(args),
                "classes": CLASSES,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path} (val_acc={best_val_acc:.4f})")

    plot_curves(train_hist, val_hist, args.outdir)
    print("Training complete. Curves saved to:", args.outdir)

if __name__ == "__main__":
    main()
