"""
Train ConvNeXt model for Alzheimer's Disease (AD) vs Cognitively Normal (NC) classification.

This script trains a ConvNeXt-based model on brain MRI images to classify between
Alzheimer's Disease and Cognitively Normal patients.
"""

# Standard library imports
import os
import argparse
import random
from typing import Tuple, List

# Scientific computing
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Utilities
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from modules import ADNIConvNeXt
from dataset import find_image_files, ADNIImageDataset, CLASSES


def set_seed(seed: int = 1337):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 1337)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_train_val(items, val_ratio=0.2, seed=1337):
    """
    Split dataset into train and validation sets using stratified sampling.

    Ensures both AD and NC classes are proportionally represented in both splits.

    Args:
        items: List of (image_path, label) tuples
        val_ratio: Fraction of data to use for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices)
    """
    labels = [lbl for _, lbl in items]
    idxs = np.arange(len(items))
    # Use stratified k-fold to ensure balanced class distribution
    skf = StratifiedKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(idxs, labels))
    return idxs[train_idx].tolist(), idxs[val_idx].tolist()


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.

    Args:
        preds: Model predictions (logits)
        targets: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    return (preds.argmax(dim=1) == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        loader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (AdamW)
        device: Device to train on (cuda/cpu)

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss, running_acc = 0.0, 0.0

    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        # Move data to GPU/CPU
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Reset gradients from previous batch
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute predictions
        logits = model(imgs)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate metrics (weighted by batch size)
        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits.detach(), labels) * imgs.size(0)

    # Calculate average metrics across all samples
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()  # Disable gradient computation for faster inference
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation/test data.

    Args:
        model: Neural network model
        loader: Validation data loader
        criterion: Loss function (CrossEntropyLoss)
        device: Device to evaluate on (cuda/cpu)

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
    running_loss, running_acc = 0.0, 0.0

    for imgs, labels, _ in tqdm(loader, desc="Val", leave=False):
        # Move data to GPU/CPU
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass only (no gradient computation)
        logits = model(imgs)
        loss = criterion(logits, labels)

        # Accumulate metrics
        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits, labels) * imgs.size(0)

    # Calculate average metrics
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def plot_curves(train_hist, val_hist, outdir):
    """
    Plot and save training curves (loss and accuracy).

    Args:
        train_hist: List of dicts with training metrics per epoch
        val_hist: List of dicts with validation metrics per epoch
        outdir: Directory to save the plots
    """
    os.makedirs(outdir, exist_ok=True)

    # Plot loss curve
    plt.figure()
    plt.plot([x["loss"] for x in train_hist], label="train")
    plt.plot([x["loss"] for x in val_hist], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # Plot accuracy curve
    plt.figure()
    plt.plot([x["acc"] for x in train_hist], label="train")
    plt.plot([x["acc"] for x in val_hist], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(outdir, "acc_curve.png"))
    plt.close()

def main():
    """Main training function."""
    # ========== Argument Parsing ==========
    parser = argparse.ArgumentParser(description="Project 8: ADNI AD vs CN with ConvNeXt (Colab-ready)")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory with NC/ and AD/ subfolders containing images")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularization weight decay (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3, higher = more regularization)")

    # Model configuration
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze ConvNeXt backbone, only train classifier head")

    # Data configuration
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2 = 20%%)")

    # System configuration
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility (default: 1337)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4, try 0 for Colab)")
    parser.add_argument("--outdir", type=str, default="./runs_adni_convnext",
                        help="Output directory for checkpoints and plots")

    args = parser.parse_args()

    # ========== Setup ==========
    set_seed(args.seed)  # Set random seeds for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ========== Load Dataset ==========
    # Find all image files (JPEG/PNG) in NC/ and AD/ subdirectories
    items = find_image_files(args.data_root)
    if len(items) == 0:
        raise SystemExit(f"No image files found under {args.data_root}. Expected folders: {CLASSES}.")

    print(f"Found {len(items)} image files")
    print(f"Using device: {device}")

    # ========== Split Data ==========
    # Split into train/val with stratified sampling (maintains class balance)
    train_idx, val_idx = split_train_val(items, val_ratio=args.val_ratio, seed=args.seed)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    # Create datasets (train uses augmentation, val does not)
    ds_train = ADNIImageDataset(train_items, train=True, augment=True)
    ds_val   = ADNIImageDataset(val_items, train=False, augment=False)

    # Create data loaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ========== Build Model ==========
    # ConvNeXt-based model with optional frozen backbone
    model = ADNIConvNeXt(num_classes=2, freeze_backbone=args.freeze_backbone, dropout=args.dropout).to(device)

    # Calculate class weights to handle imbalanced dataset
    # Count samples per class
    class_counts = [0, 0]
    for _, label in train_items:
        class_counts[label] += 1

    # Calculate weights (inverse frequency)
    total = sum(class_counts)
    class_weights = torch.tensor([total / (len(class_counts) * count) for count in class_counts],
                                  dtype=torch.float32).to(device)

    # Optional: Boost minority class weight even more if needed
    # Uncomment the line below if model still predicts majority class
    # class_weights[1] *= 1.5  # Give AD class 50% more weight

    print(f"Class distribution: NC={class_counts[0]}, AD={class_counts[1]}")
    print(f"Class weights: NC={class_weights[0]:.4f}, AD={class_weights[1]:.4f}")

    # Loss function with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (AdamW with weight decay for regularization)
    # Only optimize parameters that require gradients (important if backbone is frozen)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    # ========== Training Loop ==========
    best_val_acc = 0.0  # Track best validation accuracy for checkpointing
    train_hist, val_hist = [], []  # Store metrics for plotting

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        tr_loss, tr_acc = train_one_epoch(model, dl_train, criterion, optimizer, device)

        # Evaluate on validation set
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)

        # Store metrics
        train_hist.append({"loss": tr_loss, "acc": tr_acc})
        val_hist.append({"loss": va_loss, "acc": va_acc})

        # Print progress
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        # Save checkpoint if validation accuracy improved
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join(args.outdir, "best.ckpt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "args": vars(args),  # Save training config
                "classes": CLASSES,  # Save class names for inference
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path} (val_acc={best_val_acc:.4f})")

    # ========== Save Results ==========
    # Plot and save training curves
    plot_curves(train_hist, val_hist, args.outdir)
    print("Training complete. Curves saved to:", args.outdir)

if __name__ == "__main__":
    main()
