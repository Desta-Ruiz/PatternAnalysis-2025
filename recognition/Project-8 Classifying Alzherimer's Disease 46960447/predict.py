"""
Inference script for Alzheimer's Disease classification.

This script loads a trained model checkpoint and evaluates it on test data,
reporting overall and per-class accuracy metrics.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import ADNIConvNeXt
from dataset import ADNIImageDataset, CLASSES, find_image_files


@torch.no_grad()  # Disable gradient computation for faster inference
def main():
    """Main prediction function."""
    # ========== Argument Parsing ==========
    parser = argparse.ArgumentParser(description="Predict with Project 8 model")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory with NC/ and AD/ subfolders (or parent of test/ split)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained model checkpoint (best.ckpt)")

    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers (default: 2)")

    args = parser.parse_args()

    # ========== Setup ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== Load Model ==========
    # Load checkpoint file (contains model weights and training config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", CLASSES)  # Get class names from checkpoint
    print(f"Classes: {classes}")

    # Initialize model with same architecture as training
    model = ADNIConvNeXt(num_classes=len(classes))

    # Load trained weights
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Move to device and set to evaluation mode
    model = model.to(device).eval()

    # ========== Detect Data Directory ==========
    # Auto-detect if data_root contains a "test" subdirectory
    # This handles both structures:
    #   1. data_root/test/NC/ and data_root/test/AD/
    #   2. data_root/NC/ and data_root/AD/
    test_split_root = os.path.join(args.data_root, "test")
    has_test_split = os.path.isdir(test_split_root) and all(
        os.path.isdir(os.path.join(test_split_root, c)) for c in classes
    )

    if has_test_split:
        print(f"Detected test split structure, using: {test_split_root}")
        data_root = test_split_root
    else:
        data_root = args.data_root

    # ========== Load Test Data ==========
    # Find all image files in the test directory
    items = find_image_files(data_root, classes=classes)
    if len(items) == 0:
        raise SystemExit(f"No image files found under {data_root}. Expected folders: {classes}.")

    print(f"Found {len(items)} image files")

    # Create dataset (no augmentation for testing)
    ds = ADNIImageDataset(items, train=False, augment=False)

    # Create data loader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ========== Run Inference ==========
    # Initialize metric counters
    correct, total = 0, 0  # Overall accuracy tracking
    class_correct = {c: 0 for c in classes}  # Per-class correct predictions
    class_total = {c: 0 for c in classes}    # Per-class total samples

    print(f"Running predictions on {len(items)} images...")

    # Process batches of images
    for imgs, labels, paths in tqdm(dl, desc="Predicting"):
        # Move batch to device (GPU/CPU)
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass: get model predictions
        logits = model(imgs)

        # Convert logits to class predictions (argmax over class dimension)
        preds = logits.argmax(dim=1)

        # Update overall accuracy
        correct += (preds == labels).sum().item()
        total += labels.numel()

        # Update per-class accuracy
        for y, yhat in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[classes[y]] += 1
            if y == yhat:  # Correct prediction
                class_correct[classes[y]] += 1

    # ========== Display Results ==========
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)

    if total > 0:
        # Overall accuracy
        overall_acc = correct / total
        print(f"\nOverall Accuracy: {correct}/{total} = {overall_acc:.4f} ({100*overall_acc:.2f}%)")

        # Per-class accuracy breakdown
        print(f"\nPer-Class Accuracy:")
        for cls in classes:
            if class_total[cls] > 0:
                acc = class_correct[cls] / class_total[cls]
                print(f"  {cls}: {class_correct[cls]}/{class_total[cls]} = {acc:.4f} ({100*acc:.2f}%)")

    print("="*50)

if __name__ == "__main__":
    main()
