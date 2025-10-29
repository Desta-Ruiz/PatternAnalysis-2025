import os
import argparse
import torch
from torch.utils.data import DataLoader
from modules import ADNIConvNeXt
from dataset import (ADNISliceDataset, ADNIImageDataset, CLASSES,
                     find_nii_files, find_nii_files_split, find_image_files)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Predict with Project 8 model")
    parser.add_argument("--data_root", type=str, required=True, help="Root with NC/ and AD/ subfolders or test split")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.ckpt")
    parser.add_argument("--slice_strategy", type=str, default="center", choices=["center","three_slice_rgb"],
                        help="For NIfTI only - slice extraction strategy")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", CLASSES)
    print(f"Classes: {classes}")

    # Load model
    model = ADNIConvNeXt(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()

    # Try to detect test split structure first
    test_split_root = os.path.join(args.data_root, "test")
    has_test_split = os.path.isdir(test_split_root) and all(
        os.path.isdir(os.path.join(test_split_root, c)) for c in classes
    )

    if has_test_split:
        print(f"Detected test split structure, using: {test_split_root}")
        data_root = test_split_root
    else:
        data_root = args.data_root

    # Try to find image files first (JPEG/PNG), fall back to NIfTI
    items = find_image_files(data_root, classes=classes)
    use_images = len(items) > 0

    if not use_images:
        items = find_nii_files(data_root, classes=classes)
        if len(items) == 0:
            raise SystemExit(f"No image or NIfTI files found under {data_root}. Expected folders: {classes}.")

    print(f"Found {len(items)} {'image' if use_images else 'NIfTI'} files")

    # Create appropriate dataset
    if use_images:
        ds = ADNIImageDataset(items, train=False, augment=False)
    else:
        ds = ADNISliceDataset(items, slice_strategy=args.slice_strategy, train=False, augment=False)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    correct, total = 0, 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}

    print("Running predictions...")
    for imgs, labels, paths in dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        # Track per-class accuracy
        for y, yhat in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[classes[y]] += 1
            if y == yhat:
                class_correct[classes[y]] += 1

    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    if total > 0:
        print(f"\nOverall Accuracy: {correct}/{total} = {correct/total:.4f} ({100*correct/total:.2f}%)")
        print(f"\nPer-Class Accuracy:")
        for cls in classes:
            if class_total[cls] > 0:
                acc = class_correct[cls] / class_total[cls]
                print(f"  {cls}: {class_correct[cls]}/{class_total[cls]} = {acc:.4f} ({100*acc:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
