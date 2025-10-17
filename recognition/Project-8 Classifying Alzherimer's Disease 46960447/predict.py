import os
import argparse
import torch
from torch.utils.data import DataLoader
from modules import ADNIConvNeXt
from dataset import ADNISliceDataset, find_nii_files, find_nii_files_split

# If user passes the AD_NC parent, prefer the 'test' split automatically
test_split_root = os.path.join(args.data_root, "test")
has_test_split = os.path.isdir(test_split_root) and all(
    os.path.isdir(os.path.join(test_split_root, c)) for c in ["NC","AD"]
)

if has_test_split:
    items = find_nii_files_split(args.data_root, "test", classes=["NC","AD"])
else:
    items = find_nii_files(args.data_root, classes=["NC","AD"])

ds = ADNISliceDataset(items, slice_strategy=args.slice_strategy, train=False, augment=False)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Predict with Project 8 model")
    parser.add_argument("--data_root", type=str, required=True, help="Root with CN/ and AD/ subfolders")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.ckpt")
    parser.add_argument("--slice_strategy", type=str, default="center", choices=["center","three_slice_rgb"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", ["CN", "AD"])

    model = ADNIConvNeXt(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()

    items = find_nii_files(args.data_root, classes=classes)
    ds = ADNISliceDataset(items, slice_strategy=args.slice_strategy, train=False, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    correct, total = 0, 0
    for imgs, labels, paths in dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for p, y, yhat in zip(paths, labels.cpu().tolist(), preds.cpu().tolist()):
            print(f"{p} | gt={classes[y]} pred={classes[yhat]}")

    if total > 0:
        print(f"Total accuracy on provided root: {correct/total:.4f}")

if __name__ == "__main__":
    main()
