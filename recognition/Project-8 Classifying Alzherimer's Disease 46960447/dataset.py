import os
import glob
from typing import List, Tuple, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import warnings


CLASSES = ["NC", "AD"]  # folder names == labels (your structure)


def find_nii_files(root: str, classes=CLASSES) -> List[Tuple[str, int]]:
    """
    Find all .nii/.nii.gz files under root/{class_name}/**
    Returns list[(path, label_idx)]
    """
    out = []
    for idx, cls in enumerate(classes):
        p1 = os.path.join(root, cls, "**", "*.nii")
        p2 = os.path.join(root, cls, "**", "*.nii.gz")
        files = glob.glob(p1, recursive=True) + glob.glob(p2, recursive=True)
        out.extend((f, idx) for f in files)
    return out


def find_image_files(root: str, classes=CLASSES) -> List[Tuple[str, int]]:
    """
    Find all image files (jpg, jpeg, png) under root/{class_name}/**
    Returns list[(path, label_idx)]
    """
    out = []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for idx, cls in enumerate(classes):
        for ext in extensions:
            pattern = os.path.join(root, cls, "**", ext)
            files = glob.glob(pattern, recursive=True)
            out.extend((f, idx) for f in files)
    return out


def find_nii_files_split(root_with_split: str, split: str, classes=CLASSES):
    """
    root_with_split = ".../ADNI/AD_NC"
    split = "train" or "test"
    Returns list[(path, label_idx)]
    """
    import os, glob
    out = []
    split_root = os.path.join(root_with_split, split)
    for idx, cls in enumerate(classes):
        p1 = os.path.join(split_root, cls, "**", "*.nii")
        p2 = os.path.join(split_root, cls, "**", "*.nii.gz")
        files = glob.glob(p1, recursive=True) + glob.glob(p2, recursive=True)
        out.extend((f, idx) for f in files)
    return out


def robust_minmax(x: np.ndarray, qmin=1, qmax=99, eps=1e-6) -> np.ndarray:
    """Robust min-max normalize using percentiles to damp outliers."""
    lo, hi = np.percentile(x, [qmin, qmax])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)

def to_uint8_rgb(slice2d: np.ndarray) -> np.ndarray:
    """Map a single 2D slice [H,W] to 3-channel uint8 image for torchvision."""
    x = robust_minmax(slice2d.astype(np.float32))  # 0..1
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([x, x, x], axis=-1)  # H,W,3
    return rgb

class ADNISliceDataset(Dataset):
    """
    One sample per 3D volume by extracting a representative 2D slice.
    Strategy: 'center' -> axial middle slice
              'three_slice_rgb' -> stack 3 nearby axial slices into RGB channels
    """
    def __init__(self, items: List[Tuple[str, int]],
                 slice_strategy: str = "center",
                 resize: int = 224,
                 train: bool = True,
                 augment: bool = True):
        self.items = items
        self.slice_strategy = slice_strategy
        self.resize = resize
        self.train = train
        self.augment = augment

        # Torchvision transforms
        base = [T.ToPILImage(), T.Resize((resize, resize))]
        if train and augment:
            base += [T.RandomHorizontalFlip(p=0.5),
                     T.RandomRotation(degrees=10)]
        base += [T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
        self.tf = T.Compose(base)

    def __len__(self):
        return len(self.items)

    def _load_volume(self, path: str) -> np.ndarray:
        try:
            img = nib.load(path)
            vol = img.get_fdata(caching='unchanged')  # float64
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            return vol.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI: {path} ({e})")

    def _pick_center_slice(self, vol: np.ndarray) -> np.ndarray:
        z = vol.shape[2] // 2
        return vol[:, :, z]

    def _three_slice_rgb(self, vol: np.ndarray) -> np.ndarray:
        zc = vol.shape[2] // 2
        z_indices = [max(0, zc - 1), zc, min(vol.shape[2] - 1, zc + 1)]
        chans = []
        for zi in z_indices:
            chans.append(robust_minmax(vol[:, :, zi]))
        chans = [(c * 255.0).clip(0,255).astype(np.uint8) for c in chans]
        rgb = np.stack(chans, axis=-1)  # H,W,3 with different slices as RGB
        return rgb

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        vol = self._load_volume(path)

        if self.slice_strategy == "three_slice_rgb":
            img_rgb = self._three_slice_rgb(vol)
        else:
            sl = self._pick_center_slice(vol)
            img_rgb = to_uint8_rgb(sl)

        tensor = self.tf(img_rgb)  # 3xHxW
        return tensor, label, path


class ADNIImageDataset(Dataset):
    """
    Simple dataset for pre-extracted 2D image slices (JPEG/PNG).
    Works with already preprocessed images.
    """
    def __init__(self, items: List[Tuple[str, int]],
                 resize: int = 224,
                 train: bool = True,
                 augment: bool = True):
        self.items = items
        self.resize = resize
        self.train = train
        self.augment = augment

        # Torchvision transforms
        base = [T.Resize((resize, resize))]
        if train and augment:
            base += [T.RandomHorizontalFlip(p=0.5),
                     T.RandomRotation(degrees=10)]
        base += [T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
        self.tf = T.Compose(base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        try:
            # Load image and convert to RGB (handles grayscale too)
            img = Image.open(path).convert('RGB')
            tensor = self.tf(img)  # 3xHxW
            return tensor, label, path
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path} ({e})")