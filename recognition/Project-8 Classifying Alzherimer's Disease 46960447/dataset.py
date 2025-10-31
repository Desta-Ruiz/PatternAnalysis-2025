import os
import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


CLASSES = ["NC", "AD"]  # folder names == labels (your structure)


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