"""
Dataset loading and preprocessing for Alzheimer's Disease classification.

This module handles loading brain MRI images (JPEG/PNG format) and applying
necessary transformations for model training and inference.
"""

import os
import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# Class names corresponding to folder structure
# NC = Normal Control (Cognitively Normal)
# AD = Alzheimer's Disease
CLASSES = ["NC", "AD"]


def find_image_files(root: str, classes=CLASSES) -> List[Tuple[str, int]]:
    """
    Recursively find all image files organized by class folders.

    Expected directory structure:
        root/
            NC/
                image1.jpg
                image2.png
                ...
            AD/
                image1.jpg
                image2.png
                ...

    Args:
        root: Root directory containing class subdirectories
        classes: List of class names (folder names) to search for

    Returns:
        List of tuples (image_path, class_label_index)
        Example: [('/path/to/NC/img1.jpg', 0), ('/path/to/AD/img2.jpg', 1)]
    """
    out = []
    # Support multiple image formats (case-insensitive)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    for idx, cls in enumerate(classes):
        # For each class, search all subdirectories recursively
        for ext in extensions:
            pattern = os.path.join(root, cls, "**", ext)
            files = glob.glob(pattern, recursive=True)
            # Add (path, label_index) tuples
            out.extend((f, idx) for f in files)

    return out


class ADNIImageDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing brain MRI images.

    Handles JPEG/PNG images that have been pre-extracted from NIfTI files.
    Applies standard preprocessing and optional data augmentation.

    Args:
        items: List of (image_path, label_index) tuples from find_image_files()
        resize: Target image size (default: 224 for ConvNeXt input)
        train: Whether this is training data (affects augmentation)
        augment: Whether to apply data augmentation (only used if train=True)

    Returns:
        Tuple of (image_tensor, label, path) where:
            - image_tensor: Preprocessed image of shape (3, 224, 224)
            - label: Class index (0 for NC, 1 for AD)
            - path: Original file path (useful for debugging)

    Example:
        >>> items = find_image_files('/data/train')
        >>> dataset = ADNIImageDataset(items, train=True, augment=True)
        >>> img, label, path = dataset[0]
        >>> print(img.shape)  # torch.Size([3, 224, 224])
    """

    def __init__(self, items: List[Tuple[str, int]],
                 resize: int = 224,
                 train: bool = True,
                 augment: bool = True):
        self.items = items
        self.resize = resize
        self.train = train
        self.augment = augment

        # Build transformation pipeline
        base = [T.Resize((resize, resize))]  # Resize to model input size

        # Data augmentation (only during training)
        if train and augment:
            base += [
                T.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
                T.RandomRotation(degrees=10)     # Random rotation ±10 degrees
            ]

        # Convert to tensor and normalize using ImageNet statistics
        # (required because we use ImageNet-pretrained ConvNeXt)
        base += [
            T.ToTensor(),  # Convert PIL Image to tensor [0, 1]
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
                std=[0.229, 0.224, 0.225]     # ImageNet std (R, G, B)
            )
        ]

        self.tf = T.Compose(base)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        Load and preprocess a single image.

        Args:
            idx: Index of the image to load

        Returns:
            Tuple of (preprocessed_image, label, path)

        Raises:
            RuntimeError: If image cannot be loaded or processed
        """
        path, label = self.items[idx]

        try:
            # Load image and ensure it's RGB (converts grayscale if needed)
            img = Image.open(path).convert('RGB')

            # Apply transformations (resize, augment, normalize)
            tensor = self.tf(img)  # Output shape: (3, H, W)

            return tensor, label, path

        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path} ({e})")