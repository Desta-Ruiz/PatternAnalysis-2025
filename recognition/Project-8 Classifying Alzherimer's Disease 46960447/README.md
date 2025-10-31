# Alzheimer's Disease Classification from MRI Images

## Introduction

This project implements a deep learning solution for classifying Alzheimer's Disease (AD) from brain MRI images. Using transfer learning with pretrained Convolutional Neural Networks, the system can distinguish between Normal Control (NC) subjects and patients with Alzheimer's Disease based on structural brain imaging data.

### Background

Alzheimer's Disease is a progressive neurodegenerative disorder that affects millions worldwide. Early and accurate diagnosis is crucial for treatment planning and patient care. Structural MRI reveals brain atrophy patterns characteristic of Alzheimer's Disease, particularly in regions like the hippocampus and entorhinal cortex. This project leverages deep learning to automatically identify these patterns from medical imaging data.

### Key Features

- **2D Medical Image Classification**: Handles JPEG/PNG brain MRI slices
- **Transfer Learning**: Uses pretrained ConvNeXt architecture with ImageNet weights
- **Data Augmentation**: Random horizontal flip and rotation for improved generalization
- **Stratified Cross-Validation**: Ensures balanced class representation during training
- **Checkpoint Management**: Automatic model saving based on validation performance

## Project Structure

```
Project-8 Classifying Alzherimer's Disease 46960447/
├── README.md          # This file
├── train.py           # Training script with stratified K-fold validation
├── predict.py         # Inference script for trained models
├── modules.py         # Model architecture definitions (ConvNeXt-based)
├── dataset.py         # Data loading and preprocessing utilities
└── requirements.txt   # Python dependencies
```

## Requirements

### Dependencies

- Python 3.8+
- PyTorch 2.0+ (with torchvision)
- numpy
- scikit-learn
- matplotlib
- tqdm

### Installation

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using conda (Recommended)**
```bash
# Create environment
conda create -n alzheimer python=3.10
conda activate alzheimer

# Install PyTorch with CUDA support (adjust cuda version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scikit-learn matplotlib tqdm -c conda-forge
```

## Dataset Structure

The expected dataset structure follows class-labeled folders containing image files (JPEG/PNG):

```
dataset/
├── NC/                  # Normal Control subjects
│   ├── subject001.jpg
│   ├── subject002.jpg
│   └── ...
└── AD/                  # Alzheimer's Disease patients
    ├── subject101.jpg
    ├── subject102.jpg
    └── ...
```

**Alternative: Train/Test Split Structure**
```
dataset/
├── train/
│   ├── NC/
│   └── AD/
└── test/
    ├── NC/
    └── AD/
```

**Supported Image Formats**: `.jpg`, `.jpeg`, `.png` (case-insensitive)

## Usage

### Training a Model

Basic training command:
```bash
python train.py \
  --data_root /path/to/dataset \
  --epochs 20 \
  --batch_size 8 \
  --lr 3e-4 \
  --outdir ./runs_alzheimer
```

Advanced training with options:
```bash
python train.py \
  --data_root /path/to/dataset \
  --epochs 50 \
  --batch_size 16 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --val_ratio 0.2 \
  --dropout 0.3 \
  --freeze_backbone \
  --seed 42 \
  --num_workers 4 \
  --outdir ./runs_alzheimer_frozen
```

**Training Arguments:**
- `--data_root`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 1e-4)
- `--val_ratio`: Validation split ratio (default: 0.2)
- `--dropout`: Dropout rate before final layer (default: 0.0)
- `--freeze_backbone`: Freeze pretrained backbone layers
- `--seed`: Random seed for reproducibility (default: 1337)
- `--num_workers`: DataLoader worker processes (default: 2)
- `--outdir`: Output directory for checkpoints and logs (default: ./runs_adni_convnext)

### Running Inference

Predict on test data:
```bash
python predict.py \
  --data_root /path/to/test_data \
  --ckpt ./runs_alzheimer/best.ckpt \
  --batch_size 8
```

The script automatically:
- Detects train/test split structure if present
- Loads the trained model from checkpoint
- Generates predictions with class labels
- Calculates overall and per-class accuracy metrics

## Model Architecture

### Base Architecture: ConvNeXt-Tiny

The model uses ConvNeXt-Tiny as the backbone, a modern ConvNet architecture that achieves competitive performance with Vision Transformers:

- **Pretrained Weights**: ImageNet-1K (1000 classes)
- **Input Size**: 224×224 RGB images
- **Feature Dimensions**: 768-dimensional features from final layer

### Custom Classification Head

The pretrained classifier is replaced with:
```
Input (768-dim features) → Optional Dropout → Linear (768 → 2) → Output (NC/AD)
```

### Transfer Learning Strategy

Two training modes:
1. **Fine-tuning** (default): All layers trainable
2. **Frozen backbone** (`--freeze_backbone`): Only classification head trained

## Preprocessing Pipeline

### Image Loading

- Loads JPEG/PNG images and converts to RGB format
- Handles both grayscale and color images automatically
- Resize to 224×224 for pretrained model compatibility

### Normalization

**ImageNet Normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Data Augmentation (Training Only)

- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)

## Results

Model performance depends on dataset size and quality. Typical results on ADNI-like datasets:

- **Validation Accuracy**: 75-90%
- **Training Time**: ~5-10 minutes per epoch (GPU)
- **Inference Speed**: ~50-100 samples/second (GPU)

Performance tips:
- Larger datasets benefit from fine-tuning the full model
- Small datasets (<500 samples) work better with frozen backbones
- Increase epochs (50-100) for better convergence on small datasets

## Checkpoints

Saved checkpoints (`best.ckpt`) contain:
- Model weights (`model_state`)
- Optimizer state (`optimizer_state`)
- Best validation accuracy (`val_acc`)
- Training arguments (`args`)
- Class label mapping (`classes`)

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `--batch_size`
- Use `--freeze_backbone` to reduce memory footprint

**Low Accuracy**:
- Increase `--epochs` (50-100 for small datasets)
- Adjust `--lr` (try 1e-4 or 1e-3)
- Add `--dropout 0.3` to reduce overfitting
- Ensure images are properly preprocessed and centered

**No Files Found**:
- Check dataset structure matches `{class_name}/**/*.{jpg,jpeg,png}`
- Verify class names match NC and AD
- Ensure image file extensions are correct

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alzheimer-classification-2025,
  title={Alzheimer's Disease Classification from MRI Scans},
  author={COMP3710 Pattern Analysis and Recognition},
  year={2025},
  institution={University of Queensland}
}
```

## License

This project is part of the COMP3710 coursework at the University of Queensland (2025).

## Acknowledgments

- ConvNeXt architecture: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
- Pretrained weights from torchvision models
- ADNI dataset contributors (if applicable to your dataset)

## Contact

For questions or issues, please open an issue in the repository or contact the course instructors.
