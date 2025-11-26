# DBNet for Scanned Receipt Text Localization

![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python)

Real-time text detection on scanned receipts using Differentiable Binarization Network (DBNet) with ResNet-18 backbone and Feature Pyramid Network.

---

## 🎯 Overview

This repository implements **DBNet (Differentiable Binarization Network)** for text localization in scanned receipt images. DBNet uses a novel differentiable binarization approach that makes the entire text detection pipeline end-to-end trainable, eliminating the need for post-processing steps.

### Key Innovation: Differentiable Binarization

Traditional text detection methods apply fixed thresholds during post-processing, which cannot be optimized during training. DBNet introduces:

- **Adaptive Thresholding**: Pixel-wise learnable thresholds
- **Differentiable Approximation**: Uses a differentiable step function
- **End-to-End Training**: Binarization becomes part of the optimization process

### Architecture

```
Input Image
    ↓
ResNet-18 (Backbone) - Pretrained on ImageNet
    ↓
FPN (Neck) - Multi-scale feature fusion (256 channels)
    ↓
DBHead - Generates 3 maps:
    ├── Probability Map (Shrink Map)
    ├── Threshold Map (Adaptive)
    └── Binary Map (Differentiable Binarization)
```

---

## ✨ Features

- ✅ **Real-time Performance**: 43 FPS on standard GPU
- ✅ **End-to-End Trainable**: No post-processing required
- ✅ **Multi-Scale Detection**: FPN handles varying text sizes
- ✅ **Adaptive Thresholding**: Handles varying image quality
- ✅ **Lightweight**: ResNet-18 backbone (11.2M parameters)
- ✅ **Robust Training**: 92.3% loss reduction over 50 epochs
- ✅ **High Accuracy**: 99.7% training accuracy, 0.971 IoU

---

## 📊 Results

### Training Performance

| Metric | Epoch 1 | Epoch 50 | Improvement |
|--------|---------|----------|-------------|
| **Training Loss** | 2.536 | 0.196 | 92.3% ↓ |
| **Accuracy** | 51.2% | 99.7% | +48.5% |
| **IoU (Shrink Map)** | 0.059 | 0.971 | +0.912 |
| **Shrink Map Loss** | 0.716 | 0.028 | 96.1% ↓ |
| **Threshold Map Loss** | 0.080 | 0.017 | 78.8% ↓ |
| **Binary Map Loss** | 0.553 | 0.017 | 97.0% ↓ |

### Training Phases

1. **Rapid Convergence (Epochs 1-10)**: 68.7% loss reduction
2. **Refinement (Epochs 11-30)**: 40.6% additional reduction
3. **Fine-Tuning (Epochs 31-50)**: 55.3% further improvement

**Total Training Time**: 1.62 hours (50 epochs on GPU)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+

### Clone Repository

```
git clone https://github.com/bunnythewiz/TASK_1.git
cd TASK_1
```

### Create Virtual Environment

```
# Using conda
conda create -n dbnet python=3.13
conda activate dbnet

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### Install Dependencies

```
pip install -r requirement.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Package Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pyyaml>=6.0
tensorboard>=2.13.0
shapely>=2.0.0
scipy>=1.11.0
imgaug>=0.4.0
tqdm>=4.65.0
```

---

## 📁 Dataset Preparation

### ICDAR 2015 Dataset

1. **Download Dataset**
   ```
   # Download from: https://rrc.cvc.uab.es/?ch=4&com=downloads
   wget https://rrc.cvc.uab.es/downloads/ch4_training_images.zip
   wget https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip
   ```

2. **Organize Data**
   ```
   datasets/
   ├── train/
   │   ├── img001.jpg
   │   ├── gt_img001.txt
   │   ├── img002.jpg
   │   ├── gt_img002.txt
   │   └── ...
   └── test/
       ├── img001.jpg
       ├── gt_img001.txt
       └── ...
   ```

3. **Generate File Lists**
   ```
   bash generate_lists.sh
   ```

   This creates:
   - `train.txt`: Training image paths and annotations
   - `test.txt`: Test image paths and annotations

### Annotation Format

Ground truth files (`gt_*.txt`) should contain one line per text instance:

```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```

Example:
```
377,117,463,117,465,130,378,130,COFFEE
```

---

## 🏋️ Training

### Single GPU Training

```
bash singlel_gpu_train.sh
```

Or directly:

```
python train.py \
  --config config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml \
  --resume_checkpoint output/checkpoint/model_latest.pth  # Optional
```

### Training Configuration

Edit `config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml`:

```
arch:
  backbone:
    type: resnet18
    pretrained: True
  neck:
    type: FPN
    inner_channels: 256
  head:
    type: DBHead
    k: 50

trainer:
  epochs: 50
  log_iter: 10
  save_interval: 10

optimizer:
  type: Adam
  lr: 0.001
  weight_decay: 0.0

loss:
  type: DBLoss
  alpha: 1.0
  beta: 10
  ohem_ratio: 3
```

### Monitor Training

```
tensorboard --logdir output/DBNet_Receipt_Detection/
```

## 📂 Project Structure

```
TASK_1/
├── base/                          # Base classes
│   ├── base_dataset.py           # Dataset base class
│   └── base_trainer.py           # Trainer base class
├── config/                        # Configuration files
│   ├── icdar2015_resnet18_FPN_DBhead_polyLR.yaml
│   ├── icdar2015_resnet50_FPN_DBhead_polyLR.yaml
│   └── open_dataset_resnet18_FPN_DBhead_polyLR.yaml
├── data_loader/                   # Data loading and augmentation
│   ├── dataset.py                # ICDAR dataset loader
│   └── modules/
│       ├── augment.py            # Data augmentation
│       ├── make_shrink_map.py    # Shrink map generation
│       └── make_border_map.py    # Border map generation
├── models/                        # Model architectures
│   ├── backbone/                 # Backbone networks
│   │   ├── resnet.py             # ResNet-18/50
│   │   └── resnest.py            # ResNeSt
│   ├── neck/                     # Neck modules
│   │   └── FPN.py                # Feature Pyramid Network
│   ├── head/                     # Detection heads
│   │   └── DBHead.py             # Differentiable Binarization Head
│   ├── losses/                   # Loss functions
│   │   ├── DB_loss.py            # DBNet multi-component loss
│   │   └── basic_loss.py         # Basic loss components
│   └── model.py                  # Model builder
├── post_processing/               # Post-processing
│   └── seg_detector_representer.py  # Convert predictions to boxes
├── utils/                         # Utilities
│   ├── metrics.py                # Evaluation metrics
│   ├── schedulers.py             # Learning rate schedulers
│   └── util.py                   # General utilities
├── output/                        # Training outputs
│   └── DBNet_Receipt_Detection/
│       ├── checkpoint/           # Model checkpoints
│       │   ├── model_best.pth
│       │   └── model_latest.pth
│       └── tensorboard/          # TensorBoard logs
├── train.py                       # Training script
├── eval.py                        # Evaluation script
├── predict.py                     # Inference script
├── requirement.txt                # Python dependencies
├── environment.yml                # Conda environment
├── README.md                      # This file
└── LICENSE.md                     # Apache 2.0 License
```
