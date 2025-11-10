````markdown
# YOLO11 Training Setup Guide

This guide will help you set up and train YOLO11 models with maximum performance and accuracy.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Validation & Export](#validation--export)
6. [Inference](#inference)
7. [Optimization Tips](#optimization-tips)

---

## 🔧 Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD

**Optimal (Your Setup):**
- GPU: NVIDIA RTX 5080 (16GB VRAM)
- CPU: Intel Ultra 7 265K or equivalent (12+ cores)
- RAM: 64GB+
- Storage: 1TB+ NVMe SSD

### Software Requirements

- **Python**: 3.10
- **CUDA**: 12.8
- **Operating System**: Windows 10/11, Linux, or macOS

---

## 📦 Installation

See `docs/INSTALLATION.md` for detailed setup instructions.

Quick installation:

```powershell
.\install.ps1
```

---

## 📁 Dataset Preparation

See `docs/DATASET_GUIDE.md` for detailed dataset instructions.

Quick workflow:

```bash
# 1. Add images to raw_dataset/
# 2. Split dataset
python scripts/split_dataset.py

# 3. Label images
python scripts/label_images.py

# 4. Create config
python scripts/label_images.py --config --num-classes 3

# 5. Update class names in dataset/data.yaml
```

---

## 🚀 Training

### Quick Start Training

```powershell
# Basic training
python train_optimized.py --data dataset/data.yaml

# With custom epochs
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 32

# Start fresh (no resume)
python train_optimized.py --data dataset/data.yaml --no-resume
```

### Training Configuration

Edit `configs/train_config.yaml` to customize training:

```yaml
model: yolo11m.pt      # Model size (n/s/m/l/x)
image_size: 832        # Image size for training
batch_size: 32         # Batch size (adjust for your GPU)
epochs: 500            # Number of epochs
workers: 8             # Data loading workers
patience: 100          # Early stopping patience
```

### Model Selection Guide

| Model | Size | Speed | Accuracy | VRAM | Best For |
|-------|------|-------|----------|------|----------|
| yolo11n.pt | Nano | ⚡⚡⚡⚡⚡ | ⭐⭐ | 4GB | Real-time, Edge devices |
| yolo11s.pt | Small | ⚡⚡⚡⚡ | ⭐⭐⭐ | 6GB | Fast inference |
| yolo11m.pt | Medium | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB+ | **Balanced (Recommended)** |
| yolo11l.pt | Large | ⚡⚡ | ⭐⭐⭐⭐⭐ | 12GB+ | High accuracy |
| yolo11x.pt | XLarge | ⚡ | ⭐⭐⭐⭐⭐ | 16GB+ | Maximum accuracy |

### Batch Size Recommendations for RTX 5080 (16GB)

| Image Size | yolo11n | yolo11s | yolo11m | yolo11l | yolo11x |
|-----------|---------|---------|---------|---------|---------|
| 640 | 96 | 64 | 48 | 32 | 24 |
| 832 | 64 | 48 | 40 | 24 | 16 |
| 1024 | 48 | 32 | 24 | 16 | 12 |

### Training Output

Training results will be saved to:
```
YOLO/runs/train_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt        # Best model weights
│   └── last.pt        # Last epoch weights
├── results.png        # Training curves
├── confusion_matrix.png
└── ...
```

---

## ✅ Validation & Export

### Validate Trained Model

```powershell
# Validate with default settings
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

# Custom validation
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml --imgsz 832 --batch 16
```

### Export Model

```powershell
# Export to ONNX (recommended)
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

# Export to multiple formats
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx torchscript engine
```

---

## 🎯 Inference

### Run Inference

```powershell
# Single image
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source path/to/image.jpg

# Folder of images
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source path/to/images/

# Video file
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source path/to/video.mp4

# Webcam
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source 0 --show
```

---

## 🔥 Optimization Tips

### 1. Maximum Accuracy

```yaml
model: yolo11l.pt          # Use larger model
image_size: 1024           # Higher resolution
batch_size: 16             # As large as GPU allows
epochs: 800                # More training
patience: 150              # More patience
```

### 2. Maximum Speed

```yaml
model: yolo11n.pt          # Smaller model
image_size: 640            # Lower resolution
batch_size: 32             # Large batch for GPU utilization
```

### 3. Limited GPU Memory

```yaml
batch_size: 8              # Smaller batch
image_size: 640            # Smaller images
workers: 4                 # Fewer workers
```

### 4. Small Dataset (< 500 images)

```yaml
epochs: 300                # Fewer epochs to avoid overfitting
patience: 50               # Earlier stopping
```

---

## 📊 Performance Benchmarks

### Training Time (500 epochs, 1000 images, RTX 5080)

| Image Size | yolo11n | yolo11s | yolo11m | yolo11l | yolo11x |
|-----------|---------|---------|---------|---------|---------|
| 640 | 0.5h | 0.8h | 1.2h | 2h | 3h |
| 832 | 0.8h | 1.2h | 1.8h | 3h | 4h |
| 1024 | 1.2h | 1.8h | 2.5h | 4h | 5h |

### Inference Speed (FPS, RTX 5080, batch 1)

| Model | 640 | 832 | 1024 |
|-------|-----|-----|------|
| yolo11n | 450+ | 350+ | 300+ |
| yolo11s | 380+ | 300+ | 250+ |
| yolo11m | 250+ | 200+ | 150+ |
| yolo11l | 180+ | 150+ | 100+ |
| yolo11x | 100+ | 80+ | 60+ |

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```powershell
# Reduce batch size
python train_optimized.py --data dataset/data.yaml --batch 8

# Reduce image size
python train_optimized.py --data dataset/data.yaml --imgsz 640
```

### Issue: Training Too Slow

```powershell
# Increase batch size (if GPU allows)
python train_optimized.py --data dataset/data.yaml --batch 64

# Enable caching
python train_optimized.py --data dataset/data.yaml --cache ram
```

### Issue: Low Accuracy

- Train longer: increase `--epochs`
- Use larger model: `yolo11l.pt` instead of `yolo11m.pt`
- Increase image size: `--imgsz 1024`
- Collect more training data
- Improve label quality
- Check for class imbalance

### Issue: Overfitting

- Collect more training data
- Increase augmentation
- Use early stopping with patience
- Reduce model size
- Reduce training epochs

---

## 📚 Quick Reference Commands

```powershell
# Training
python train_optimized.py --data dataset/data.yaml

# Validation
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

# Inference
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source test.jpg

# Export
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

# Dataset
python scripts/split_dataset.py
python scripts/label_images.py
```

---

## 🏆 Expected Results

**Good Model:**
- mAP@0.5: > 0.85
- mAP@0.5:0.95: > 0.60
- Precision: > 0.85
- Recall: > 0.80

---

**Good luck with your YOLO training! 🚀**

````
