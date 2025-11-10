# YOLO11 Training Setup Guide
## Maximum Performance & Accuracy Configuration

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

**Optimal:**
- GPU: NVIDIA RTX 5080 or better (16GB+ VRAM)
- CPU: Intel Ultra 7 265K or equivalent (12+ cores)
- RAM: 64GB+
- Storage: 1TB+ NVMe SSD

### Software Requirements

- **Python**: 3.8 - 3.11 (3.10 recommended)
- **CUDA**: 11.8 or 12.1 (for GPU training)
- **cuDNN**: Latest version matching CUDA
- **Operating System**: Windows 10/11, Linux, or macOS

---

## 📦 Installation

### Step 1: Create Virtual Environment

```powershell
# Navigate to YOLO directory
cd YOLO

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate  # Linux/macOS
```

### Step 2: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (for RTX 5080)
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Alternative: For older CUDA versions
# CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
# pip install torch torchvision torchaudio

# Install Ultralytics YOLO
pip install ultralytics

# Install additional dependencies
pip install opencv-python
pip install pyyaml
pip install psutil
pip install tqdm
```

### Step 3: Verify Installation

```powershell
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully!')"
```

**Expected Output (with GPU):**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 5080
Ultralytics YOLO installed successfully!
```

---

## 📁 Dataset Preparation

### Dataset Structure

Your dataset should follow this structure:

```
YOLO/dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/ (optional)
    ├── images/
    └── labels/
```

### Label Format

YOLO uses normalized coordinates in txt files:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`fire.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.25
```
- All values are normalized (0-1)
- `class_id`: Integer starting from 0
- `x_center, y_center`: Center of bounding box
- `width, height`: Size of bounding box

### Using Dataset Utilities

```powershell
# Split existing dataset
python utils/dataset_utils.py split --source "path/to/raw/data" --output "dataset" --train 0.8 --val 0.1 --test 0.1

# Create dataset YAML
python utils/dataset_utils.py create-yaml --output "dataset" --classes fire smoke --name fire_detection

# Validate dataset structure
python utils/dataset_utils.py validate --dataset "dataset"
```

### Create Dataset YAML Manually

Create `dataset/fire_detection.yaml`:

```yaml
path: C:\Users\ADMIN\Documents\Code\SimulationControl\YOLO\dataset
train: train/images
val: val/images
test: test/images

nc: 2  # Number of classes

names:
  0: fire
  1: smoke
```

**Important:** Use absolute paths or relative paths from the YAML file location.

---

## 🚀 Training

### Quick Start Training

```powershell
# Basic training
python train_optimized.py --data dataset/fire_detection.yaml

# With custom config
python train_optimized.py --data dataset/fire_detection.yaml --config configs/train_config.yaml

# Start fresh (no resume)
python train_optimized.py --data dataset/fire_detection.yaml --no-resume
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

### Batch Size Recommendations

| GPU Model | VRAM | Image Size 640 | Image Size 832 | Image Size 1024 |
|-----------|------|----------------|----------------|-----------------|
| RTX 3060 | 12GB | 24 | 16 | 8 |
| RTX 3070 | 8GB | 16 | 12 | 6 |
| RTX 3080 | 10GB | 20 | 16 | 8 |
| RTX 4060 Ti | 8GB | 16 | 12 | 6 |
| RTX 4070 Ti | 12GB | 28 | 20 | 12 |
| RTX 4070 Ti Super | 16GB | 40 | 32 | 20 |
| RTX 4080 | 16GB | 40 | 32 | 20 |
| RTX 4090 | 24GB | 64 | 48 | 32 |
| **RTX 5080** | **16GB** | **48** | **40** | **24** |
| RTX 5090 | 32GB | 96 | 80 | 48 |

### Training Output

Training results will be saved to:
```
YOLO/runs/train_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt        # Best model weights
│   └── last.pt        # Last epoch weights
├── results.png        # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── ...
```

### Monitoring Training

**Training Metrics:**
- **mAP@0.5**: Mean Average Precision at 0.5 IoU (higher is better)
- **mAP@0.5:0.95**: mAP across IoU thresholds (more comprehensive)
- **Precision**: % of correct positive predictions
- **Recall**: % of actual positives found
- **Loss**: Training loss (should decrease)

**Good Training Signs:**
- ✅ Loss decreasing steadily
- ✅ mAP increasing over time
- ✅ Validation metrics improving
- ✅ No severe overfitting (train vs val gap)

**Warning Signs:**
- ⚠️ Loss not decreasing
- ⚠️ Very high train mAP but low val mAP (overfitting)
- ⚠️ Metrics plateauing early
- ⚠️ GPU memory errors

---

## ✅ Validation & Export

### Validate Trained Model

```powershell
# Validate with default settings
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/fire_detection.yaml

# Custom validation
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/fire_detection.yaml --imgsz 832 --batch 16
```

### Export Model

```powershell
# Export to ONNX (recommended)
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

# Export to multiple formats
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx torchscript engine

# Export with half precision (FP16)
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx --half
```

**Export Formats:**
- **ONNX**: Universal format, works with most frameworks
- **TorchScript**: PyTorch native, best for Python deployment
- **TensorRT (engine)**: Maximum speed on NVIDIA GPUs
- **OpenVINO**: Optimized for Intel CPUs
- **CoreML**: For iOS/macOS deployment
- **TFLite**: For Android/mobile deployment

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

# Webcam (camera index 0)
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source 0 --show

# Custom confidence threshold
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source image.jpg --conf 0.5

# Save and display results
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source video.mp4 --save --show
```

---

## 🔥 Optimization Tips

### 1. Maximum Accuracy

```yaml
# configs/train_config.yaml
model: yolo11l.pt          # Use larger model
image_size: 1024           # Higher resolution
batch_size: 16             # As large as GPU allows
epochs: 800                # More training
patience: 150              # More patience
```

**Additional techniques:**
- Use more training data (1000+ images per class)
- Apply strong augmentation
- Train longer with early stopping
- Use test-time augmentation (TTA)

### 2. Maximum Speed

```yaml
# configs/train_config.yaml
model: yolo11n.pt          # Smaller model
image_size: 640            # Lower resolution
batch_size: 32             # Large batch for GPU utilization
```

**Deployment optimization:**
- Export to TensorRT for NVIDIA GPUs
- Use FP16 half precision
- Reduce image size during inference
- Batch inference when possible

### 3. Limited GPU Memory

If you get "CUDA out of memory" errors:

```yaml
# Reduce these parameters
batch_size: 8              # Smaller batch
image_size: 640            # Smaller images
workers: 4                 # Fewer workers
```

Or use gradient accumulation:
- Batch size 8 = real batch 8
- Accumulate 4 steps = effective batch 32

### 4. Small Dataset (< 500 images)

```yaml
epochs: 300                # Fewer epochs to avoid overfitting
patience: 50               # Earlier stopping

# Stronger augmentation
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 20
mixup: 0.2
copy_paste: 0.5
```

### 5. Class Imbalance

If one class has many more examples:

1. **Weighted sampling**: Oversample minority class
2. **Augmentation**: Apply more augmentation to minority class
3. **Class weights**: Adjust loss weights in training
4. **Collect more data**: Best solution

---

## 📊 Performance Benchmarks

### Training Time Estimates (500 epochs, 1000 images)

| GPU | Image Size 640 | Image Size 832 | Image Size 1024 |
|-----|----------------|----------------|-----------------|
| RTX 3060 (12GB) | 3-4 hours | 4-5 hours | 6-8 hours |
| RTX 4070 Ti (12GB) | 2-3 hours | 3-4 hours | 5-6 hours |
| RTX 4070 Ti Super (16GB) | 1.5-2 hours | 2.5-3 hours | 4-5 hours |
| RTX 4080 (16GB) | 1-1.5 hours | 2-2.5 hours | 3-4 hours |
| RTX 4090 (24GB) | 0.5-1 hour | 1-1.5 hours | 2-3 hours |
| **RTX 5080 (16GB)** | **0.8-1.2 hours** | **1.5-2 hours** | **2.5-3.5 hours** |
| RTX 5090 (32GB) | 0.3-0.5 hours | 0.8-1 hour | 1.5-2 hours |

### Inference Speed (FPS)

| Model | RTX 3060 | RTX 4070 Ti Super | RTX 4090 | **RTX 5080** |
|-------|----------|-------------------|----------|--------------|
| yolo11n | 200+ | 300+ | 400+ | **350+** |
| yolo11s | 150+ | 250+ | 350+ | **300+** |
| yolo11m | 100+ | 180+ | 280+ | **220+** |
| yolo11l | 60+ | 120+ | 200+ | **150+** |
| yolo11x | 40+ | 80+ | 150+ | **100+** |

*Tested with image size 640, batch size 1, FP16 precision*

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce batch size: `batch_size: 8` or `4`
2. Reduce image size: `image_size: 640`
3. Close other GPU applications
4. Reduce workers: `workers: 4`
5. Use smaller model: `yolo11s.pt` instead of `yolo11m.pt`

### Issue: Training Too Slow

**Solutions:**
1. Increase batch size (if GPU allows)
2. Reduce image size
3. Enable caching: `cache: 'ram'` or `cache: 'disk'`
4. Use SSD storage for dataset
5. Increase workers: `workers: 8` or `12`

### Issue: Low Accuracy

**Solutions:**
1. Train longer: increase `epochs`
2. Use larger model: `yolo11l.pt` or `yolo11x.pt`
3. Increase image size: `image_size: 1024`
4. Collect more training data
5. Improve data quality (better labels)
6. Apply augmentation
7. Check dataset for errors

### Issue: Overfitting

**Symptoms:** High training accuracy but low validation accuracy

**Solutions:**
1. Collect more training data
2. Increase augmentation
3. Use dropout/regularization
4. Early stopping with patience
5. Reduce model size
6. Reduce training epochs

### Issue: Model Not Learning

**Symptoms:** Loss not decreasing

**Solutions:**
1. Check dataset labels are correct
2. Verify dataset YAML paths
3. Increase learning rate: `lr0: 0.02`
4. Use longer warmup: `warmup_epochs: 10`
5. Check for dataset errors
6. Try different optimizer: `optimizer: 'Adam'`

---

## 📚 Additional Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com
- **YOLO GitHub**: https://github.com/ultralytics/ultralytics
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Installation**: https://developer.nvidia.com/cuda-downloads

---

## 🎓 Best Practices

1. **Always validate your dataset** before training
2. **Start with pretrained models** (transfer learning)
3. **Use version control** for experiments
4. **Monitor training metrics** regularly
5. **Save best models** for different metrics
6. **Test on real-world data** before deployment
7. **Document your experiments** and results
8. **Use early stopping** to prevent overfitting
9. **Validate on diverse data** (different conditions)
10. **Export models** for deployment efficiency

---

## 🏆 Expected Results

**Good Object Detection Model:**
- mAP@0.5: > 0.85 (Excellent: > 0.90)
- mAP@0.5:0.95: > 0.60 (Excellent: > 0.70)
- Precision: > 0.85 (Excellent: > 0.90)
- Recall: > 0.80 (Excellent: > 0.85)
- F1-Score: > 0.82 (Excellent: > 0.87)

**Training typically takes:**
- Small dataset (< 500 images): 0.8-1.5 hours
- Medium dataset (500-2000 images): 1.5-3 hours
- Large dataset (2000+ images): 3-6 hours

*Times are for RTX 5080 (16GB) with optimized settings*

---

## 📝 Quick Reference Commands

```powershell
# Training
python train_optimized.py --data dataset/fire_detection.yaml

# Validation
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/fire_detection.yaml

# Inference
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source test_image.jpg

# Export
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

# Dataset utilities
python utils/dataset_utils.py validate --dataset dataset
```

---

**Good luck with your YOLO training! 🚀**
