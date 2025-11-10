````markdown
# Training Guide

Complete YOLO11 training instructions optimized for RTX 5080.

---

## Prerequisites

Before training, ensure:
1. Installation complete (`docs/INSTALLATION.md`)
2. Dataset prepared (`docs/DATASET_GUIDE.md`)
3. `dataset/data.yaml` configured with your classes
4. `dataset/labels/train/` populated with annotations

---

## 🚀 Training Commands

### Basic Training

```powershell
python train_optimized.py --data dataset/data.yaml
```

### With Custom Settings

```powershell
# 100 epochs, batch 40, image size 832
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 40 --imgsz 832

# No resume (fresh start)
python train_optimized.py --data dataset/data.yaml --no-resume

# Specific model (yolo11s, yolo11m, yolo11l, yolo11x)
python train_optimized.py --data dataset/data.yaml --model yolo11l.pt
```

---

## 🎯 Model Selection

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| yolo11n | ⚡⚡⚡⚡⚡ | ⭐⭐ | 4GB | Edge devices |
| yolo11s | ⚡⚡⚡⚡ | ⭐⭐⭐ | 6GB | Fast inference |
| **yolo11m** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB+ | **Balanced (Recommended)** |
| yolo11l | ⚡⚡ | ⭐⭐⭐⭐⭐ | 12GB+ | High accuracy |
| yolo11x | ⚡ | ⭐⭐⭐⭐⭐ | 16GB+ | Maximum accuracy |

**For RTX 5080:** Start with `yolo11m.pt`, upgrade to `yolo11l.pt` or `yolo11x.pt` for higher accuracy.

---

## 📊 Batch Size Recommendations for RTX 5080 (16GB)

Adjust batch size based on image size and model:

| Image Size | yolo11n | yolo11s | yolo11m | yolo11l | yolo11x |
|-----------|---------|---------|---------|---------|---------|
| 640 | 96 | 64 | 48 | 32 | 24 |
| 832 | 64 | 48 | 40 | 24 | 16 |
| 1024 | 48 | 32 | 24 | 16 | 12 |

**RTX 5080 sweet spot:** Image size 832, batch 40-48, yolo11m-l

---

## ⚙️ Configuration File

Edit `configs/train_config.yaml` to customize defaults:

```yaml
model: yolo11m.pt      # Model to use
image_size: 832        # Input image size
batch_size: 40         # Batch size
epochs: 500            # Total epochs
patience: 100          # Early stopping patience
workers: 8             # Data loading workers
```

---

## 📈 Training Output

Training results saved to: `runs/train_YYYYMMDD_HHMMSS/`

```
runs/train_xxx/
├── weights/
│   ├── best.pt        # Best model (highest mAP)
│   └── last.pt        # Last checkpoint
├── results.png        # Training curves
├── confusion_matrix.png
└── results.csv
```

Monitor training in real-time with TensorBoard:
```powershell
tensorboard --logdir runs/detect
```

---

## ✅ Validation

Validate trained model:

```powershell
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml
```

---

## 📤 Export Model

Export to ONNX (recommended for deployment):

```powershell
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx
```

Export to multiple formats:
```powershell
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx torchscript engine
```

---

## 🎯 Inference

Run predictions:

```powershell
# Single image
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source image.jpg

# Folder of images
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source path/to/images/

# Video
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source video.mp4

# Webcam
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source 0 --show
```

---

## 🔥 Training Strategies

### For Maximum Accuracy

```yaml
model: yolo11x.pt      # Largest model
image_size: 1024       # High resolution
batch_size: 12         # Large effective batch
epochs: 800            # More training
patience: 200          # Late stopping
```

### For Speed

```yaml
model: yolo11n.pt      # Smallest model
image_size: 640        # Lower resolution
batch_size: 64         # Optimize GPU
epochs: 300            # Quick training
```

### For Small Datasets (<500 images)

```yaml
model: yolo11s.pt      # Smaller model
image_size: 640        # Lower resolution
batch_size: 16         # Conservative
epochs: 300            # Prevent overfitting
patience: 50           # Early stop
```

### For Limited GPU Memory

```yaml
batch_size: 8          # Small batch
image_size: 640        # Small images
workers: 4             # Fewer workers
```

---

## 📊 Performance Benchmarks

### Training Time (500 epochs, 1000 images, RTX 5080)

| Image Size | yolo11n | yolo11s | yolo11m | yolo11l | yolo11x |
|-----------|---------|---------|---------|---------|---------|
| 640 | 0.5h | 0.8h | 1.2h | 2h | 3h |
| 832 | 0.8h | 1.2h | 1.8h | 3h | 4h |
| 1024 | 1.2h | 1.8h | 2.5h | 4h | 5h |

### Inference Speed (FPS at batch 1, RTX 5080)

| Model | 640 | 832 | 1024 |
|-------|-----|-----|------|
| yolo11n | 450+ | 350+ | 300+ |
| yolo11s | 380+ | 300+ | 250+ |
| yolo11m | 250+ | 200+ | 150+ |
| yolo11l | 180+ | 150+ | 100+ |
| yolo11x | 100+ | 80+ | 60+ |

---

## 🆘 Troubleshooting

### CUDA Out of Memory

```powershell
# Reduce batch size
python train_optimized.py --data dataset/data.yaml --batch 8

# Reduce image size
python train_optimized.py --data dataset/data.yaml --imgsz 640
```

### Training Too Slow

```powershell
# Increase batch size
python train_optimized.py --data dataset/data.yaml --batch 64

# Enable caching
python train_optimized.py --data dataset/data.yaml --cache ram
```

### Low Accuracy

1. Train longer (increase epochs)
2. Use larger model (yolo11l vs yolo11m)
3. Increase image size to 1024
4. Collect more training data
5. Improve annotation quality
6. Check for class imbalance

### Overfitting (train loss low, val loss high)

1. Collect more data
2. Use data augmentation
3. Reduce model size
4. Train fewer epochs
5. Enable early stopping

---

## 📚 Quick Commands

```powershell
# Train
python train_optimized.py --data dataset/data.yaml

# Resume training
python train_optimized.py --data dataset/data.yaml --resume

# Validate
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

# Inference
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source test.jpg

# Export
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx
```

---

## 🏆 Expected Results

**Good Model:**
- mAP@0.5: > 0.85
- mAP@0.5:0.95: > 0.60
- Precision: > 0.85
- Recall: > 0.80

---

**Good luck training! 🚀**

````
