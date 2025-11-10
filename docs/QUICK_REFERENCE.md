# Quick Reference

Fast command lookup and workflow overview.

---

## 📊 One-Page Workflow

```
1. INSTALL               → docs/INSTALLATION.md
   .\install.ps1

2. DATASET PREP          → docs/DATASET_GUIDE.md
   .\quickstart_dataset.ps1

3. LABEL IMAGES          → docs/DATASET_GUIDE.md
   .\quickstart_label.ps1

4. TRAIN                 → docs/TRAINING_GUIDE.md
   .\quickstart_train.ps1

5. VALIDATE
   python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

6. EXPORT
   python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

7. INFERENCE
   python scripts/inference.py --model runs/train_xxx/weights/best.pt --source image.jpg
```

---

## 🚀 Essential Commands

### Dataset Management
```powershell
# Split images into train/val/test
python scripts/split_dataset.py

# Custom split (80% train, 15% val, 5% test)
python scripts/split_dataset.py --train 0.8 --val 0.15 --test 0.05

# Launch annotation tools
python scripts/label_images.py

# Create dataset config (3 classes)
python scripts/label_images.py --config --num-classes 3
```

### Training
```powershell
# Basic training
python scripts/train_optimized.py --data dataset/data.yaml

# Custom settings
python scripts/train_optimized.py --data dataset/data.yaml --epochs 100 --batch 40 --imgsz 832

# Specific model
python scripts/train_optimized.py --data dataset/data.yaml --model yolo11l.pt

# Resume training
python scripts/train_optimized.py --data dataset/data.yaml --resume

# Fresh start (no resume)
python scripts/train_optimized.py --data dataset/data.yaml --no-resume
```

### Validation & Export
```powershell
# Validate model
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

# Export to ONNX
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx

# Export to multiple formats
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx torchscript engine
```

### Inference
```powershell
# Single image
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source image.jpg

# Folder of images
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source folder/

# Video
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source video.mp4

# Webcam
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source 0 --show
```

### Utilities
```powershell
# Check setup
python scripts/check_setup.py

# Monitor training with TensorBoard
tensorboard --logdir runs/detect
```

---

## 📁 File Locations

| Purpose | Location |
|---------|----------|
| Raw images | `raw_dataset/` |
| Split images | `dataset/images/{train,val,test}/` |
| Annotations | `dataset/labels/{train,val,test}/` |
| Config | `dataset/data.yaml` |
| Training config | `configs/train_config.yaml` |
| Results | `runs/train_YYYYMMDD_HHMMSS/` |
| Best model | `runs/train_xxx/weights/best.pt` |

---

## 🎯 RTX 5080 Recommendations

**Best Settings:**
- Model: `yolo11m.pt` (or `yolo11l.pt` for higher accuracy)
- Image size: `832` (or `1024` for high accuracy)
- Batch size: `40-48` (or `24-32` for conservative)
- Epochs: `100-300` (or `500+` for small datasets)

**Quick Best-Balanced Command:**
```powershell
python scripts/train_optimized.py --data dataset/data.yaml --epochs 200 --batch 40 --imgsz 832
```

---

## 📖 Full Documentation

- **Installation**: `docs/INSTALLATION.md`
- **Dataset**: `docs/DATASET_GUIDE.md`
- **Training**: `docs/TRAINING_GUIDE.md`
- **GPU Tips**: `docs/RTX5080_OPTIMIZED.md`
- **Main**: `README.md`

---

**Need more details? See the full guides above! 📚**
