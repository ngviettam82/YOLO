# Quick Reference

All commands at a glance for YOLO training.

---

## 🚀 One-Click Quick Start (Easiest)

Simply double-click these files in the project root **in order**:

| Step | File | What It Does |
|------|------|-------------|
| 1 | `1.install.bat` | Setup Python environment & install PyTorch |
| 2 | `2.label.bat` | Open Label Studio to annotate images |
| 3 | `3.dataset.bat` | Split images into train/val/test |
| 4 | `4.train.bat` | Train with aerial fire/smoke optimized settings |

**Total time:** ~15 minutes setup + 2-8 hours training

---

## ⚙️ Setup & Installation

### Quick Setup (Double-Click)
```batch
1.install.bat
```

### Manual Setup
```batch
python3.10 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts\check_setup.py
```

### Verify Installation
```batch
python scripts\check_setup.py
```

---

## 📦 Dataset Preparation

### Quick Dataset Prep (Double-Click)
```batch
2.dataset.bat
```

### Manual Dataset Split
```batch
.venv\Scripts\activate.bat
python scripts\split_dataset.py --train 0.7 --val 0.2 --test 0.1
```

### Custom Split Ratios
```batch
python scripts\split_dataset.py --train 0.8 --val 0.15 --test 0.05
```

---

## 🎯 Image Annotation

### Quick Labeling (Double-Click)
```batch
2.label.bat
```

### Label Studio (Default)
```batch
.venv\Scripts\activate.bat
python scripts\label_images.py --tool label-studio
```

### Server-Based Import (Large Datasets)
```batch
AutoLabel\import_to_label_studio.bat
```

### Create Dataset Config
```batch
python scripts\label_images.py --config --num-classes 3
```

---

## 🚂 Model Training

### Quick Training (Double-Click)
```batch
4.train.bat
```

### Manual Training with Default Settings
```batch
.venv\Scripts\activate.bat
python scripts\train_optimized.py --data dataset/data.yaml --model yolo11m.pt --epochs 800 --batch 4 --imgsz 1280
```

### Custom Training Parameters
```batch
python scripts\train_optimized.py ^
  --data dataset/data.yaml ^
  --model yolo11m.pt ^
  --epochs 800 ^
  --batch 4 ^
  --imgsz 1280 ^
  --patience 120 ^
  --device 0
```

### Resume Training
```batch
python scripts\train_optimized.py --data dataset/data.yaml --resume
```

### Training with Different Model Sizes
```batch
python scripts\train_optimized.py --model yolo11n.pt  # Nano (fastest)
python scripts\train_optimized.py --model yolo11s.pt  # Small
python scripts\train_optimized.py --model yolo11m.pt  # Medium (default)
python scripts\train_optimized.py --model yolo11l.pt  # Large
python scripts\train_optimized.py --model yolo11x.pt  # Extra Large (most accurate)
```

### Training with Different Image Sizes
```batch
python scripts\train_optimized.py --imgsz 640   # Fast (less VRAM, worse for small objects)
python scripts\train_optimized.py --imgsz 1280  # Default (aerial fire/smoke optimized)
```

### SAHI Sliced Inference (Best for Drone Small Objects)
```batch
python scripts\inference_sahi.py --model runs\best.pt --source image.jpg --slice 640
python scripts\inference_sahi.py --model runs\best.pt --source video.mp4 --slice 640
python scripts\inference_sahi.py --model runs\best.pt --source folder/ --slice 640
```

---

## ✅ Model Validation & Testing

### Validate Model
```batch
.venv\Scripts\activate.bat
python scripts\validate_model.py --model runs/train_001/weights/best.pt --data dataset/data.yaml
```

### Run Inference on Single Image
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source test.jpg
```

### Run Inference on Multiple Images
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source dataset/images/test/
```

### Run Inference on Video
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source video.mp4
```

### Adjust Confidence Threshold
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source test.jpg --conf 0.7
```

---

## 📦 Model Export

### Export to ONNX
```batch
.venv\Scripts\activate.bat
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats onnx
```

### Export to TensorRT
```batch
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats engine
```

### Export Multiple Formats
```batch
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats onnx engine pb
```

### Supported Export Formats
```
pt            # PyTorch native
onnx          # Open Neural Network Exchange
engine        # TensorRT (NVIDIA optimized)
pb            # TensorFlow SavedModel
tflite        # TensorFlow Lite (mobile)
torchscript   # TorchScript (C++ compatible)
```

---

## 🛠️ Environment Management

### Activate Virtual Environment
```batch
.venv\Scripts\activate.bat
```

### Deactivate Virtual Environment
```batch
deactivate
```

### List Installed Packages
```batch
pip list
```

### Install Additional Package
```batch
pip install package_name
```

### Update All Packages
```batch
pip install --upgrade -r requirements.txt
```

---

## 🔍 Monitoring & Debugging

### Check GPU Status
```batch
nvidia-smi
```

### Check GPU Memory Usage
```batch
nvidia-smi -l 1
```

### Monitor CPU Usage During Training
```batch
tasklist | findstr python
```

### View Training Plots
Navigate to: `runs/train_001/plots/` and open `results.png`

### View Confusion Matrix
Navigate to: `runs/train_001/plots/confusion_matrix.png`

---

## 📁 Project Structure Reference

```
YOLO/
├── raw_dataset/              # Your raw images
├── dataset/                  # Processed dataset (auto-created)
│   ├── images/train/, val/, test/
│   ├── labels/train/, val/, test/
│   └── data.yaml
├── configs/
│   └── train_config.yaml     # Training settings
├── scripts/
│   ├── split_dataset.py      # Dataset split script
│   ├── label_images.py       # Annotation launcher
│   ├── train_optimized.py    # Training script
│   ├── validate_model.py     # Validation script
│   ├── inference.py          # Inference script
│   ├── export_model.py       # Export script
│   └── check_setup.py        # Setup verification
├── runs/                     # Training outputs (auto-created)
│   └── train_001/weights/best.pt  # Your trained model
├── docs/                     # Documentation
└── *.bat files               # Batch quick-start scripts
```

---

## 🆘 Common Issues & Solutions

### Python 3.10 Not Found
```batch
# Download from: https://www.python.org/downloads/release/python-3100/
# During install, check "Add Python to PATH"
# Verify:
python3.10 --version
```

### Virtual Environment Errors
```batch
# Delete old venv and recreate
rmdir /s /q .venv
python3.10 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### CUDA Out of Memory
```batch
# Reduce batch size
python scripts\train_optimized.py --batch 32

# Or reduce image size
python scripts\train_optimized.py --imgsz 512

# Or use smaller model
python scripts\train_optimized.py --model yolo11s.pt
```

### Training Not Using GPU
```batch
# Check GPU with
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Dataset Not Found
```batch
# Verify folder structure:
# C:\path\to\YOLO\dataset\data.yaml must exist

# Create it manually:
python scripts\label_images.py --config --num-classes 3
```

---

## 📊 Default Optimized Settings (RTX 5080)

| Setting | Value | Notes |
|---------|-------|-------|
| Model | yolo11m.pt | Good balance |
| Batch Size | 64 | Maximum for RTX 5080 |
| Image Size | 640 | Balanced speed/accuracy |
| Epochs | 1000 | Maximum accuracy |
| Learning Rate | Auto | Optimized per epoch |
| Augmentation | Full | Mosaic, mixup, copy-paste |
| Workers | 16 | Data loading threads |

---

## 📚 Documentation Links

- **Installation:** `docs/INSTALLATION.md`
- **Dataset Guide:** `docs/DATASET_GUIDE.md`
- **Training Guide:** `docs/TRAINING_GUIDE.md`
- **GPU Optimization:** `docs/RTX5080_OPTIMIZED.md`
- **Main README:** `README.md`

---

## 💡 Quick Tips

✅ **Best Practices:**
- Always verify `dataset/data.yaml` has correct class names
- Use `best.pt` for deployment, not `last.pt`
- Export to ONNX for cross-platform compatibility
- Monitor GPU with `nvidia-smi` during training

❌ **Common Mistakes:**
- Forgetting to activate virtual environment
- Using wrong dataset path in data.yaml
- Training with mismatched image sizes
- Not verifying GPU is being used

🎯 **Performance Tips:**
- Clean, well-labeled data → biggest impact
- Larger datasets → better accuracy
- More epochs → better accuracy (with patience)
- Batch size 64 → optimal for RTX 5080

