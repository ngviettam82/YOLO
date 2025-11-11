# YOLO11 Object Detection Training Project

Professional YOLO11 training setup for maximum GPU performance and accuracy.

## 🚀 Quick Start (4 Easy Steps)

**Just double-click each file in order:**

```batch
1.install.bat     → Setup environment
2.dataset.bat     → Prepare dataset 
3.label.bat       → Label images
4.train.bat       → Train model
```

## ✨ Features

- ✅ **Optimized Training Pipeline**: RTX 5080 GPU optimized
- ✅ **Advanced Augmentation**: Better generalization on small datasets
- ✅ **Automatic Checkpointing**: Resume training on interruption
- ✅ **Multiple Export Formats**: ONNX, TensorRT, TorchScript, CoreML
- ✅ **Comprehensive Utilities**: Dataset, validation, inference, export scripts
- ✅ **Batch Processing**: Process 100+ images automatically
- ✅ **Production Ready**: Clean code structure and best practices

## 📁 Project Structure

```
YOLO/
├── raw_dataset/             # 📥 Place your raw images here
│   ├── image1.jpg
│   └── ...
├── configs/                 # Configuration files
│   ├── train_config.yaml   # Training configuration
│   └── dataset_template.yaml # Dataset YAML template
├── dataset/                 # 📦 Processed datasets (auto-organized)
│   ├── images/
│   │   ├── train/          # 70% training images
│   │   ├── val/            # 20% validation images
│   │   └── test/           # 10% test images
│   ├── labels/
│   │   ├── train/          # Training annotations (YOLO format)
│   │   ├── val/            # Validation annotations
│   │   └── test/           # Test annotations
│   └── data.yaml           # Dataset configuration
├── models/                  # Pre-trained model weights
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   ├── yolo11m.pt
│   └── ...
├── runs/                    # Training outputs
│   ├── train_YYYYMMDD_HHMMSS/
│   └── ...
├── scripts/                 # Utility scripts
│   ├── split_dataset.py    # Split raw images into train/val/test
│   ├── label_images.py     # Launch annotation tools
│   ├── validate_model.py   # Model validation
│   ├── export_model.py     # Model export
│   ├── train_optimized.py  # Main training script
│   ├── check_setup.py      # Verify installation
│   └── inference.py        # Run inference
├── utils/                   # Helper utilities
│   └── dataset_utils.py    # Dataset utilities
├── AutoLabel/               # Optional: Auto-label with pre-trained YOLO
│   ├── run_auto_label.bat  # Auto-label images
│   ├── verify_labels.bat   # Verify labels
│   ├── import_to_label_studio.bat  # Review in web UI
│   ├── README.md           # Auto-label guide
│   ├── QUICKSTART.md       # Quick start
│   └── scripts/
├── docs/                   # 📚 Documentation
│   ├── INSTALLATION.md     # Installation guide
│   ├── DATASET_GUIDE.md    # Dataset preparation
│   ├── TRAINING_GUIDE.md   # Training guide
│   ├── QUICK_REFERENCE.md  # Quick reference
│   └── RTX5080_OPTIMIZED.md # GPU optimization
├── 1.install.bat           # Step 1: Setup environment
├── 2.dataset.bat           # Step 2: Prepare dataset
├── 3.label.bat             # Step 3: Label images
├── 4.train.bat             # Step 4: Train model
├── QUICKSTART.md           # Quick reference (read first!)
├── requirements.txt        # Python dependencies
└── .venv/                  # Virtual environment (created by Step 1)
```

## 🎯 What Each Step Does

| Step | Purpose | Time |
|------|---------|------|
| **1.install.bat** | Setup environment & dependencies | ~10-15 min |
| **2.dataset.bat** | Prepare & organize dataset | ~1-5 min |
| **3.label.bat** | Label images with annotations | 30 min - 2 hrs |
| **4.train.bat** | Train YOLO model | 2-8 hrs |

**See `QUICKSTART.md` for detailed step-by-step instructions.**

---

## 💡 Optional: Auto-Label Images

**Don't want to label manually?** Use a pre-trained YOLO to auto-generate labels:

```batch
cd AutoLabel
run_auto_label.bat          ← Auto-label images
verify_labels.bat           ← Verify quality
import_to_label_studio.bat  ← Review/edit in web UI
```

**Use cases:**
- Too many images to label manually
- Quick baseline labels for verification
- Rapid prototyping and testing

**See:** `AutoLabel/README.md` for detailed guide

---

## 📚 Documentation

| Topic | File |
|-------|------|
| **Quick reference** | `QUICKSTART.md` |
| **Installation** | `docs/INSTALLATION.md` |
| **Dataset prep** | `docs/DATASET_GUIDE.md` |
| **Training guide** | `docs/TRAINING_GUIDE.md` |
| **Commands** | `docs/QUICK_REFERENCE.md` |
| **Labeling help** | `docs/LABELING_TROUBLESHOOTING.md` |
| **GPU optimization** | `docs/RTX5080_OPTIMIZED.md` |
| **Auto-label guide** | `AutoLabel/README.md` |

---
