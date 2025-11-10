# YOLO11 Training Project

Professional YOLO11 training setup for maximum performance and accuracy.

## 🚀 Features

- **Optimized Training Pipeline**: Maximum performance on modern GPUs
- **Advanced Augmentation**: Enhanced data augmentation for better generalization
- **Automatic Checkpointing**: Resume training automatically
- **Multiple Export Formats**: ONNX, TensorRT, TorchScript, and more
- **Comprehensive Utilities**: Dataset preparation, validation, and inference tools
- **Production Ready**: Clean code structure and best practices

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
├── YoloLabel/               # ⭐ Auto-label with pre-trained YOLO
│   ├── auto_label.py       # Main auto-labeling script
│   ├── auto_label.bat      # Quick launcher
│   ├── compare_labels.py   # Compare original vs corrected
│   ├── README.md           # Full documentation
│   └── QUICKSTART.md       # 5-minute quick start
├── docs/                   # 📚 Documentation
│   ├── INSTALLATION.md     # Installation guide
│   ├── DATASET_GUIDE.md    # Dataset preparation
│   ├── TRAINING_GUIDE.md   # Training guide
│   ├── QUICK_REFERENCE.md  # Quick reference
│   └── RTX5080_OPTIMIZED.md # GPU optimization
├── 🚀 1.install.bat             # Step 1: Setup environment
├── 🚀 2.dataset.bat             # Step 2: Prepare dataset
├── 🚀 3.label.bat               # Step 3: Label images
├── 🚀 4.train.bat               # Step 4: Start training
└── README.md               # This file
```

## 🎯 Quick Start Options

### 🚀 NEW: Auto-Label with Pre-trained YOLO (Fastest)

**Skip manual labeling entirely! Use a pre-trained YOLO model to generate labels automatically.**

```bash
YoloLabel/auto_label.bat
```

**What it does:**
1. Loads pre-trained YOLO11 model
2. Detects objects in all images automatically
3. Generates YOLO format labels in `dataset/labels/train/`
4. Creates visualizations for review
5. Ready to verify in Label Studio or train immediately

**Workflow:**
```
1400+ Images
     ↓
Auto-label (5 min) ← YoloLabel/auto_label.bat
     ↓
Auto-generated labels (YOLO format)
     ↓
Review visualizations (optional)
     ↓
Verify in Label Studio (3.label.bat) ← optional
     ↓
Train (4.train.bat)
```

**For detailed guide:** See [YoloLabel README](YoloLabel/README.md) and [Quick Start](YoloLabel/QUICKSTART.md)

---

## 🚀 Quick Start

### Option 1: Double-click BAT Files (Easiest - Recommended ⭐)

**Follow these steps in order:**

1. **Double-click** `1.install.bat` - Setup environment (Python, PyTorch, dependencies)
2. **Double-click** `2.dataset.bat` - Prepare and split dataset
3. **Double-click** `3.label.bat` - Label images with LabelImg
4. **Double-click** `4.train.bat` - Start training with RTX 5080 optimized settings

That's it! All files will activate the virtual environment and run automatically.

### Option 2: Command Line (Manual Control)

**Step 1: Setup Environment**
```bash
1.install.bat
```
Or manually:
```bash
python3.10 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts\check_setup.py
```

**Step 2: Prepare Dataset**
```bash
2.dataset.bat
```
Or manually:
```bash
.venv\Scripts\activate.bat
python scripts\split_dataset.py --train 0.7 --val 0.2 --test 0.1
```

**Step 3: Label Images**
```bash
3.label.bat
```
Or manually:
```bash
.venv\Scripts\activate.bat
python scripts\label_images.py --tool labelimg
```

**Step 4: Train Model**
```bash
4.train.bat
```
Or manually:
```bash
.venv\Scripts\activate.bat
python scripts\train_optimized.py --data dataset/data.yaml --model yolo11m.pt --epochs 1000 --batch 64
```

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Setup & troubleshooting
- **[Dataset Guide](docs/DATASET_GUIDE.md)** - Dataset preparation & labeling
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Training, validation & export
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - All commands in one place
- **[Large Dataset Guide](LARGE_DATASET_GUIDE.md)** - Batch upload guide for 1000+ images ⭐
- **[Label Studio Troubleshooting](LABEL_STUDIO_TROUBLESHOOTING.md)** - Upload issues & solutions
- **[Labeling Troubleshooting](docs/LABELING_TROUBLESHOOTING.md)** - LabelImg alternatives & tools
- **[RTX 5080 Optimization](docs/RTX5080_OPTIMIZED.md)** - GPU-specific tips

## 🤝 Contributing

This is an optimized training setup. Feel free to customize and extend based on your needs.

## 📄 License

This project uses Ultralytics YOLO which is licensed under AGPL-3.0.

## 🙏 Acknowledgments

- Ultralytics for the amazing YOLO implementation
- PyTorch team for the deep learning framework
- NVIDIA for CUDA and TensorRT optimization

---

**Ready to train? Start with [Installation](docs/INSTALLATION.md)! 🚀**
