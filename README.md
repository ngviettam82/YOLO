# YOLO11 Aerial Fire & Smoke Detection

Drone-optimized YOLO11 training pipeline for detecting fire and smoke from aerial imagery (100m altitude, 1280x720).

## Quick Start (4 Steps)

```batch
1.install.bat     → Setup environment
2.label.bat       → Label images (Label Studio)
3.dataset.bat     → Split dataset
4.train.bat       → Train model
```

## Features

- **Small Object Optimized**: imgsz=1280, aerial augmentation (180° rotation, vertical flip, copy-paste)
- **SAHI Inference**: Sliced inference for detecting tiny fire/smoke from drone altitude
- **Label Studio Integration**: Server-based import for large datasets (no upload limits)
- **Auto-Labeling**: Pre-trained YOLO auto-labels images, review in Label Studio
- **GPU Optimized**: RTX 5080 / CUDA 12.8, mixed precision training
- **Multiple Export Formats**: ONNX, TensorRT, TorchScript, CoreML

## Project Structure

```
YOLO/
├── raw_dataset/             # Place your raw images + labels here
├── configs/                 # Configuration files
│   ├── train_config.yaml   # Training config (imgsz=1280, aerial-optimized)
│   └── dataset_template.yaml
├── dataset/                 # Processed datasets (auto-organized)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── scripts/                 # Core scripts
│   ├── train_optimized.py  # Main training script
│   ├── inference.py        # Standard inference
│   ├── inference_sahi.py   # SAHI sliced inference (small objects)
│   ├── split_dataset.py    # Dataset splitting
│   ├── label_images.py     # Label Studio launcher
│   ├── validate_model.py   # Model validation
│   ├── export_model.py     # Model export
│   └── check_setup.py      # Verify installation
├── AutoLabel/               # Auto-label + Label Studio import
│   ├── run_auto_label.bat
│   ├── verify_labels.bat
│   ├── import_to_label_studio.bat
│   └── scripts/
├── docs/                    # Documentation
├── runs/                    # Training outputs
├── 1.install.bat → 4.train.bat  # Pipeline steps
├── data.yaml               # Dataset config (fire, smoke)
└── requirements.txt
```

## Pipeline Steps

| Step | File | What It Does |
|------|------|-------------|
| 1 | `1.install.bat` | Setup Python environment, PyTorch + CUDA |
| 2 | `2.label.bat` | Open Label Studio for annotation |
| 3 | `3.dataset.bat` | Split raw_dataset into train/val/test |
| 4 | `4.train.bat` | Train YOLO11m (imgsz=1280, aerial optimized) |

**See `QUICKSTART.md` for detailed instructions.**

---

## Auto-Label Workflow (Optional)

For large datasets, auto-label first, then review:

```batch
AutoLabel\run_auto_label.bat          ← Auto-label with pre-trained YOLO
AutoLabel\verify_labels.bat           ← Visual quality check
AutoLabel\import_to_label_studio.bat  ← Review/edit in Label Studio (HTTP-based, no upload limit)
```

See `AutoLabel/README.md` for details.

---

## Inference

### Standard
```batch
python scripts\inference.py --model runs\fire_smoke_xxx\weights\best.pt --source image.jpg
```

### SAHI Sliced (best for small objects from drone)
```batch
python scripts\inference_sahi.py --model runs\fire_smoke_xxx\weights\best.pt --source image.jpg --slice 640
```

---

## Documentation

| Topic | File |
|-------|------|
| Quick start | `QUICKSTART.md` |
| Installation | `docs/INSTALLATION.md` |
| Dataset prep | `docs/DATASET_GUIDE.md` |
| Training | `docs/TRAINING_GUIDE.md` |
| Commands | `docs/QUICK_REFERENCE.md` |
| Labeling issues | `docs/LABELING_TROUBLESHOOTING.md` |
| Auto-labeling | `AutoLabel/README.md` |
| **Quick reference** | `QUICKSTART.md` |
| **Installation** | `docs/INSTALLATION.md` |
| **Dataset prep** | `docs/DATASET_GUIDE.md` |
| **Training guide** | `docs/TRAINING_GUIDE.md` |
| **Commands** | `docs/QUICK_REFERENCE.md` |
| **Labeling help** | `docs/LABELING_TROUBLESHOOTING.md` |
| **GPU optimization** | `docs/RTX5080_OPTIMIZED.md` |
| **Auto-label guide** | `AutoLabel/README.md` |

---
