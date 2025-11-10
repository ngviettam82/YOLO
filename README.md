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
│   └── inference.py        # Run inference
├── utils/                   # Helper utilities
│   └── dataset_utils.py    # Dataset utilities
├── examples/                # Example scripts
│   └── dataset_example.py  # Dataset preparation example
├── train_optimized.py      # Main training script
├── manage_dataset.bat      # Dataset manager (Windows batch)
├── manage_dataset.ps1      # Dataset manager (PowerShell)
├── DATASET_GUIDE.md        # Dataset preparation guide
├── TRAINING_GUIDE.md       # Comprehensive training guide
└── README.md               # This file
```

## 🔧 Quick Start

### 1. Installation

```powershell
# Run automated installation
.\install.ps1

# Or manually:
python3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Dataset (NEW! 🆕)

#### Easy 3-Step Process:

**Step 1: Add your raw images**
```powershell
# Copy all images to raw_dataset folder
cp your_images/* raw_dataset/
```

**Step 2: Auto-split into train/val/test**
```powershell
# Use the interactive dataset manager
.\manage_dataset.bat

# Or command line:
.\.venv\Scripts\Activate.ps1
python scripts/split_dataset.py
```

**Step 3: Label your images** 
```powershell
# Launch annotation tool
python scripts/label_images.py

# Choose from:
# 1. LabelImg (Desktop - Recommended)
# 2. CVAT (Web - Team collaboration)
# 3. Label Studio (Web - Easy setup)
# 4. OpenLabeling (Fast desktop)
# 5. Roboflow (Cloud AI-assisted)
```

📖 [See DATASET_GUIDE.md for detailed instructions](DATASET_GUIDE.md)

### 3. Train Model

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Basic training
python train_optimized.py --data dataset/data.yaml

# With custom config
python train_optimized.py --data dataset/data.yaml --config configs/train_config.yaml

# Custom parameters
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 32 --imgsz 832
```

### 4. Validate & Export

```powershell
# Validate trained model
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml

# Export to ONNX
python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx
```

### 5. Run Inference

```powershell
# Single image
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source image.jpg

# Video
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source video.mp4

# Webcam
python scripts/inference.py --model runs/train_xxx/weights/best.pt --source 0 --show
```

## 📊 Performance Optimization

### For Maximum Accuracy

```yaml
model: yolo11l.pt
image_size: 1024
batch_size: 16
epochs: 800
```

### For Maximum Speed

```yaml
model: yolo11n.pt
image_size: 640
batch_size: 32
epochs: 300
```

### For Balanced Performance (Recommended)

```yaml
model: yolo11m.pt
image_size: 832
batch_size: 32
epochs: 500
```

## 🎯 GPU Recommendations

| GPU | VRAM | Recommended Batch Size | Image Size |
|-----|------|------------------------|------------|
| RTX 3060 | 12GB | 20 | 832 |
| RTX 4060 Ti | 8GB | 16 | 832 |
| RTX 4070 Ti | 12GB | 24 | 832 |
| RTX 4070 Ti Super | 16GB | 32 | 832 |
| RTX 4080 | 16GB | 32 | 832 |
| RTX 4090 | 24GB | 48 | 832 |
| **RTX 5080** | **16GB** | **40-48** | **832-1024** |
| RTX 5090 | 32GB | 80-96 | 1024 |

## 📚 Documentation

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive documentation including:

- Detailed installation instructions
- Dataset preparation guide
- Training optimization tips
- Troubleshooting common issues
- Performance benchmarks
- Best practices

## 🔑 Key Features

### Advanced Training Configuration

- **Automatic Mixed Precision (AMP)**: FP16 training for 2x speed
- **Cosine Learning Rate**: Smooth learning rate scheduling
- **Early Stopping**: Automatic stop when no improvement
- **Multi-Scale Training**: Better generalization
- **Advanced Augmentation**: Mosaic, MixUp, Copy-Paste

### Production-Ready Code

- Clean, modular architecture
- Comprehensive error handling
- Detailed logging and monitoring
- Automatic checkpointing
- Resume from checkpoint

### Multiple Export Formats

- ONNX (universal compatibility)
- TensorRT (maximum NVIDIA GPU speed)
- TorchScript (PyTorch native)
- OpenVINO (Intel optimization)
- CoreML (Apple devices)
- TFLite (mobile devices)

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce these in train_config.yaml
batch_size: 8
image_size: 640
workers: 4
```

### Training Too Slow

```yaml
# Increase these
batch_size: 32
workers: 12
cache: 'ram'  # Cache dataset in RAM
```

### Low Accuracy

- Increase training epochs
- Use larger model (yolo11l.pt or yolo11x.pt)
- Collect more training data
- Verify label quality
- Increase image size

## 📈 Expected Results

**Good Model Performance:**
- mAP@0.5: > 0.85
- mAP@0.5:0.95: > 0.60
- Precision: > 0.85
- Recall: > 0.80

**Training Time (500 epochs, 1000 images):**
- RTX 4070 Ti Super: 2.5-3 hours
- RTX 4090: 1-1.5 hours

## 🤝 Contributing

This is an optimized training setup. Feel free to customize and extend based on your needs.

## 📄 License

This project uses Ultralytics YOLO which is licensed under AGPL-3.0.

## 🙏 Acknowledgments

- Ultralytics for the amazing YOLO implementation
- PyTorch team for the deep learning framework
- NVIDIA for CUDA and TensorRT optimization

---

**Happy Training! 🚀**

For detailed instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
