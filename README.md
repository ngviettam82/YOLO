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
├── configs/                 # Configuration files
│   ├── train_config.yaml   # Training configuration
│   └── dataset_template.yaml # Dataset YAML template
├── dataset/                 # Your datasets go here
│   ├── train/
│   ├── val/
│   └── test/
├── models/                  # Pre-trained model weights
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   ├── yolo11m.pt
│   └── ...
├── runs/                    # Training outputs
│   ├── train_YYYYMMDD_HHMMSS/
│   └── ...
├── scripts/                 # Utility scripts
│   ├── validate_model.py   # Model validation
│   ├── export_model.py     # Model export
│   └── inference.py        # Run inference
├── utils/                   # Helper utilities
│   └── dataset_utils.py    # Dataset preparation tools
├── train_optimized.py      # Main training script
├── TRAINING_GUIDE.md       # Comprehensive training guide
└── README.md               # This file
```

## 🔧 Quick Start

### 1. Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.8 (for RTX 5080)
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install ultralytics opencv-python pyyaml
```

### 2. Prepare Dataset

```powershell
# Validate your dataset structure
python utils/dataset_utils.py validate --dataset dataset

# Or split an existing dataset
python utils/dataset_utils.py split --source "path/to/data" --output dataset
```

### 3. Train Model

```powershell
# Basic training
python train_optimized.py --data dataset/your_data.yaml

# With custom config
python train_optimized.py --data dataset/your_data.yaml --config configs/train_config.yaml
```

### 4. Validate & Export

```powershell
# Validate trained model
python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/your_data.yaml

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
