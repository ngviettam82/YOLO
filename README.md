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
├── docs/                   # 📚 Documentation
│   ├── INSTALLATION.md     # Installation guide
│   ├── DATASET_GUIDE.md    # Dataset preparation
│   ├── TRAINING_GUIDE.md   # Training guide
│   ├── QUICK_REFERENCE.md  # Quick reference
│   └── RTX5080_OPTIMIZED.md # GPU optimization
├── train_optimized.py      # Main training script
├── manage_dataset.bat      # Dataset manager (Windows batch)
├── manage_dataset.ps1      # Dataset manager (PowerShell)
└── README.md               # This file
```

## � Quick Start

1. **[Install](docs/INSTALLATION.md)** - Setup environment
   ```powershell
   .\install.ps1
   ```

2. **[Prepare Dataset](docs/DATASET_GUIDE.md)** - Add and split images
   ```powershell
   python scripts/split_dataset.py
   python scripts/label_images.py
   ```

3. **[Train](docs/TRAINING_GUIDE.md)** - Start training
   ```powershell
   python train_optimized.py --data dataset/data.yaml
   ```

4. **[Commands Reference](docs/QUICK_REFERENCE.md)** - See all commands

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Setup & troubleshooting
- **[Dataset Guide](docs/DATASET_GUIDE.md)** - Dataset preparation & labeling
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Training, validation & export
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - All commands in one place
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
