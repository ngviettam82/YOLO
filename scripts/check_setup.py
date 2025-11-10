#!/usr/bin/env python3
"""
Quick Setup Script for YOLO Training
Checks dependencies and GPU availability
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8 or higher is required!")
        return False
    
    if version.minor > 11:
        print("⚠️  Python 3.12+ may have compatibility issues. Python 3.10 is recommended.")
    
    print("✅ Python version is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        print(f"✅ pip: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ pip not found: {e}")
        return False

def check_torch():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ VRAM: {gpu_mem:.1f} GB")
            
            # Recommend batch size
            # Check for RTX 5080/5090 (Blackwell architecture)
            gpu_name = torch.cuda.get_device_name(0)
            is_blackwell = '5080' in gpu_name or '5090' in gpu_name
            
            if is_blackwell and gpu_mem >= 16:
                batch = 40  # RTX 5080 optimized
                print(f"💡 Recommended batch size: {batch} (Blackwell optimized)")
            elif gpu_mem >= 24:
                batch = 48
                print(f"💡 Recommended batch size: {batch}")
            elif gpu_mem >= 16:
                batch = 32
                print(f"💡 Recommended batch size: {batch}")
            elif gpu_mem >= 12:
                batch = 24
                print(f"💡 Recommended batch size: {batch}")
            elif gpu_mem >= 8:
                batch = 16
                print(f"💡 Recommended batch size: {batch}")
            else:
                batch = 8
                print(f"💡 Recommended batch size: {batch}")
            
        else:
            print("⚠️  CUDA not available - training will use CPU (much slower)")
            print("💡 Install PyTorch with CUDA support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        return True
    
    except ImportError:
        print("❌ PyTorch not installed")
        print("💡 Install with:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

def check_ultralytics():
    """Check Ultralytics installation"""
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("❌ Ultralytics not installed")
        print("💡 Install with: pip install ultralytics")
        return False

def check_opencv():
    """Check OpenCV installation"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not installed")
        print("💡 Install with: pip install opencv-python")
        return False

def check_dataset():
    """Check if dataset exists"""
    from pathlib import Path
    
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print("⚠️  Dataset folder not found")
        print("💡 Create dataset folder structure:")
        print("   dataset/")
        print("   ├── train/images/")
        print("   ├── train/labels/")
        print("   ├── val/images/")
        print("   └── val/labels/")
        return False
    
    train_dir = dataset_dir / "train" / "images"
    val_dir = dataset_dir / "val" / "images"
    
    if not train_dir.exists() or not val_dir.exists():
        print("⚠️  Dataset structure incomplete")
        print("💡 Use: python utils/dataset_utils.py validate --dataset dataset")
        return False
    
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    
    print(f"✅ Dataset found:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
    
    if len(train_images) == 0:
        print("❌ No training images found!")
        return False
    
    return True

def main():
    """Main setup check"""
    print("="*80)
    print("YOLO11 Training Setup Check")
    print("="*80)
    print()
    
    print("🔍 Checking system requirements...")
    print()
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    print()
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    print()
    
    # Check PyTorch
    pytorch_ok = check_torch()
    print()
    
    # Check Ultralytics
    ultralytics_ok = check_ultralytics()
    print()
    
    # Check OpenCV
    opencv_ok = check_opencv()
    print()
    
    # Check dataset
    dataset_ok = check_dataset()
    print()
    
    # Summary
    print("="*80)
    print("Setup Summary")
    print("="*80)
    
    all_ok = pytorch_ok and ultralytics_ok and opencv_ok
    
    if all_ok and dataset_ok:
        print("✅ All requirements met! Ready to train.")
        print()
        print("🚀 Start training with:")
        print("   python train_optimized.py --data dataset/your_data.yaml")
        print()
        print("📚 Read the training guide:")
        print("   TRAINING_GUIDE.md")
    
    elif all_ok:
        print("✅ Dependencies installed")
        print("⚠️  Dataset not ready")
        print()
        print("📁 Prepare your dataset first:")
        print("   1. Create dataset/train and dataset/val folders")
        print("   2. Add images and labels")
        print("   3. Run: python utils/dataset_utils.py validate --dataset dataset")
    
    else:
        print("❌ Some dependencies missing")
        print()
        print("💡 Quick install all dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*80)

if __name__ == "__main__":
    main()
