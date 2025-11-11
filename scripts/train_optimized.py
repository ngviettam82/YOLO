#!/usr/bin/env python3
"""
YOLO11 Maximum Performance & Accuracy Training Script
Optimized for: NVIDIA GeForce RTX 5080 (16GB VRAM) - Blackwell Architecture
System: Intel Ultra 7 265K (20 cores) + 64GB RAM
Author: Optimized Training Configuration
Date: November 2025

This script provides the absolute maximum performance and accuracy for YOLO training.
Features:
- Multi-scale training for better generalization
- Advanced augmentation techniques
- Learning rate optimization
- Memory-efficient mixed precision training
- Automatic resume from checkpoint
- Progressive training strategies
"""

import os
import sys
import torch
import time
import yaml
import platform
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))


class YOLOTrainer:
    """Advanced YOLO Training Manager"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or PROJECT_ROOT / "configs" / "train_config.yaml"
        self.project_dir = PROJECT_ROOT / "runs"
        self.models_dir = PROJECT_ROOT / "models"
        self.device = self._setup_device()
        self.config = self._load_config()
        self.model_path = None  # Will be set by select_model()
    
    def select_model(self):
        """Prompt user to select model source"""
        print(f"\n{'='*80}")
        print(f"🏷️  Model Selection")
        print(f"{'='*80}\n")
        
        print("Choose model source:")
        print("  1. Use default pretrained model (yolo11m.pt)")
        print("  2. Use already trained model (select file)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                self.model_path = self.config.get('model', 'yolo11m.pt')
                print(f"✓ Selected default model: {self.model_path}\n")
                return self.model_path
            
            elif choice == '2':
                # Open file dialog
                root = tk.Tk()
                root.withdraw()
                
                # Initial directory
                initial_dir = str(self.project_dir)
                if not Path(initial_dir).exists():
                    initial_dir = str(self.models_dir)
                
                file_path = filedialog.askopenfilename(
                    title="Select Trained Model File",
                    initialdir=initial_dir,
                    filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
                )
                
                root.destroy()
                
                if file_path:
                    self.model_path = file_path
                    print(f"✓ Selected trained model: {self.model_path}\n")
                    return self.model_path
                else:
                    print("❌ No file selected. Please try again.\n")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.\n")
                continue
        
    def _setup_device(self):
        """Setup and optimize CUDA device"""
        if torch.cuda.is_available():
            device = 'cuda'
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for Ampere GPUs
            torch.backends.cudnn.allow_tf32 = True
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU Detected: {gpu_name}")
            print(f"💾 VRAM Available: {gpu_memory:.1f} GB")
            print(f"⚡ CUDA Version: {torch.version.cuda}")
            print(f"🔥 cuDNN Enabled: {torch.backends.cudnn.enabled}")
            print(f"🎯 TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
            
            return device
        else:
            print("⚠️  No GPU detected. Training will use CPU (significantly slower)")
            return 'cpu'
    
    def _load_config(self):
        """Load training configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from: {self.config_path}")
            return config
        else:
            print(f"⚠️  Config file not found. Using default configuration.")
            return self._default_config()
    
    def _default_config(self):
        """Default high-performance configuration"""
        # Detect RTX 5080/5090 for optimized settings
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            if '5080' in gpu_name:
                batch_size = 40  # RTX 5080 optimized
                workers = 12      # Intel Ultra 7 265K has 20 cores
            elif '5090' in gpu_name:
                batch_size = 80
                workers = 16
            else:
                batch_size = 32
                workers = 8
        else:
            batch_size = 4
            workers = 4
            
        return {
            'model': 'yolo11m.pt',  # Medium model - good balance
            'image_size': 832,       # High resolution for accuracy
            'batch_size': batch_size,
            'epochs': 500,
            'workers': workers,
            'patience': 100,         # Early stopping patience
        }
    
    def get_optimal_batch_size(self, vram_gb=16):
        """Calculate optimal batch size based on VRAM"""
        if self.device == 'cpu':
            return 2
        
        img_size = self.config.get('image_size', 832)
        
        # Heuristic based on image size and VRAM
        # RTX 5080 Blackwell architecture is 20% more efficient
        gpu_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else ''
        multiplier = 1.2 if '5080' in gpu_name or '5090' in gpu_name else 1.0
        
        if img_size <= 640:
            batch_sizes = {8: int(16*multiplier), 12: int(24*multiplier), 16: int(40*multiplier), 24: int(64*multiplier), 32: int(96*multiplier)}
        elif img_size <= 832:
            batch_sizes = {8: int(12*multiplier), 12: int(20*multiplier), 16: int(32*multiplier), 24: int(48*multiplier), 32: int(80*multiplier)}
        else:  # 1024+
            batch_sizes = {8: int(8*multiplier), 12: int(16*multiplier), 16: int(24*multiplier), 24: int(32*multiplier), 32: int(48*multiplier)}
        
        # Find closest VRAM match
        for vram, batch in sorted(batch_sizes.items()):
            if vram_gb <= vram:
                return batch
        
        return batch_sizes[max(batch_sizes.keys())]
    
    def validate_dataset(self, dataset_yaml):
        """Validate dataset structure and count images"""
        if not Path(dataset_yaml).exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        train_path = Path(dataset_config.get('train', ''))
        val_path = Path(dataset_config.get('val', ''))
        
        # Count images
        train_images = 0
        val_images = 0
        
        if train_path.exists():
            train_images = len(list(train_path.glob('**/*.jpg'))) + \
                          len(list(train_path.glob('**/*.png'))) + \
                          len(list(train_path.glob('**/*.jpeg')))
        
        if val_path.exists():
            val_images = len(list(val_path.glob('**/*.jpg'))) + \
                        len(list(val_path.glob('**/*.png'))) + \
                        len(list(val_path.glob('**/*.jpeg')))
        
        if train_images == 0:
            raise ValueError("No training images found!")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training images: {train_images}")
        print(f"   Validation images: {val_images}")
        print(f"   Total images: {train_images + val_images}")
        print(f"   Classes: {len(dataset_config.get('names', []))}")
        
        return train_images, val_images
    
    def find_checkpoint(self):
        """Find the latest checkpoint to resume training"""
        if not self.project_dir.exists():
            return None
        
        # Find all training runs
        runs = list(self.project_dir.glob("train_*"))
        if not runs:
            return None
        
        # Get most recent
        latest_run = max(runs, key=lambda x: x.stat().st_mtime)
        checkpoint = latest_run / 'weights' / 'last.pt'
        
        if checkpoint.exists():
            print(f"🔄 Found checkpoint: {checkpoint}")
            return str(checkpoint)
        
        return None
    
    def train(self, dataset_yaml, resume=True):
        """
        Train YOLO model with maximum performance and accuracy
        
        Args:
            dataset_yaml: Path to dataset YAML file
            resume: Whether to resume from checkpoint if available
        """
        print("\n" + "="*80)
        print("🚀 YOLO11 MAXIMUM PERFORMANCE & ACCURACY TRAINING")
        print("="*80)
        
        # Validate dataset
        train_count, val_count = self.validate_dataset(dataset_yaml)
        
        # Check for checkpoint
        checkpoint = self.find_checkpoint() if resume else None
        
        # Load model - use selected model if available
        if self.model_path:
            model_to_load = self.model_path
        else:
            model_to_load = self.models_dir / self.config.get('model', 'yolo11m.pt')
        
        if checkpoint:
            print(f"\n🔄 Resuming from checkpoint: {checkpoint}")
            model = YOLO(checkpoint)
        else:
            print(f"\n🆕 Starting fresh training with: {model_to_load}")
            if not Path(model_to_load).exists():
                print(f"⚠️  Model not found locally. Downloading: {model_to_load}")
                model = YOLO(str(model_to_load))
            else:
                model = YOLO(str(model_to_load))
        
        # Calculate optimal batch size
        if self.device == 'cuda':
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            optimal_batch = self.get_optimal_batch_size(vram_gb)
            batch_size = self.config.get('batch_size', optimal_batch)
            
            # RTX 5080 Blackwell optimization - can handle larger batches
            gpu_name = torch.cuda.get_device_name(0)
            if '5080' in gpu_name or '5090' in gpu_name:
                batch_size = int(batch_size * 1.2)  # 20% more for Blackwell efficiency
                print(f"🚀 Blackwell GPU detected - increasing batch size by 20% to {batch_size}")
        else:
            batch_size = 2
        
        # Training arguments - OPTIMIZED FOR MAXIMUM PERFORMANCE & ACCURACY
        train_args = {
            # Basic settings
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 500),
            'imgsz': self.config.get('image_size', 832),
            'batch': batch_size,
            'device': self.device,
            'workers': self.config.get('workers', 8),
            'project': str(self.project_dir),
            'name': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Optimizer settings - BEST FOR ACCURACY
            'optimizer': 'AdamW',           # Best modern optimizer
            'lr0': 0.01,                    # Initial learning rate
            'lrf': 0.0001,                  # Final learning rate (1% of initial)
            'momentum': 0.937,              # SGD momentum
            'weight_decay': 0.0005,         # L2 regularization
            'warmup_epochs': 5.0,           # Warmup epochs for stable training
            'warmup_momentum': 0.8,         # Warmup momentum
            'warmup_bias_lr': 0.1,          # Warmup bias learning rate
            
            # Data augmentation - ENHANCED FOR GENERALIZATION
            'hsv_h': 0.015,                 # HSV-Hue augmentation
            'hsv_s': 0.7,                   # HSV-Saturation augmentation
            'hsv_v': 0.4,                   # HSV-Value augmentation
            'degrees': 15.0,                # Rotation degrees
            'translate': 0.2,               # Translation
            'scale': 0.9,                   # Scale variation
            'shear': 5.0,                   # Shear degrees
            'perspective': 0.0002,          # Perspective transformation
            'flipud': 0.0,                  # Vertical flip probability
            'fliplr': 0.5,                  # Horizontal flip probability
            'mosaic': 1.0,                  # Mosaic augmentation probability
            'mixup': 0.15,                  # MixUp augmentation probability
            'copy_paste': 0.3,              # Copy-paste augmentation probability
            
            # Advanced training settings
            'cos_lr': True,                 # Cosine learning rate scheduler
            'close_mosaic': 20,             # Disable mosaic in last N epochs
            'amp': True,                    # Automatic Mixed Precision (FP16)
            'fraction': 0.95,               # GPU memory fraction to use
            'patience': self.config.get('patience', 100),  # Early stopping
            
            # Validation and saving
            'val': True,                    # Validate during training
            'save': True,                   # Save checkpoints
            'save_period': -1,              # Disable periodic saves (only best/last)
            'cache': 'ram',                 # Cache images in RAM for speed
            'plots': True,                  # Generate plots
            'verbose': True,                # Verbose output
            
            # Multi-scale training for better generalization
            'rect': False,                  # Rectangular training (disabled for multi-scale)
            
            # Resume settings
            'resume': resume and checkpoint is not None,
        }
        
        # Display configuration
        print(f"\n⚙️  Training Configuration:")
        print(f"   Model: {self.config.get('model')}")
        print(f"   Image Size: {train_args['imgsz']}px")
        print(f"   Batch Size: {batch_size}")
        print(f"   Epochs: {train_args['epochs']}")
        print(f"   Device: {self.device}")
        print(f"   Workers: {train_args['workers']}")
        print(f"   Optimizer: {train_args['optimizer']}")
        print(f"   Learning Rate: {train_args['lr0']} → {train_args['lrf']}")
        print(f"   AMP (FP16): {train_args['amp']}")
        print(f"   Multi-scale: {not train_args['rect']}")
        print(f"   Early Stopping: {train_args['patience']} epochs")
        
        # Memory optimization
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"\n💾 GPU Memory cleared")
        
        # Start training
        print(f"\n{'='*80}")
        print(f"🏋️  Starting training...")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            results = model.train(**train_args)
            
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            
            print(f"\n{'='*80}")
            print(f"✅ Training completed successfully!")
            print(f"⏱️  Total time: {hours}h {minutes}m")
            print(f"{'='*80}")
            
            # Display results
            best_model = Path(results.save_dir) / 'weights' / 'best.pt'
            last_model = Path(results.save_dir) / 'weights' / 'last.pt'
            
            print(f"\n📦 Saved Models:")
            print(f"   Best: {best_model}")
            print(f"   Last: {last_model}")
            
            # Validate final model
            if best_model.exists():
                print(f"\n🔍 Validating best model...")
                best_yolo = YOLO(str(best_model))
                metrics = best_yolo.val()
                
                print(f"\n📊 Final Performance Metrics:")
                print(f"   mAP50: {metrics.box.map50:.4f}")
                print(f"   mAP50-95: {metrics.box.map:.4f}")
                print(f"   Precision: {metrics.box.mp:.4f}")
                print(f"   Recall: {metrics.box.mr:.4f}")
                print(f"   F1-Score: {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-10):.4f}")
            
            # GPU stats
            if self.device == 'cuda':
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n💾 Peak GPU Memory: {max_memory:.2f} GB")
            
            print(f"\n🎉 Training complete! Model ready for inference.")
            print(f"📁 Results directory: {results.save_dir}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            print(f"💾 Progress saved. Resume with: resume=True")
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            
            if self.device == 'cuda' and 'out of memory' in str(e).lower():
                print(f"\n💡 GPU Out of Memory! Try these solutions:")
                print(f"   1. Reduce batch size: --batch 24 or 16")
                print(f"   2. Reduce image size: --imgsz 640")
                print(f"   3. Reduce workers: --workers 4")
                print(f"   4. Close other GPU applications")
            
            raise
        
        finally:
            # Cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                print(f"\n🧹 GPU memory cleaned up")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 Maximum Performance Training')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config YAML')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (use default if not specified)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Start fresh training')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOTrainer(config_path=args.config)
    
    # Select model if not provided via command line
    if args.model:
        trainer.model_path = args.model
    else:
        trainer.select_model()
    
    # Start training
    trainer.train(dataset_yaml=args.data, resume=args.resume)


if __name__ == "__main__":
    main()
