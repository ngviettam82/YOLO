#!/usr/bin/env python3
"""
GPU-Optimized YOLO11m Fire Detection Training Script
Simplified & Stable - Based on RTX 4070 Ti Super proven configuration
Features:
- Model selection: Pretrained (fresh) or Trained (file selection)
- Simple, stable training configuration
- Optimized for RTX 5080 (16GB VRAM)
"""

import os
import sys
import torch
import time
import yaml
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class SimpleYOLOTrainer:
    """Simplified YOLO Trainer with model selection"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.model_path = None
    
    def _setup_device(self):
        """Setup CUDA device"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU Detected: {gpu_name}")
            print(f"💾 VRAM Available: {gpu_memory:.1f} GB")
            print(f"⚡ CUDA Version: {torch.version.cuda}")
            
            return device
        else:
            print("⚠️  No GPU detected. Training will use CPU (significantly slower)")
            return 'cpu'
    
    def select_model(self):
        """Prompt user to select model source and return (model_path, model_type)"""
        print(f"\n{'='*80}")
        print(f"🏷️  Model Selection")
        print(f"{'='*80}\n")
        
        print("Choose model source:")
        print("  1. Use pretrained model (fresh download - start from scratch)")
        print("  2. Use trained model (select file - continue training)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                self.model_path = 'yolo11m.pt'
                print(f"✓ Selected: yolo11m.pt (pretrained)")
                print(f"⏳ Model will be downloaded automatically if not found...\n")
                return self.model_path, 'pretrained'
            
            elif choice == '2':
                # Open file dialog
                root = tk.Tk()
                root.withdraw()
                
                file_path = filedialog.askopenfilename(
                    title="Select Trained Model File",
                    filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
                )
                
                root.destroy()
                
                if file_path:
                    self.model_path = file_path
                    print(f"✓ Selected trained model: {self.model_path}\n")
                    return self.model_path, 'trained'
                else:
                    print("❌ No file selected. Please try again.\n")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.\n")
                continue
    
    def validate_dataset(self, dataset_yaml):
        """Validate dataset"""
        dataset_yaml = Path(dataset_yaml)
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get base path from YAML (if it exists, use it; otherwise use YAML directory)
        base_path = dataset_config.get('path', str(dataset_yaml.parent))
        
        # Build full paths for train and val
        train_rel = dataset_config.get('train', '')
        val_rel = dataset_config.get('val', '')
        
        # Handle both absolute and relative paths
        train_path = Path(base_path) / train_rel if train_rel else None
        val_path = Path(base_path) / val_rel if val_rel else None
        
        # Count images
        train_images = 0
        val_images = 0
        
        if train_path and train_path.exists():
            train_images = len(list(train_path.glob('**/*.jpg'))) + \
                          len(list(train_path.glob('**/*.png')))
        
        if val_path and val_path.exists():
            val_images = len(list(val_path.glob('**/*.jpg'))) + \
                        len(list(val_path.glob('**/*.png')))
        
        if train_images == 0:
            raise ValueError(f"No training images found! Checked: {train_path}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training path: {train_path}")
        print(f"   Training images: {train_images}")
        print(f"   Validation images: {val_images}")
        print(f"   Total images: {train_images + val_images}")
        
        return train_images, val_images
    
    def train(self, dataset_yaml, resume=False, model_type='pretrained'):
        """Train with stable configuration
        
        Args:
            dataset_yaml: Path to dataset YAML
            resume: Whether to resume training (only for trained models)
            model_type: 'pretrained' or 'trained'
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 YOLO11m FIRE DETECTION TRAINING")
        print(f"{'='*80}")
        
        # Validate dataset
        self.validate_dataset(dataset_yaml)
        
        # Load model
        if not Path(self.model_path).exists() and self.model_path != 'yolo11m.pt':
            print(f"❌ Model file not found: {self.model_path}")
            return
        
        print(f"\n🔄 Loading model: {self.model_path}")
        model = YOLO(self.model_path)
        
        # Determine resume behavior based on model type
        # Pretrained models should always start fresh training
        # Trained models can resume if requested
        use_resume = False
        if model_type == 'trained' and resume:
            use_resume = True
            print(f"📌 Resume mode: ENABLED (continuing from last checkpoint)")
        else:
            print(f"📌 Training mode: FRESH START (new training)")
        
        # GPU memory optimization
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory cleared")
        
        # Stable training configuration (proven to work without NaN)
        train_args = {
            'data': str(dataset_yaml),
            'epochs': 500,
            'imgsz': 832,
            'batch': 16,  # Stable batch size
            'device': self.device,
            'workers': 12,
            'project': str(PROJECT_ROOT / 'runs'),
            'name': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Optimizer and learning rate
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Data augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.9,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            
            # Advanced settings
            'cos_lr': True,
            'close_mosaic': 15,
            'amp': True,
            'fraction': 0.95,
            'patience': 50,
            
            # Validation and saving
            'val': True,
            'save': True,
            'save_period': -1,
            'cache': 'ram',
            'plots': True,
            'verbose': True,
            
            # Multi-scale disabled for stability
            'rect': False,
            'resume': use_resume,  # Use computed value instead of parameter
        }
        
        # Display configuration
        print(f"\n⚙️  Training Configuration:")
        print(f"   Image Size: 832px")
        print(f"   Batch Size: 16")
        print(f"   Epochs: 500")
        print(f"   Device: {self.device}")
        print(f"   Workers: 12")
        print(f"   Optimizer: AdamW")
        print(f"   Learning Rate: 0.01 → 0.001")
        print(f"   AMP (FP16): True")
        print(f"   Early Stopping: 50 epochs")
        
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
            raise
        
        finally:
            # Cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                print(f"\n🧹 GPU memory cleaned up")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11m Fire Detection Training')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint (only used with trained models)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimpleYOLOTrainer()
    
    # Select model and get model type
    model_path, model_type = trainer.select_model()
    
    # Start training with model type info
    trainer.train(dataset_yaml=args.data, resume=args.resume, model_type=model_type)


if __name__ == "__main__":
    main()
