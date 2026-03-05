#!/usr/bin/env python3
"""
GPU-Optimized YOLO11m Aerial Fire & Smoke Detection Training Script
Optimized for small object detection from drone footage (100m altitude)
Features:
- Model selection: Pretrained (fresh) or Trained (file selection)
- Small object optimizations (high imgsz, tuned augmentation)
- Optimized for RTX 5080 (16GB VRAM) at imgsz=1280
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
        """Prompt user to select model source"""
        print(f"\n{'='*80}")
        print(f"🏷️  Model Selection")
        print(f"{'='*80}\n")
        
        print("Choose model source:")
        print("  1. Use pretrained model (yolo11m - fresh download)")
        print("  2. Load from file (previously trained model)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                self.model_path = 'yolo11m.pt'
                print(f"✓ Selected: yolo11m.pt (pretrained)")
                print(f"⏳ Model will be downloaded automatically if not found...\n")
                return self.model_path
            
            elif choice == '2':
                # Open file dialog
                root = tk.Tk()
                root.withdraw()
                
                file_path = filedialog.askopenfilename(
                    title="Select Model File",
                    filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
                )
                
                root.destroy()
                
                if file_path:
                    self.model_path = file_path
                    print(f"✓ Selected model: {Path(file_path).name}\n")
                    return self.model_path
                else:
                    print("❌ No file selected. Please try again.\n")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.\n")
                continue
    
    def select_training_mode(self):
        """Ask user if they want fresh start or resume training"""
        print(f"\n{'='*80}")
        print(f"⚙️  Training Mode Selection")
        print(f"{'='*80}\n")
        
        print("Choose training mode:")
        print("  1. Fresh start (start training from epoch 1)")
        print("  2. Continue training (resume from checkpoint)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                print(f"✓ Selected: Fresh start training\n")
                return False, None  # resume=False, no checkpoint needed
            
            elif choice == '2':
                print(f"✓ Selected: Continue training\n")
                # Ask user to select checkpoint file
                checkpoint_path = self._select_checkpoint_file()
                if checkpoint_path:
                    return True, checkpoint_path  # resume=True, with checkpoint path
                else:
                    print("❌ No checkpoint selected. Please try again.\n")
                    continue
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2.\n")
                continue
    
    def _select_checkpoint_file(self):
        """Prompt user to select a checkpoint file to resume from"""
        print(f"Select checkpoint file to resume from...")
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Checkpoint File to Resume From",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            print(f"✓ Selected checkpoint: {Path(file_path).name}\n")
            return file_path
        else:
            return None
    
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
    
    def train(self, dataset_yaml, resume=False, checkpoint_path=None, 
              epochs=800, imgsz=1280, batch=4, lr0=0.001, patience=120, overfit=False):
        """Train with configuration optimized for aerial small-object fire/smoke detection
        
        Args:
            dataset_yaml: Path to dataset YAML
            resume: Whether to resume training from checkpoint
            checkpoint_path: Path to checkpoint file for resuming (if resume=True)
            epochs: Number of training epochs (default: 800)
            imgsz: Image size (default: 1280 for small object detection)
            batch: Batch size (default: 4, lower due to high imgsz)
            lr0: Initial learning rate (default: 0.001)
            patience: Early stopping patience (default: 120)
            overfit: Enable overfit mode — disables augmentation & regularization (default: False)
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 YOLO11m AERIAL FIRE & SMOKE DETECTION TRAINING")
        print(f"   Optimized for small objects from drone @ 100m altitude")
        print(f"{'='*80}")
        
        # Validate dataset
        self.validate_dataset(dataset_yaml)
        
        # Load model
        if not Path(self.model_path).exists() and self.model_path != 'yolo11m.pt':
            print(f"❌ Model file not found: {self.model_path}")
            return
        
        print(f"\n🔄 Loading model: {self.model_path}")
        model = YOLO(self.model_path)
        
        # Determine training mode
        if resume and checkpoint_path:
            print(f"📌 Training mode: RESUME (continuing from checkpoint)")
            print(f"   Checkpoint: {Path(checkpoint_path).name}")
            # Use checkpoint for resuming
            model = YOLO(checkpoint_path)
        else:
            print(f"📌 Training mode: FRESH START (starting from epoch 1)")
        
        # GPU memory optimization
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory cleared")
        
        if overfit:
            print(f"\n🎯 OVERFIT MODE ENABLED")
            print(f"   All augmentation DISABLED — model will memorize training data")
            print(f"   Perfect for CosysAirSim demo with known scenes\n")

        # Aerial small-object optimized training configuration
        train_args = {
            'data': str(dataset_yaml),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'workers': 8,
            'project': str(PROJECT_ROOT / 'runs'),
            'name': f"fire_smoke_{'overfit_' if overfit else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Optimizer
            'optimizer': 'AdamW',
            'lr0': lr0 if not overfit else 0.01,
            'lrf': (lr0 * 0.1) if not overfit else 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005 if not overfit else 0.0,
            'warmup_epochs': 10 if not overfit else 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.0,
            
            # === AUGMENTATION ===
            'hsv_h': 0.02 if not overfit else 0.0,
            'hsv_s': 0.8 if not overfit else 0.0,
            'hsv_v': 0.5 if not overfit else 0.0,
            'degrees': 180.0 if not overfit else 0.0,
            'translate': 0.2 if not overfit else 0.0,
            'scale': 0.9 if not overfit else 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.5 if not overfit else 0.0,
            'fliplr': 0.5 if not overfit else 0.0,
            'mosaic': 1.0 if not overfit else 0.0,
            'mixup': 0.15 if not overfit else 0.0,
            'copy_paste': 0.4 if not overfit else 0.0,
            
            # Training schedule
            'cos_lr': True if not overfit else False,
            'close_mosaic': 20 if not overfit else 0,
            'amp': True,
            'fraction': 1.0,
            'patience': patience if not overfit else 0,  # No early stopping in overfit mode
            
            # Validation and saving
            'val': True,
            'save': True,
            'save_period': 50,   # Save checkpoint every 50 epochs
            'cache': 'disk',     # Cache to disk (imgsz=1280 images are large)
            'plots': True,
            'verbose': True,
            
            # Stability
            'rect': False,
            'resume': resume,
        }
        
        # Display configuration
        mode_label = "OVERFIT (Demo)" if overfit else "Small Object Optimized"
        print(f"\n⚙️  Training Configuration ({mode_label}):")
        print(f"   Image Size: {imgsz}px {'(⚡ HIGH RES for small objects)' if imgsz >= 1280 else ''}")
        print(f"   Batch Size: {batch}")
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print(f"   Workers: 8")
        print(f"   Optimizer: AdamW")
        if overfit:
            print(f"   Learning Rate: {train_args['lr0']} (constant, high for fast memorization)")
            print(f"   Weight Decay: 0 (no regularization)")
            print(f"   Augmentation: ALL DISABLED")
            print(f"   Early Stopping: DISABLED")
            print(f"   Warmup: 3 epochs")
        else:
            print(f"   Learning Rate: {lr0} → {lr0 * 0.1} (initial → final)")
            print(f"   Warmup: 10 epochs")
            print(f"   Early Stopping (Patience): {patience} epochs")
            print(f"   Rotation: 180° (full aerial rotation)")
            print(f"   Vertical Flip: Enabled (aerial top-down)")
            print(f"   Copy-Paste: 0.4 (duplicates small fire/smoke objects)")
            print(f"   Close Mosaic: last 20 epochs")
        print(f"   AMP (FP16): Enabled")
        print(f"   ⚠️  High imgsz = more VRAM. Reduce batch if OOM.")
        
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
    
    parser = argparse.ArgumentParser(description='YOLO11m Aerial Fire & Smoke Detection Training')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=800,
                       help='Number of training epochs (default: 800)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size (default: 1280 for small object detection)')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size (default: 4 at imgsz=1280)')
    parser.add_argument('--lr0', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=120,
                       help='Early stopping patience in epochs (default: 120)')
    parser.add_argument('--overfit', action='store_true',
                       help='Overfit mode: disable augmentation & regularization for demo/simulator')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint (skip training mode selection)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimpleYOLOTrainer()
    
    # Select model
    trainer.select_model()
    
    # Select training mode (unless --resume flag provided)
    checkpoint_path = None
    if args.resume:
        resume = True
        print(f"📌 Training mode: RESUME (from command line flag)")
    else:
        resume, checkpoint_path = trainer.select_training_mode()
    
    # Start training with custom parameters
    trainer.train(
        dataset_yaml=args.data, 
        resume=resume, 
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        overfit=args.overfit
    )


if __name__ == "__main__":
    main()
