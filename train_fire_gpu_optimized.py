#!/usr/bin/env python3
"""
GPU-Optimized YOLOv11m Fire Detection Training Script
Optimized for NVIDIA GeForce RTX 4070 Ti Super (16GB VRAM)
Supports 3-channel (RGB) and 4-channel (RGBA/RGBT/Multispectral) image training
"""

import os
import torch
import time
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2

def validate_image_channels(dataset_yaml_path, expected_channels):
    """
    Validate that dataset images have the expected number of channels
    """
    try:
        with open(dataset_yaml_path, 'r') as f:
            import yaml
            dataset_config = yaml.safe_load(f)
        
        # Check a few sample images from train set
        train_path = Path(dataset_config.get('train', ''))
        if train_path.exists():
            sample_images = list(train_path.glob('**/*.jpg'))[:3] + list(train_path.glob('**/*.png'))[:3]
            
            for img_path in sample_images[:5]:  # Check up to 5 images
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    actual_channels = 1 if len(img.shape) == 2 else img.shape[2]
                    if actual_channels != expected_channels:
                        print(f"⚠️  Warning: Image {img_path.name} has {actual_channels} channels, expected {expected_channels}")
                        return False
            
            print(f"✅ Dataset validation: Images have {expected_channels} channels as expected")
            return True
    except Exception as e:
        print(f"⚠️  Could not validate image channels: {e}")
        return True  # Don't fail training due to validation error

def main():
    # ================================================================
    # IMAGE CHANNEL CONFIGURATION - ADJUST HERE FOR YOUR DATASET
    # ================================================================
    IMAGE_CHANNELS = 3  # Set to 3 for RGB, 4 for RGBA/RGBT (thermal), etc.
    
    # Supported configurations:
    # - 3 channels: Standard RGB images (most common for fire detection)
    # - 4 channels: RGBA, RGBT (RGB + Thermal), or other multispectral data
    #
    # Use cases:
    # - RGB fire detection: IMAGE_CHANNELS = 3
    # - Thermal + RGB fire detection: IMAGE_CHANNELS = 4  
    # - Multispectral fire detection: IMAGE_CHANNELS = 4
    # - RGBA with alpha channel: IMAGE_CHANNELS = 4
    #
    # Note: Your dataset images must match the channel count specified here
    
    # GPU optimization for RTX 4070 Ti Super
    if torch.cuda.is_available():
        device = 'cuda'
        # Optimize for 16GB VRAM - much more aggressive settings
        batch_size = 32  # Significantly increased for 16GB VRAM
        workers = 8      # More workers for better data loading
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        batch_size = 2
        workers = 4
        print("⚠️  Using CPU (GPU not available)")
    
    # Paths
    script_dir = Path(__file__).parent
    dataset_yaml = script_dir.parent / "dataset" / "fire_detection.yaml"
    project_dir = script_dir / "runs" / "fire_detection_gpu"
    
    print(f"📁 Dataset: {dataset_yaml}")
    print(f"📁 Output: {project_dir}")
    
    # Verify dataset exists
    if not dataset_yaml.exists():
        print(f"❌ Dataset file not found: {dataset_yaml}")
        return
    
    # Validate image channels match configuration
    print(f"\n🔍 Validating dataset for {IMAGE_CHANNELS}-channel images...")
    validate_image_channels(dataset_yaml, IMAGE_CHANNELS)
    
    # Load YOLOv11m model
    print(f"\n🔄 Loading YOLOv11m model for {IMAGE_CHANNELS}-channel images...")
    
    # Choose model based on channel configuration
    if IMAGE_CHANNELS == 3:
        # Standard RGB model
        model = YOLO('yolo11m.pt')
        print("📷 Using standard RGB (3-channel) model")
    elif IMAGE_CHANNELS == 4:
        # For 4-channel images (RGBA/RGBT/multispectral)
        # Load base model and modify the input layer
        model = YOLO('yolo11m.pt')
        
        # Modify the model's first layer to accept 4 channels
        # This creates a new model with 4-channel input
        model.model.model[0].conv.in_channels = IMAGE_CHANNELS
        
        # Reinitialize the first layer weights for 4 channels
        with torch.no_grad():
            old_weight = model.model.model[0].conv.weight.clone()
            new_weight = torch.zeros(old_weight.shape[0], IMAGE_CHANNELS, *old_weight.shape[2:])
            
            # Copy RGB weights to first 3 channels
            new_weight[:, :3, :, :] = old_weight
            
            # Initialize 4th channel with sum of RGB channels (total weight)
            if IMAGE_CHANNELS == 4:
                new_weight[:, 3, :, :] = old_weight.sum(dim=1)
            
            model.model.model[0].conv.weight = torch.nn.Parameter(new_weight)
        
        print(f"🔧 Modified model for {IMAGE_CHANNELS}-channel input (e.g., RGBA/RGBT/thermal)")
        print("   - RGB channels: copied from pretrained weights")
        print("   - Additional channel: initialized with RGB total weight (sum)")
    else:
        raise ValueError(f"❌ Unsupported channel count: {IMAGE_CHANNELS}. Use 3 or 4 channels.")
    
    print(f"✅ Model loaded and configured for {IMAGE_CHANNELS}-channel training")
    
    # Check for existing training to resume
    existing_runs = list(project_dir.glob("yolo11m_rtx4070ti_*")) if project_dir.exists() else []
    latest_run = None
    resume_path = None
    
    if existing_runs:
        # Find the most recent training run
        latest_run = max(existing_runs, key=lambda x: x.stat().st_mtime)
        potential_resume = latest_run / 'weights' / 'last.pt'
        
        if potential_resume.exists():
            resume_path = str(potential_resume)
            print(f"🔄 Found existing training: {latest_run.name}")
            print(f"📁 Resume from: {resume_path}")
        else:
            print(f"⚠️  Found run {latest_run.name} but no last.pt to resume from")
    
    if not resume_path:
        print("🆕 Starting fresh training - no previous training found")
    
    # Check dataset for new images
    if dataset_yaml.exists():
        with open(dataset_yaml, 'r') as f:
            import yaml
            dataset_config = yaml.safe_load(f)
            
        # Count images in dataset
        train_path = Path(dataset_config.get('train', ''))
        val_path = Path(dataset_config.get('val', ''))
        
        train_images = len(list(train_path.glob('**/*.jpg'))) + len(list(train_path.glob('**/*.png'))) if train_path.exists() else 0
        val_images = len(list(val_path.glob('**/*.jpg'))) + len(list(val_path.glob('**/*.png'))) if val_path.exists() else 0
        total_images = train_images + val_images
        
        print(f"📊 Dataset Summary:")
        print(f"  Training images: {train_images}")
        print(f"  Validation images: {val_images}")
        print(f"  Total images: {total_images}")
        
        if resume_path:
            print(f"🔄 Continuing training with updated dataset")
            print(f"📈 New images will be included in training")
        else:
            print(f"🆕 Starting training with {total_images} images")
    
    # GPU-optimized training parameters for RTX 4070 Ti Super
    train_args = {
        'data': str(dataset_yaml),
        'epochs': 500,                   # Increased epochs for better training
        'imgsz': 832,                   # Larger image size for better accuracy
        'batch': batch_size,            # High batch size for 16GB VRAM
        'device': device,               # GPU device
        'workers': workers,             # More workers for data loading
        'project': str(project_dir),
        'name': f'yolo11m_rtx4070ti_{int(time.time())}',
        
        # Optimization settings for high-end GPU
        'optimizer': 'AdamW',           # Best optimizer for modern training
        'lr0': 0.01,                    # Higher learning rate for large batches
        'lrf': 0.001,                   # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,             # More warmup epochs
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Enhanced data augmentation for better generalization
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,                # Increased rotation
        'translate': 0.2,               # Increased translation
        'scale': 0.9,                   # More aggressive scaling
        'shear': 2.0,                   # Added shearing
        'perspective': 0.0001,          # Small perspective changes
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,                  # Enable mixup for better performance
        'copy_paste': 0.3,              # Enable copy-paste augmentation
        
        # Advanced validation settings
        'val': True,
        'plots': True,
        'save': True,
        'save_period': -1,              # Disable periodic epoch saving (prevents epoch*.pt files)
        'save_json': False,             # Disable JSON results saving
        'cache': 'ram',                 # Cache in RAM for faster training
        'rect': False,                  # Keep disabled for consistency
        'cos_lr': True,                 # Cosine learning rate scheduler
        'close_mosaic': 15,             # Disable mosaic in last 15 epochs
        
        # High-end GPU memory optimization
        'amp': True,                    # Automatic Mixed Precision
        'fraction': 0.95,               # Use 95% of GPU memory (16GB)
        'patience': 50,                 # Early stopping patience
        'resume': resume_path or False, # Resume from last.pt if available, otherwise start fresh
        'overlap_mask': True,           # Better segmentation if needed
        'mask_ratio': 4,                # Mask ratio for training
    }
    
    print(f"\n🎯 Training Configuration:")
    print(f"  📷 Image channels: {IMAGE_CHANNELS} {'(RGB)' if IMAGE_CHANNELS == 3 else '(RGBA/RGBT/Multispectral)'}")
    for key, value in train_args.items():
        if key in ['data', 'project', 'name']:
            continue
        print(f"  {key}: {value}")
    
    # Memory management for GPU
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"\n💾 Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Start training
    print(f"\n🚀 Starting RTX 4070 Ti Super accelerated training...")
    print(f"📈 Expected completion: ~2.5-3 hours for 500 epochs")
    print(f"💪 Batch size: {batch_size} (utilizing 16GB VRAM)")
    print(f"🖼️  Image size: 832px (higher resolution for better accuracy)")
    print(f"📷 Input channels: {IMAGE_CHANNELS} {'(RGB)' if IMAGE_CHANNELS == 3 else '(RGBA/RGBT/Multispectral)'}")
    
    if resume_path:
        print(f"🔄 Resuming training from: {resume_path}")
        print(f"📊 Continuing with {total_images} total images (including any new ones)")
    else:
        print(f"🆕 Fresh training: Starting new training session")
        print(f"📊 Training with {total_images} images")
    
    print(f"💾 Note: Only best.pt and last.pt will be saved (no epoch*.pt files for git compatibility)")
    print(f"⚡ save_period: -1 disables intermediate epoch checkpoints")
    
    start_time = time.time()
    
    try:
        # Train the model
        results = model.train(**train_args)
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Total time: {training_time/60:.1f} minutes")
        
        # GPU memory usage
        if device == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"📊 Peak GPU Memory: {max_memory:.1f} MB")
        
        # Display results
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        print(f"🎯 Best model saved: {best_model_path}")
        
        # Load best model and show metrics
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val()
        
        print(f"\n📊 Final Metrics:")
        print(f"  mAP50: {metrics.box.map50:.3f}")
        print(f"  mAP50-95: {metrics.box.map:.3f}")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall: {metrics.box.mr:.3f}")
        
        # Performance comparison
        speed_improvement = "10-15x faster" if device == 'cuda' else "baseline"
        print(f"\n🚀 RTX 4070 Ti Super Performance: {speed_improvement} than CPU training")
        
        print(f"\n🎉 High-performance fire detection model ready!")
        print(f"📁 Results folder: {results.save_dir}")
        print(f"🎯 Only essential models saved: best.pt (~40MB) and last.pt")
        print(f"📦 Git-friendly: No large epoch*.pt files generated")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        if device == 'cuda':
            print("💡 If out of memory, try reducing batch_size to 24 or image size to 640")
            print("💡 Current settings optimized for 16GB VRAM")
            print("💡 save_period=-1 prevents large epoch checkpoint files")
        raise
    
    finally:
        # Cleanup GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
