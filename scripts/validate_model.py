#!/usr/bin/env python3
"""
YOLO Model Validation Script
Validates trained model and generates detailed metrics
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def validate_model(model_path, data_yaml, img_size=832, batch_size=16):
    """
    Validate YOLO model and display comprehensive metrics
    
    Args:
        model_path: Path to trained model (.pt file)
        data_yaml: Path to dataset YAML file
        img_size: Image size for validation
        batch_size: Batch size for validation
    """
    print(f"🔍 Validating Model: {model_path}")
    print("="*80)
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Check if data YAML exists
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        print(f"❌ Dataset YAML not found: {data_yaml}")
        return
    
    # Load model
    print(f"\n📦 Loading model...")
    model = YOLO(str(model_path))
    
    # Get model info
    print(f"\n📊 Model Information:")
    print(f"   Model file: {model_path.name}")
    print(f"   Model size: {model_path.stat().st_size / 1024**2:.2f} MB")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"   Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   Device: CPU")
    
    # Validate
    print(f"\n🚀 Running validation...")
    print(f"   Dataset: {data_yaml}")
    print(f"   Image size: {img_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}\n")
    
    try:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=img_size,
            batch=batch_size,
            device=device,
            plots=True,
            verbose=True
        )
        
        # Display detailed metrics
        print(f"\n{'='*80}")
        print(f"✅ Validation Complete!")
        print(f"{'='*80}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   mAP@0.5: {metrics.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"   mAP@0.75: {metrics.box.map75:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        # Calculate F1 Score
        if metrics.box.mp > 0 and metrics.box.mr > 0:
            f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
            print(f"   F1-Score: {f1:.4f}")
        
        # Per-class metrics
        if hasattr(metrics.box, 'class_result'):
            print(f"\n📋 Per-Class Metrics:")
            for i, class_name in enumerate(metrics.names.values()):
                print(f"   Class '{class_name}':")
                print(f"      mAP@0.5: {metrics.box.maps[i]:.4f}")
                print(f"      Precision: {metrics.box.p[i]:.4f}")
                print(f"      Recall: {metrics.box.r[i]:.4f}")
        
        # Speed metrics
        print(f"\n⚡ Speed Metrics:")
        print(f"   Preprocess: {metrics.speed['preprocess']:.2f} ms")
        print(f"   Inference: {metrics.speed['inference']:.2f} ms")
        print(f"   Postprocess: {metrics.speed['postprocess']:.2f} ms")
        total_time = sum(metrics.speed.values())
        fps = 1000 / total_time if total_time > 0 else 0
        print(f"   Total: {total_time:.2f} ms ({fps:.1f} FPS)")
        
        print(f"\n🎉 Validation successful!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Validation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--imgsz', type=int, default=832,
                       help='Image size for validation (default: 832)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for validation (default: 16)')
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        batch_size=args.batch
    )


if __name__ == "__main__":
    main()
