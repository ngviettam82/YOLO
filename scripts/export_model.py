#!/usr/bin/env python3
"""
YOLO Model Export Script
Export trained models to various formats for deployment
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def export_model(model_path, formats=None, img_size=832, half=False):
    """
    Export YOLO model to various formats
    
    Args:
        model_path: Path to trained model (.pt file)
        formats: List of export formats
                Available: ['onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'tflite']
        img_size: Image size for export
        half: Use FP16 half precision (faster, less accurate)
    """
    if formats is None:
        formats = ['onnx']  # Default to ONNX
    
    print(f"📦 Exporting Model: {model_path}")
    print("="*80)
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"\n📥 Loading model...")
    model = YOLO(str(model_path))
    
    print(f"   Model size: {model_path.stat().st_size / 1024**2:.2f} MB")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Export each format
    export_dir = model_path.parent
    
    for fmt in formats:
        print(f"\n🔄 Exporting to {fmt.upper()}...")
        
        try:
            if fmt == 'onnx':
                # ONNX format - most compatible
                exported = model.export(
                    format='onnx',
                    imgsz=img_size,
                    half=half,
                    dynamic=True,  # Dynamic batch size
                    simplify=True  # Simplify ONNX graph
                )
                print(f"✅ ONNX export successful: {exported}")
                print(f"   - Compatible with most inference frameworks")
                print(f"   - Dynamic batch size enabled")
                
            elif fmt == 'torchscript':
                # TorchScript format - PyTorch native
                exported = model.export(
                    format='torchscript',
                    imgsz=img_size,
                    half=half
                )
                print(f"✅ TorchScript export successful: {exported}")
                print(f"   - Native PyTorch format")
                print(f"   - Best for Python deployment")
                
            elif fmt == 'engine':
                # TensorRT format - NVIDIA GPUs only
                if not torch.cuda.is_available():
                    print(f"⚠️  TensorRT requires NVIDIA GPU. Skipping...")
                    continue
                
                exported = model.export(
                    format='engine',
                    imgsz=img_size,
                    half=half,
                    workspace=4  # 4GB workspace for TensorRT
                )
                print(f"✅ TensorRT export successful: {exported}")
                print(f"   - Optimized for NVIDIA GPUs")
                print(f"   - Maximum inference speed")
                
            elif fmt == 'openvino':
                # OpenVINO format - Intel hardware
                exported = model.export(
                    format='openvino',
                    imgsz=img_size,
                    half=half
                )
                print(f"✅ OpenVINO export successful: {exported}")
                print(f"   - Optimized for Intel hardware")
                print(f"   - Good CPU performance")
                
            elif fmt == 'coreml':
                # CoreML format - Apple devices
                exported = model.export(
                    format='coreml',
                    imgsz=img_size,
                    half=half
                )
                print(f"✅ CoreML export successful: {exported}")
                print(f"   - Optimized for Apple devices")
                print(f"   - iOS/macOS deployment")
                
            elif fmt == 'tflite':
                # TensorFlow Lite - Mobile/Edge devices
                exported = model.export(
                    format='tflite',
                    imgsz=img_size,
                    half=half
                )
                print(f"✅ TFLite export successful: {exported}")
                print(f"   - Optimized for mobile devices")
                print(f"   - Android deployment")
            
            else:
                print(f"⚠️  Unknown format: {fmt}")
                
        except Exception as e:
            print(f"❌ Export failed for {fmt}: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"✅ Export complete! Check: {export_dir}")
    print(f"{'='*80}")
    
    # Display export summary
    print(f"\n📋 Export Summary:")
    for ext in ['.onnx', '.torchscript', '.engine', '_openvino_model', '.mlmodel', '.tflite']:
        exported_files = list(export_dir.glob(f"*{ext}*"))
        if exported_files:
            for f in exported_files:
                size_mb = f.stat().st_size / 1024**2 if f.is_file() else 0
                print(f"   ✓ {f.name} ({size_mb:.2f} MB)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Export')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--formats', nargs='+', default=['onnx'],
                       choices=['onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'tflite'],
                       help='Export formats (default: onnx)')
    parser.add_argument('--imgsz', type=int, default=832,
                       help='Image size for export (default: 832)')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half precision')
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        formats=args.formats,
        img_size=args.imgsz,
        half=args.half
    )


if __name__ == "__main__":
    main()
