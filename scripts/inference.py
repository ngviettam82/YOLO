#!/usr/bin/env python3
"""
YOLO Model Inference Script
Run inference on images, videos, or camera streams
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def run_inference(model_path, source, conf=0.25, iou=0.45, img_size=832, 
                 save=True, show=False, device=''):
    """
    Run YOLO inference on various sources
    
    Args:
        model_path: Path to trained model (.pt file)
        source: Input source (image path, video path, folder, or camera index)
        conf: Confidence threshold (0-1)
        iou: IOU threshold for NMS
        img_size: Image size for inference
        save: Save results
        show: Display results
        device: Device to use ('', 'cpu', '0', '0,1', etc.)
    """
    print(f"🚀 Running YOLO Inference")
    print("="*80)
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"\n📦 Loading model: {model_path.name}")
    model = YOLO(str(model_path))
    
    # Setup device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device.startswith('cuda') or device.isdigit():
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ⚠️  CUDA not available, using CPU")
            device = 'cpu'
    else:
        print(f"   Device: CPU")
    
    # Check source
    source_path = Path(source) if not source.isdigit() else source
    print(f"\n📥 Input source: {source}")
    
    # Determine source type
    if isinstance(source_path, Path):
        if source_path.is_file():
            if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                source_type = "image"
            elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                source_type = "video"
            else:
                source_type = "unknown"
        elif source_path.is_dir():
            source_type = "folder"
        else:
            source_type = "unknown"
    else:
        source_type = "camera"
    
    print(f"   Type: {source_type}")
    print(f"   Confidence threshold: {conf}")
    print(f"   IOU threshold: {iou}")
    print(f"   Image size: {img_size}")
    
    # Run inference
    print(f"\n🔄 Processing...")
    start_time = time.time()
    
    try:
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            device=device,
            save=save,
            show=show,
            stream=True if source_type in ['video', 'camera'] else False
        )
        
        # Process results
        frame_count = 0
        total_detections = 0
        
        for result in results:
            frame_count += 1
            detections = len(result.boxes)
            total_detections += detections
            
            if frame_count % 30 == 0 or source_type == "image":
                print(f"   Frame {frame_count}: {detections} detections")
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"✅ Inference complete!")
        print(f"{'='*80}")
        print(f"\n📊 Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Processing time: {elapsed_time:.2f}s")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Average detections per frame: {total_detections/frame_count:.1f}")
        
        if save:
            # Find output directory
            output_dir = Path("runs/detect")
            if output_dir.exists():
                latest_run = max(output_dir.glob("predict*"), 
                               key=lambda x: x.stat().st_mtime, 
                               default=None)
                if latest_run:
                    print(f"\n💾 Results saved to: {latest_run}")
        
        print(f"\n🎉 Inference successful!")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {str(e)}")
        raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Input source (image/video/folder path or camera index)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS (default: 0.45)')
    parser.add_argument('--imgsz', type=int, default=832,
                       help='Image size for inference (default: 832)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results (default: True)')
    parser.add_argument('--show', action='store_true',
                       help='Display results in real-time')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
        save=args.save,
        show=args.show,
        device=args.device
    )


if __name__ == "__main__":
    main()
