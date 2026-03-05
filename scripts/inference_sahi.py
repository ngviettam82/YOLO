#!/usr/bin/env python3
"""
SAHI-Powered YOLO Inference for Aerial Fire & Smoke Detection
=============================================================
Slicing Aided Hyper Inference (SAHI) dramatically improves detection of small 
objects by slicing the image into overlapping tiles, running detection on each,
then merging results with NMS.

This is CRITICAL for drone fire detection at 100m+ altitude where fire/smoke
may occupy only 15-50 pixels in a 1280x720 frame.

Usage:
    python scripts/inference_sahi.py --model runs/best.pt --source image.jpg
    python scripts/inference_sahi.py --model runs/best.pt --source video.mp4 --slice 640
    python scripts/inference_sahi.py --model runs/best.pt --source folder/ --no-sliced
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def check_sahi_installed():
    """Check if SAHI is installed, provide install instructions if not."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction, get_prediction
        return True
    except ImportError:
        print("=" * 60)
        print("SAHI is not installed. Install it with:")
        print("  pip install sahi")
        print("=" * 60)
        return False


def run_sahi_inference(
    model_path: str,
    source: str,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    conf: float = 0.2,
    device: str = '',
    save: bool = True,
    show: bool = False,
    no_sliced: bool = False,
):
    """
    Run SAHI sliced inference for small object detection.

    Args:
        model_path: Path to trained YOLO .pt model
        source: Image path, video path, or folder
        slice_size: Size of each slice in pixels (default 640)
        overlap_ratio: Overlap between slices (default 0.25 = 25%)
        conf: Confidence threshold (lower for small objects)
        device: Device ('cuda', 'cpu', or '' for auto)
        save: Whether to save annotated results
        show: Whether to display results
        no_sliced: If True, run standard (non-sliced) inference
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    if not device:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"{'='*70}")
    print(f"  SAHI Aerial Fire & Smoke Detection")
    print(f"{'='*70}")
    print(f"  Model:          {model_path.name}")
    print(f"  Source:          {source}")
    print(f"  Device:          {device}")
    print(f"  Slice Size:      {slice_size}x{slice_size}")
    print(f"  Overlap:         {overlap_ratio*100:.0f}%")
    print(f"  Confidence:      {conf}")
    print(f"  Mode:            {'Standard' if no_sliced else 'SAHI Sliced'}")
    print(f"{'='*70}\n")

    # Load model via SAHI wrapper
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # SAHI uses yolov8 type for ultralytics models
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
    )

    source_path = Path(source)

    # Create output directory
    output_dir = PROJECT_ROOT / 'runs' / 'sahi_results' / f"predict_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    if source_path.is_file():
        if source_path.suffix.lower() in image_extensions:
            _process_images(
                [source_path], detection_model, slice_size, overlap_ratio,
                conf, output_dir, save, show, no_sliced
            )
        elif source_path.suffix.lower() in video_extensions:
            _process_video(
                source_path, detection_model, slice_size, overlap_ratio,
                conf, output_dir, save, show, no_sliced
            )
        else:
            print(f"Unsupported file type: {source_path.suffix}")
    elif source_path.is_dir():
        images = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in image_extensions
        )
        if not images:
            print(f"No images found in {source_path}")
            return
        print(f"Found {len(images)} images in {source_path}\n")
        _process_images(
            images, detection_model, slice_size, overlap_ratio,
            conf, output_dir, save, show, no_sliced
        )
    else:
        print(f"Source not found: {source}")
        return

    print(f"\nResults saved to: {output_dir}")


def _process_images(image_paths, detection_model, slice_size, overlap_ratio,
                    conf, output_dir, save, show, no_sliced):
    """Process a list of images with SAHI."""
    from sahi.predict import get_sliced_prediction, get_prediction

    total_detections = 0
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing: {img_path.name}")
        t0 = time.time()

        if no_sliced:
            result = get_prediction(
                str(img_path),
                detection_model,
            )
        else:
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
            )

        detections = result.object_prediction_list
        total_detections += len(detections)
        elapsed = time.time() - t0

        # Print detections
        for det in detections:
            bbox = det.bbox
            cat = det.category.name
            score = det.score.value
            print(f"   {cat}: {score:.2f}  [{bbox.minx:.0f},{bbox.miny:.0f},{bbox.maxx:.0f},{bbox.maxy:.0f}]")

        if not detections:
            print("   No detections")

        print(f"   Time: {elapsed:.2f}s | Detections: {len(detections)}\n")

        if save:
            result.export_visuals(
                export_dir=str(output_dir),
                file_name=img_path.stem,
            )

        if show:
            img = cv2.imread(str(img_path))
            for det in detections:
                bbox = det.bbox
                cat = det.category.name
                score = det.score.value
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                color = (0, 0, 255) if cat == 'fire' else (128, 128, 128)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{cat} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("SAHI Fire Detection", img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    print(f"Total detections across all images: {total_detections}")


def _process_video(video_path, detection_model, slice_size, overlap_ratio,
                   conf, output_dir, save, show, no_sliced):
    """Process video frame-by-frame with SAHI."""
    from sahi.predict import get_sliced_prediction, get_prediction

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames\n")

    writer = None
    if save:
        out_path = output_dir / f"{video_path.stem}_sahi.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # Save temp frame for SAHI (it requires file path)
            temp_path = output_dir / '_temp_frame.jpg'
            cv2.imwrite(str(temp_path), frame)

            if no_sliced:
                result = get_prediction(str(temp_path), detection_model)
            else:
                result = get_sliced_prediction(
                    str(temp_path),
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                )

            # Draw detections on frame
            for det in result.object_prediction_list:
                bbox = det.bbox
                cat = det.category.name
                score = det.score.value
                x1, y1 = int(bbox.minx), int(bbox.miny)
                x2, y2 = int(bbox.maxx), int(bbox.maxy)
                color = (0, 0, 255) if cat == 'fire' else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cat} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            n_det = len(result.object_prediction_list)
            if frame_idx % 30 == 0 or n_det > 0:
                print(f"  Frame {frame_idx}/{total_frames}: {n_det} detections")

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("SAHI Fire Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        temp_path = output_dir / '_temp_frame.jpg'
        if temp_path.exists():
            temp_path.unlink()

    print(f"\nProcessed {frame_idx} frames")
    if save and writer:
        print(f"Output video: {output_dir / f'{video_path.stem}_sahi.mp4'}")


def main():
    parser = argparse.ArgumentParser(
        description='SAHI-Powered YOLO Inference for Aerial Fire/Smoke Detection'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO .pt model')
    parser.add_argument('--source', type=str, required=True,
                        help='Image, video, or folder path')
    parser.add_argument('--slice', type=int, default=640,
                        help='Slice size in pixels (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap ratio between slices (default: 0.25)')
    parser.add_argument('--conf', type=float, default=0.2,
                        help='Confidence threshold (default: 0.2, lower for small objects)')
    parser.add_argument('--device', type=str, default='',
                        help='Device (cuda:0, cpu, or empty for auto)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save annotated results')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--no-sliced', action='store_true',
                        help='Disable slicing (use standard inference)')

    args = parser.parse_args()

    if not check_sahi_installed():
        return

    run_sahi_inference(
        model_path=args.model,
        source=args.source,
        slice_size=args.slice,
        overlap_ratio=args.overlap,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
        no_sliced=args.no_sliced,
    )


if __name__ == '__main__':
    main()
