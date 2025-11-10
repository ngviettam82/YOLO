"""
Auto-labeling using Pre-trained YOLO Model
Generates YOLO format labels using a pre-trained model
Then verify/refine in Label Studio
"""

import os
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOAutoLabeler:
    """Auto-label images using pre-trained YOLO model"""
    
    def __init__(self, model_name='yolo11m.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLO model for auto-labeling
        
        Args:
            model_name: YOLO model to use (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IOU threshold for NMS
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained YOLO model"""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading pre-trained model: {self.model_name}")
            
            # Download model if not exists
            self.model = YOLO(self.model_name)
            self.class_names = self.model.names
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Classes: {list(self.class_names.values())}")
            
        except ImportError:
            logger.error("ultralytics not installed. Install with:")
            logger.error("  pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_objects(self, image_path, conf=None, iou=None):
        """
        Detect objects in image using YOLO
        
        Returns:
            List of detections: [class_id, x_center, y_center, width, height, confidence]
        """
        if conf is None:
            conf = self.conf_threshold
        if iou is None:
            iou = self.iou_threshold
        
        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                # Get image dimensions
                img_h, img_w = result.orig_shape
                
                # Extract boxes
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
        
        return detections
    
    def save_labels(self, detections, label_path, include_confidence=False):
        """
        Save detections to YOLO format label file
        
        Args:
            detections: List of detection dictionaries
            label_path: Path to save .txt file
            include_confidence: Whether to include confidence score
        """
        with open(label_path, 'w') as f:
            for det in detections:
                if include_confidence:
                    line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f} {det['confidence']:.2f}\n"
                else:
                    line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n"
                f.write(line)
    
    def process_directory(self, image_dir, output_dir, conf=None, iou=None, include_confidence=False):
        """
        Process all images in directory and generate labels
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save labels
            conf: Confidence threshold (uses self.conf_threshold if None)
            iou: IOU threshold (uses self.iou_threshold if None)
            include_confidence: Include confidence scores in label files
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(image_path.glob(f'*{ext}'))
            images.extend(image_path.glob(f'*{ext.upper()}'))
        
        images = sorted(list(set(images)))  # Remove duplicates and sort
        
        if not images:
            logger.warning(f"No images found in {image_dir}")
            return 0, 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Auto-labeling {len(images)} images using {self.model_name}")
        logger.info(f"Confidence threshold: {conf or self.conf_threshold}")
        logger.info(f"IOU threshold: {iou or self.iou_threshold}")
        logger.info(f"{'='*70}\n")
        
        labeled_count = 0
        total_objects = 0
        
        # Process each image
        for img_path in tqdm(images, desc="Auto-labeling images"):
            try:
                # Detect objects
                detections = self.detect_objects(img_path, conf, iou)
                
                # Save labels
                label_path = output_path / f"{img_path.stem}.txt"
                
                if detections:
                    self.save_labels(detections, label_path, include_confidence)
                    labeled_count += 1
                    total_objects += len(detections)
                else:
                    # Create empty label file (no objects detected)
                    label_path.touch()
                
            except Exception as e:
                logger.warning(f"Error processing {img_path.name}: {e}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Auto-labeling Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Images processed: {len(images)}")
        logger.info(f"Images with detections: {labeled_count}")
        logger.info(f"Total objects detected: {total_objects}")
        logger.info(f"Average objects per image: {total_objects/len(images):.1f}")
        logger.info(f"Labels saved to: {output_path}")
        logger.info(f"\n✨ Next steps:")
        logger.info(f"1. Review labels in Label Studio")
        logger.info(f"2. Correct any false positives/negatives")
        logger.info(f"3. Export corrected labels for training")
        logger.info(f"{'='*70}\n")
        
        return labeled_count, total_objects


class LabelViewer:
    """View auto-generated labels on images"""
    
    @staticmethod
    def visualize_labels(image_path, label_path, class_names, save_path=None):
        """
        Visualize labels on image
        
        Args:
            image_path: Path to image
            label_path: Path to label file
            class_names: Dictionary of class names
            save_path: Optional path to save visualization
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return
        
        h, w = img.shape[:2]
        
        # Read labels
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bboxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h
                    
                    # Calculate corners
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    # Draw box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Put label
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    cv2.putText(img, class_name, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(str(save_path), img)
        
        return img
    
    @staticmethod
    def create_visualization_directory(image_dir, label_dir, output_dir, class_names, limit=50):
        """
        Create visualization directory with sample labeled images
        
        Args:
            image_dir: Directory with images
            label_dir: Directory with labels
            output_dir: Directory to save visualizations
            class_names: Dictionary of class names
            limit: Maximum number of images to visualize
        """
        image_path = Path(image_dir)
        label_path = Path(label_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(image_path.glob(f'*{ext}'))
            images.extend(image_path.glob(f'*{ext.upper()}'))
        
        images = sorted(list(set(images)))[:limit]
        
        logger.info(f"\nCreating visualizations for {len(images)} images...")
        
        for img_file in tqdm(images, desc="Creating visualizations"):
            label_file = label_path / f"{img_file.stem}.txt"
            save_file = output_path / f"labeled_{img_file.name}"
            
            LabelViewer.visualize_labels(img_file, label_file, class_names, save_file)
        
        logger.info(f"✅ Visualizations saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Auto-label images using pre-trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-label images
  python auto_label.py --images dataset/images/train --output dataset/labels/train
  
  # With specific model and confidence threshold
  python auto_label.py --images dataset/images/train --output dataset/labels/train --model yolo11l.pt --conf 0.6
  
  # Include confidence scores in labels
  python auto_label.py --images dataset/images/train --output dataset/labels/train --include-confidence
  
  # Create visualizations of labels
  python auto_label.py --images dataset/images/train --labels dataset/labels/train --visualize --viz-limit 50
        """
    )
    
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images to label')
    parser.add_argument('--output', type=str,
                       help='Directory to save labels (default: same as images but labels)')
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                       choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                       help='YOLO model to use (default: yolo11m.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1, default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IOU threshold for NMS (0-1, default: 0.5)')
    parser.add_argument('--include-confidence', action='store_true',
                       help='Include confidence scores in label files')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of labeled images')
    parser.add_argument('--viz-limit', type=int, default=50,
                       help='Maximum number of images to visualize (default: 50)')
    parser.add_argument('--labels', type=str,
                       help='Directory with labels (for visualization)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        image_path = Path(args.images)
        args.output = str(image_path.parent / f"{image_path.name}_labels")
    
    # Create auto-labeler
    labeler = YOLOAutoLabeler(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Process images
    labeled_count, total_objects = labeler.process_directory(
        image_dir=args.images,
        output_dir=args.output,
        include_confidence=args.include_confidence
    )
    
    # Create visualizations if requested
    if args.visualize:
        if args.labels is None:
            args.labels = args.output
        
        viz_dir = Path(args.output).parent / "visualizations"
        LabelViewer.create_visualization_directory(
            image_dir=args.images,
            label_dir=args.labels,
            output_dir=str(viz_dir),
            class_names=labeler.class_names,
            limit=args.viz_limit
        )
    
    logger.info(f"\n✅ Auto-labeling complete! Results:")
    logger.info(f"   Images labeled: {labeled_count}")
    logger.info(f"   Total objects: {total_objects}")
    logger.info(f"   Output: {args.output}")
    
    if args.visualize:
        logger.info(f"   Visualizations: {Path(args.output).parent / 'visualizations'}")
    
    return 0


if __name__ == '__main__':
    exit(main())
