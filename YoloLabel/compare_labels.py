"""
Compare auto-generated labels with manually corrected labels
"""

import os
import sys
import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelComparator:
    """Compare original and corrected labels"""
    
    @staticmethod
    def read_labels(label_file):
        """Read YOLO format label file"""
        detections = []
        if Path(label_file).exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        detection = {
                            'class_id': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        }
                        detections.append(detection)
        return detections
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes"""
        x1_min = box1['x_center'] - box1['width'] / 2
        y1_min = box1['y_center'] - box1['height'] / 2
        x1_max = box1['x_center'] + box1['width'] / 2
        y1_max = box1['y_center'] + box1['height'] / 2
        
        x2_min = box2['x_center'] - box2['width'] / 2
        y2_min = box2['y_center'] - box2['height'] / 2
        x2_max = box2['x_center'] + box2['width'] / 2
        y2_max = box2['y_center'] + box2['height'] / 2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    @staticmethod
    def compare_labels(original_file, corrected_file, iou_threshold=0.5):
        """
        Compare original and corrected labels
        
        Returns:
            {
                'original_count': int,
                'corrected_count': int,
                'matched': int,
                'added': int,
                'removed': int,
                'modified': int
            }
        """
        original = LabelComparator.read_labels(original_file)
        corrected = LabelComparator.read_labels(corrected_file)
        
        matched = 0
        removed = 0
        added = 0
        modified = 0
        
        used_indices = set()
        
        # Match original boxes with corrected
        for orig_box in original:
            best_iou = 0
            best_idx = -1
            
            for idx, corr_box in enumerate(corrected):
                if idx in used_indices:
                    continue
                
                if orig_box['class_id'] != corr_box['class_id']:
                    continue
                
                iou = LabelComparator.calculate_iou(orig_box, corr_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                matched += 1
                if best_idx >= 0:
                    used_indices.add(best_idx)
                
                # Check if modified
                if best_iou < 1.0:
                    modified += 1
            else:
                removed += 1
        
        added = len(corrected) - matched
        
        return {
            'original_count': len(original),
            'corrected_count': len(corrected),
            'matched': matched,
            'added': added,
            'removed': removed,
            'modified': modified
        }
    
    @staticmethod
    def compare_directories(original_dir, corrected_dir, iou_threshold=0.5):
        """Compare all labels in two directories"""
        original_path = Path(original_dir)
        corrected_path = Path(corrected_dir)
        
        label_files = sorted(original_path.glob('*.txt'))
        
        total_stats = {
            'files_compared': 0,
            'total_original': 0,
            'total_corrected': 0,
            'total_matched': 0,
            'total_added': 0,
            'total_removed': 0,
            'total_modified': 0,
            'unchanged_files': 0,
            'modified_files': 0
        }
        
        details = []
        
        logger.info(f"Comparing {len(label_files)} label files...")
        logger.info(f"Original: {original_dir}")
        logger.info(f"Corrected: {corrected_dir}")
        logger.info(f"IoU threshold: {iou_threshold}")
        logger.info("")
        
        for label_file in label_files:
            corrected_file = corrected_path / label_file.name
            
            if not corrected_file.exists():
                logger.warning(f"Corrected file not found: {label_file.name}")
                continue
            
            stats = LabelComparator.compare_labels(
                str(label_file),
                str(corrected_file),
                iou_threshold
            )
            
            total_stats['files_compared'] += 1
            total_stats['total_original'] += stats['original_count']
            total_stats['total_corrected'] += stats['corrected_count']
            total_stats['total_matched'] += stats['matched']
            total_stats['total_added'] += stats['added']
            total_stats['total_removed'] += stats['removed']
            total_stats['total_modified'] += stats['modified']
            
            if stats['added'] + stats['removed'] + stats['modified'] == 0:
                total_stats['unchanged_files'] += 1
            else:
                total_stats['modified_files'] += 1
            
            if stats['added'] + stats['removed'] > 0:
                details.append({
                    'file': label_file.name,
                    'stats': stats
                })
        
        return total_stats, details


def print_report(total_stats, details, show_details=False):
    """Print comparison report"""
    logger.info("=" * 70)
    logger.info("Label Comparison Report")
    logger.info("=" * 70)
    logger.info("")
    
    logger.info("📊 Overall Statistics:")
    logger.info(f"  Files compared: {total_stats['files_compared']}")
    logger.info(f"  Unchanged: {total_stats['unchanged_files']}")
    logger.info(f"  Modified: {total_stats['modified_files']}")
    logger.info("")
    
    logger.info("📈 Object Count:")
    logger.info(f"  Original detections: {total_stats['total_original']}")
    logger.info(f"  Corrected detections: {total_stats['total_corrected']}")
    logger.info("")
    
    logger.info("✏️  Changes:")
    logger.info(f"  Matched: {total_stats['total_matched']}")
    logger.info(f"  Added: {total_stats['total_added']}")
    logger.info(f"  Removed: {total_stats['total_removed']}")
    logger.info(f"  Modified: {total_stats['total_modified']}")
    logger.info("")
    
    # Calculate accuracy
    if total_stats['total_original'] > 0:
        accuracy = (total_stats['total_matched'] / total_stats['total_original']) * 100
        logger.info(f"✅ Auto-label Accuracy: {accuracy:.1f}%")
    
    logger.info("")
    
    if show_details and details:
        logger.info("📝 Files with changes:")
        logger.info("-" * 70)
        
        for detail in sorted(details, key=lambda x: x['stats']['added'] + x['stats']['removed'], reverse=True)[:20]:
            file = detail['file']
            stats = detail['stats']
            
            change_str = f"→ {stats['corrected_count']}"
            if stats['added'] > 0:
                change_str += f" (+{stats['added']})"
            if stats['removed'] > 0:
                change_str += f" (-{stats['removed']})"
            
            logger.info(f"  {file:50} {stats['original_count']} {change_str}")
        
        logger.info("")
    
    logger.info("=" * 70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compare auto-generated labels with corrected labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_labels.py --original dataset/labels/train --corrected dataset/labels/train_corrected
  python compare_labels.py --original dataset/labels/train --corrected dataset/labels/train_corrected --details
        """
    )
    
    parser.add_argument('--original', type=str, required=True,
                       help='Directory with original auto-generated labels')
    parser.add_argument('--corrected', type=str, required=True,
                       help='Directory with corrected labels')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed list of modified files')
    
    args = parser.parse_args()
    
    # Compare directories
    total_stats, details = LabelComparator.compare_directories(
        args.original,
        args.corrected,
        args.iou
    )
    
    # Print report
    print_report(total_stats, details, args.details)
    
    return 0


if __name__ == '__main__':
    exit(main())
