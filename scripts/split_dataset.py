"""
Dataset splitting utility for YOLO training
Splits raw images into train/val/test folders with optimized ratios
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


class DatasetSplitter:
    """Splits raw dataset into train/val/test sets with specified ratios"""
    
    def __init__(self, 
                 raw_dataset_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int = 42):
        """
        Initialize the dataset splitter
        
        Args:
            raw_dataset_dir: Path to raw dataset folder with images
            output_dir: Path to output directory (dataset folder)
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.2)
            test_ratio: Proportion for test set (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.raw_dataset_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Validate paths
        if not self.raw_dataset_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {self.raw_dataset_dir}")
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Set seed for reproducibility
        random.seed(seed)
        
        logger.info(f"Dataset splitter initialized")
        logger.info(f"Raw dataset: {self.raw_dataset_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Split ratios - Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%")
    
    def get_images(self) -> List[Path]:
        """Get all image files from raw dataset directory"""
        images = []
        for ext in SUPPORTED_FORMATS:
            images.extend(self.raw_dataset_dir.glob(f"*{ext}"))
            images.extend(self.raw_dataset_dir.glob(f"*{ext.upper()}"))
        
        # Remove duplicates
        images = list(set(images))
        
        if not images:
            logger.warning(f"No images found in {self.raw_dataset_dir}")
        else:
            logger.info(f"Found {len(images)} images in raw dataset")
        
        return sorted(images)
    
    def split_images(self, images: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split images into train/val/test sets"""
        random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * self.train_ratio)
        n_val = int(n_images * self.val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Train: {len(train_images)} images ({len(train_images)/n_images*100:.1f}%)")
        logger.info(f"  Val:   {len(val_images)} images ({len(val_images)/n_images*100:.1f}%)")
        logger.info(f"  Test:  {len(test_images)} images ({len(test_images)/n_images*100:.1f}%)")
        
        return train_images, val_images, test_images
    
    def copy_images(self, images: List[Path], destination: Path) -> int:
        """Copy images to destination folder"""
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        
        for image in images:
            try:
                dest_path = destination / image.name
                shutil.copy2(image, dest_path)
                copied += 1
            except Exception as e:
                logger.error(f"Error copying {image.name}: {e}")
        
        return copied
    
    def run(self, move_instead_of_copy: bool = False) -> bool:
        """
        Run the dataset splitting process
        
        Args:
            move_instead_of_copy: If True, move files instead of copying
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get images
            images = self.get_images()
            if not images:
                logger.error("No images to split!")
                return False
            
            # Split images
            train_images, val_images, test_images = self.split_images(images)
            
            # Copy/move images
            operation = "Moving" if move_instead_of_copy else "Copying"
            logger.info(f"\n{operation} images to destination folders...")
            
            dest_train = self.output_dir / "images" / "train"
            dest_val = self.output_dir / "images" / "val"
            dest_test = self.output_dir / "images" / "test"
            
            n_train = self.copy_images(train_images, dest_train)
            n_val = self.copy_images(val_images, dest_val)
            n_test = self.copy_images(test_images, dest_test)
            
            logger.info(f"\nSuccessfully {operation.lower()}ed:")
            logger.info(f"  Train: {n_train} images to {dest_train}")
            logger.info(f"  Val:   {n_val} images to {dest_val}")
            logger.info(f"  Test:  {n_test} images to {dest_test}")
            
            # Create placeholder label files
            logger.info("\nCreating placeholder label directories...")
            for label_dir in [
                self.output_dir / "labels" / "train",
                self.output_dir / "labels" / "val",
                self.output_dir / "labels" / "test"
            ]:
                label_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Dataset split completed successfully!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error during dataset split: {e}")
            return False


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Split raw dataset into train/val/test folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_dataset.py --raw-dir raw_dataset --output-dir dataset
  python split_dataset.py --raw-dir raw_dataset --output-dir dataset --train 0.8 --val 0.15 --test 0.05
  python split_dataset.py --raw-dir raw_dataset --output-dir dataset --move
        """
    )
    
    parser.add_argument('--raw-dir', type=str, default='raw_dataset',
                       help='Path to raw dataset directory (default: raw_dataset)')
    parser.add_argument('--output-dir', type=str, default='dataset',
                       help='Path to output dataset directory (default: dataset)')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying (default: copy)')
    
    args = parser.parse_args()
    
    # Create splitter and run
    splitter = DatasetSplitter(
        raw_dataset_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
    
    success = splitter.run(move_instead_of_copy=args.move)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
