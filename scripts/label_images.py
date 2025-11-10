"""
Image labeling utility for YOLO dataset
Supports multiple labeling tools and formats
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
import webbrowser
import time
import json
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelingTool:
    """Base class for labeling tools"""
    
    def __init__(self, train_dir: str):
        self.train_dir = Path(train_dir)
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        
        self.image_count = len(list(self.train_dir.glob('*.jpg'))) + \
                          len(list(self.train_dir.glob('*.png'))) + \
                          len(list(self.train_dir.glob('*.jpeg')))
    
    def check_installation(self) -> bool:
        """Check if the tool is installed"""
        raise NotImplementedError
    
    def run(self) -> bool:
        """Run the labeling tool"""
        raise NotImplementedError


class CVATLabeling(LabelingTool):
    """CVAT (Computer Vision Annotation Tool) - web-based annotation"""
    
    def check_installation(self) -> bool:
        """Check if CVAT is available"""
        # CVAT requires Docker, just provide instructions
        return True
    
    def run(self) -> bool:
        """Provide CVAT setup instructions"""
        logger.info("\n" + "="*60)
        logger.info("CVAT (Web-based Annotation Tool)")
        logger.info("="*60)
        logger.info(f"\nFound {self.image_count} images to label in {self.train_dir}")
        logger.info("\nCVAT Installation (requires Docker):")
        logger.info("1. Install Docker: https://www.docker.com/products/docker-desktop")
        logger.info("2. Run CVAT with Docker:")
        logger.info("   docker run -d -p 8080:8080 --name cvat -v cvat_db:/var/lib/postgresql cvat/cvat:latest")
        logger.info("3. Open http://localhost:8080 in your browser")
        logger.info("4. Create a project and upload images from:")
        logger.info(f"   {self.train_dir}")
        logger.info("\n✅ Follow the CVAT UI to create annotations")
        logger.info("📝 Export labels in YOLO format")
        logger.info("="*60 + "\n")
        return True


class BatchUploadHelper:
    """Helper class for managing batch uploads to Label Studio"""
    
    @staticmethod
    def create_batch_subdirs(train_dir: str, batch_size: int = 100) -> dict:
        """Create subdirectories for batch uploads
        
        Args:
            train_dir: Path to training images directory
            batch_size: Number of images per batch
            
        Returns:
            Dictionary mapping batch number to list of image files
        """
        train_path = Path(train_dir)
        images = sorted(
            [f for f in train_path.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        )
        
        batches = {}
        for i, image in enumerate(images):
            batch_num = i // batch_size
            if batch_num not in batches:
                batches[batch_num] = []
            batches[batch_num].append(image)
        
        return batches
    
    @staticmethod
    def print_upload_guide(image_count: int, batch_size: int = 100):
        """Print detailed batch upload instructions"""
        num_batches = (image_count + batch_size - 1) // batch_size
        
        logger.info("\n" + "="*70)
        logger.info("⚠️  LARGE DATASET DETECTED - BATCH UPLOAD REQUIRED")
        logger.info("="*70)
        logger.info(f"\nYou have {image_count} images to label.")
        logger.info(f"Uploading all at once causes Django streaming errors.")
        logger.info(f"\n✅ Solution: Upload in batches of {batch_size} images")
        logger.info(f"   This requires approximately {num_batches} upload sessions")
        logger.info("\n📋 STEP-BY-STEP WORKFLOW:")
        logger.info("="*70)
        
        batch_num = 1
        for i in range(0, image_count, batch_size):
            end = min(i + batch_size, image_count)
            logger.info(f"\n   Batch {batch_num}: Images {i+1} to {end}")
            batch_num += 1
        
        logger.info("\n" + "="*70)
        logger.info("📝 UPLOAD INSTRUCTIONS:")
        logger.info("="*70)
        logger.info("\n1. Label Studio opens at http://localhost:8080")
        logger.info("2. Create a NEW project or use existing")
        logger.info("3. Go to Data tab → Upload Files")
        logger.info("4. Use DRAG & DROP (more reliable than file picker):")
        logger.info(f"   - Drag batch 1 images ({batch_size} files)")
        logger.info("   - Wait for upload to complete (shows progress bar)")
        logger.info("   - Press Enter in terminal to continue")
        logger.info(f"   - Repeat for next batch")
        logger.info("\n5. After ALL images uploaded:")
        logger.info("   - Annotate as normal")
        logger.info("   - Export in YOLO format")
        logger.info("   - Place .txt files in dataset/labels/train/")
        logger.info("\n" + "="*70)
        logger.info("\n💡 If you still get errors:")
        logger.info("   - Use SMALLER batches (try 50 images at a time)")
        logger.info("   - Or switch to Roboflow (cloud-based, no upload limit)")
        logger.info("="*70 + "\n")


class AnnotationLabeling(LabelingTool):
    """Label Studio - web-based annotation tool (DEFAULT)"""
    
    def check_installation(self) -> bool:
        """Check if Label Studio is available"""
        try:
            __import__('label_studio')
            return True
        except ImportError:
            return False
    
    def run(self) -> bool:
        """Launch Label Studio for annotation"""
        logger.info("\n" + "="*60)
        logger.info("Label Studio - Web-based Annotation Tool (Default)")
        logger.info("="*60)
        logger.info(f"\nFound {self.image_count} images to label in {self.train_dir}")
        
        # For large datasets, show batch upload guide
        if self.image_count > 500:
            BatchUploadHelper.print_upload_guide(self.image_count, batch_size=100)
        
        try:
            # Import subprocess (doesn't require label_studio to be pre-installed)
            import subprocess as sp
            
            logger.info("\n✅ Label Studio Installation")
            logger.info("1. Installing/checking Label Studio...")
            
            # Ensure label-studio is installed
            result = sp.run(
                [sys.executable, '-m', 'pip', 'install', 'label-studio', '-q'],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("   ✅ Label Studio ready")
            else:
                logger.warning("   ⚠️  Label Studio install issue, trying to launch anyway...")
            
            # Set environment variables for large file uploads
            os.environ['DJANGO_DATA_UPLOAD_MAX_NUMBER_FILES'] = '10000'
            os.environ['LABEL_STUDIO_DATA_UPLOAD_MAX_NUMBER_FILES'] = '10000'
            os.environ['DJANGO_FILE_UPLOAD_MAX_MEMORY_SIZE'] = '5242880'
            os.environ['DATA_UPLOAD_MAX_MEMORY_SIZE'] = '5242880'
            os.environ['DJANGO_MIDDLEWARE_APPEND'] = 'django.middleware.security.SecurityMiddleware'
            os.environ['CLIENT_MAX_BODY_SIZE'] = '100m'
            os.environ['NGINX_CLIENT_MAX_BODY_SIZE'] = '100m'
            
            # For large datasets (>500 images), add extra settings
            if self.image_count > 500:
                os.environ['LABEL_STUDIO_DATA_UPLOAD_MAX_MEMORY_SIZE'] = '52428800'  # 50MB
                os.environ['UWSGI_HTTP_TIMEOUT'] = '600'
                os.environ['LABEL_STUDIO_WEB_LOCKED_UI'] = 'false'
            
            logger.info("\n🚀 Starting Label Studio server...")
            logger.info("   This will open http://localhost:8080 in your browser")
            logger.info("\n📋 Steps:")
            logger.info("   1. Sign up or login to Label Studio")
            logger.info("   2. Create a new project")
            logger.info("   3. Upload images from:")
            logger.info(f"      {self.train_dir}")
            logger.info("      (Use DRAG & DROP for best results)")
            logger.info("   4. Create bounding box annotations")
            logger.info("   5. Export in YOLO format")
            logger.info("   6. Place .txt files in dataset/labels/train/")
            logger.info("\n✅ When done:")
            logger.info("   1. Close the browser")
            logger.info("   2. Press Ctrl+C in this terminal")
            logger.info("="*60 + "\n")
            
            # Launch label-studio with suppressed output
            logger.info("Opening Label Studio at http://localhost:8080...\n")
            logger.info("Label Studio is starting in the background. Check your browser.\n")
            result = sp.run(
                ['label-studio'],
                capture_output=True,  # Suppress Label Studio's verbose logging
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Label Studio completed successfully")
                return True
            else:
                logger.warning("Label Studio exited with warning code")
                return True  # Consider it successful even if exit code non-zero
                
        except Exception as e:
            logger.error(f"Error launching Label Studio: {e}")
            logger.info("\n" + "="*60)
            logger.info("Manual Label Studio Setup")
            logger.info("="*60)
            logger.info("\nRun these commands manually:")
            logger.info("1. Install Label Studio:")
            logger.info("   pip install label-studio")
            logger.info("\n2. Start Label Studio:")
            logger.info("   label-studio")
            logger.info("\n3. Open browser to http://localhost:8080")
            logger.info("4. Create project and upload images from:")
            logger.info(f"   {self.train_dir}")
            logger.info("="*60 + "\n")
            return False


class RoboflowAI(LabelingTool):
    """Roboflow - AI-assisted annotation"""
    
    def check_installation(self) -> bool:
        """Check if Roboflow is available"""
        return True
    
    def run(self) -> bool:
        """Provide Roboflow instructions"""
        logger.info("\n" + "="*60)
        logger.info("Roboflow - AI-Assisted Annotation")
        logger.info("="*60)
        logger.info(f"\nFound {self.image_count} images to label in {self.train_dir}")
        logger.info("\nUsage (Free tier available):")
        logger.info("1. Visit: https://roboflow.com")
        logger.info("2. Sign up for free account")
        logger.info("3. Create new dataset")
        logger.info("4. Upload images from:")
        logger.info(f"   {self.train_dir}")
        logger.info("5. Use AI-assisted annotation (Auto Orient)")
        logger.info("6. Manual review and adjustment")
        logger.info("\n✅ AI helps speed up annotation process")
        logger.info("📝 Download annotations in YOLO format")
        logger.info("="*60 + "\n")
        return True


def display_menu():
    """Display labeling tool menu"""
    print("\n" + "="*60)
    print("YOLO Image Labeling Tools")
    print("="*60)
    print("\nAvailable Labeling Tools:")
    print("\n1. Label Studio - Web-based (Recommended - DEFAULT)")
    print("   • Simple web interface")
    print("   • Good for small to medium projects")
    print("   • Easy to install")
    
    print("\n2. CVAT - Web-based (Best for team collaboration)")
    print("   • Professional annotation platform")
    print("   • Requires Docker")
    print("   • Best for large teams")
    
    print("\n3. Roboflow - Cloud AI-Assisted (Easiest)")
    print("   • AI auto-annotation")
    print("   • No installation needed")
    print("   • Free tier available")
    
    print("\n4. Create Config File Only (Manual setup)")
    print("   • Set up your own tool")
    print("   • Create dataset configuration")
    
    print("\n" + "="*60)


def create_dataset_config(dataset_dir: str, num_classes: int = None) -> bool:
    """Create YOLO dataset.yaml configuration file"""
    
    dataset_path = Path(dataset_dir)
    config_path = dataset_path / "data.yaml"
    
    logger.info(f"\nCreating dataset configuration file...")
    
    # Count classes (folders in labels/train)
    labels_train = dataset_path / "labels" / "train"
    
    if num_classes is None:
        num_classes = 1  # Default to 1 class
        logger.warning(f"Using default {num_classes} class. Update data.yaml with your classes.")
    
    # Create YAML content
    yaml_content = f"""# YOLO dataset configuration
path: {dataset_path.absolute()}  # dataset root
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {num_classes}

# Class names
names:
"""
    
    # Add class names
    for i in range(num_classes):
        yaml_content += f"  {i}: class_{i}\n"
    
    yaml_content += """
# Update class names above to match your object categories
# Example:
#   0: person
#   1: car
#   2: dog
"""
    
    try:
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"✅ Configuration file created: {config_path}")
        logger.info(f"   Update class names in the file to match your categories")
        return True
    except Exception as e:
        logger.error(f"Error creating config file: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Label images for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python label_images.py --train-dir dataset/images/train
  python label_images.py --train-dir dataset/images/train --tool label-studio
  python label_images.py --train-dir dataset/images/train --tool roboflow
  python label_images.py --train-dir dataset/images/train --config --num-classes 3
        """
    )
    
    parser.add_argument('--train-dir', type=str, default='dataset/images/train',
                       help='Path to training images directory')
    parser.add_argument('--tool', type=str, choices=['label-studio', 'cvat', 'roboflow', 'menu'],
                       default='label-studio', help='Labeling tool to use (default: label-studio)')
    parser.add_argument('--config', action='store_true',
                       help='Create dataset configuration file only')
    parser.add_argument('--num-classes', type=int,
                       help='Number of object classes')
    
    args = parser.parse_args()
    
    train_dir = args.train_dir
    
    # Validate directory
    if not Path(train_dir).exists():
        logger.error(f"Training directory not found: {train_dir}")
        return 1
    
    # Create config if requested
    if args.config:
        dataset_dir = Path(train_dir).parent.parent  # Go up to dataset root
        return 0 if create_dataset_config(str(dataset_dir), args.num_classes) else 1
    
    # Show menu if not specified
    if args.tool == 'menu':
        display_menu()
        choice = input("\nSelect tool (1-5): ").strip()
        tool_map = {
            '1': 'label-studio',
            '2': 'cvat',
            '3': 'roboflow',
            '4': 'config'
        }
        args.tool = tool_map.get(choice, 'menu')
    
    # Create and run labeling tool
    tools = {
        'label-studio': AnnotationLabeling,
        'cvat': CVATLabeling,
        'roboflow': RoboflowAI
    }
    
    if args.tool == 'config':
        dataset_dir = Path(train_dir).parent.parent
        return 0 if create_dataset_config(str(dataset_dir), args.num_classes) else 1
    
    if args.tool not in tools:
        logger.error(f"Unknown tool: {args.tool}")
        return 1
    
    try:
        tool = tools[args.tool](train_dir)
        if not tool.run():
            logger.warning("Tool did not complete successfully")
        
        # Create config file after annotation
        logger.info("\n✅ After you finish annotation:")
        logger.info("1. Save all label files in YOLO format")
        logger.info("2. Place .txt files in dataset/labels/train/")
        logger.info("3. Create dataset configuration:")
        logger.info("   python label_images.py --config --num-classes 3")
        logger.info("4. Start training:")
        logger.info("   python train_optimized.py --data dataset/data.yaml")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
