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


class LabelImg(LabelingTool):
    """LabelImg - graphical image annotation tool"""
    
    def check_installation(self) -> bool:
        """Check if LabelImg is installed"""
        try:
            import labelImg
            return True
        except ImportError:
            return False
    
    def run(self) -> bool:
        """Launch LabelImg for annotation"""
        logger.info("Launching LabelImg...")
        logger.info(f"Found {self.image_count} images to label in {self.train_dir}")
        
        try:
            # Import and run labelImg
            from labelImg.labelImg import APP
            import sys
            
            # Set up arguments for labelImg
            sys.argv = ['labelImg', str(self.train_dir), 'yolo']
            
            # Note: This is simplified. Full implementation would require more setup
            logger.info("Please use labelImg from command line:")
            logger.info(f"  labelImg {self.train_dir} yolo")
            return True
        except Exception as e:
            logger.error(f"Error launching LabelImg: {e}")
            return False


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


class AnnotationLabeling(LabelingTool):
    """Annotation tool - simple web-based annotation"""
    
    def check_installation(self) -> bool:
        """Check if annotation tool is available"""
        return True
    
    def run(self) -> bool:
        """Provide Annotation tool instructions"""
        logger.info("\n" + "="*60)
        logger.info("Simple Annotation Web Tool")
        logger.info("="*60)
        logger.info(f"\nFound {self.image_count} images to label in {self.train_dir}")
        logger.info("\nUsage:")
        logger.info("1. Visit: https://github.com/heartexlabs/label-studio")
        logger.info("2. Install Label Studio:")
        logger.info("   pip install label-studio")
        logger.info("3. Start Label Studio:")
        logger.info("   label-studio")
        logger.info("4. Create a project and import images from:")
        logger.info(f"   {self.train_dir}")
        logger.info("\n✅ Create bounding box annotations using the interface")
        logger.info("📝 Export in YOLO format (YOLO v5 PyTorch format)")
        logger.info("="*60 + "\n")
        return True


class OpenLabelImg(LabelingTool):
    """OpenLabeling - fast and efficient annotation tool"""
    
    def check_installation(self) -> bool:
        """Check if OpenLabeling is available"""
        return True
    
    def run(self) -> bool:
        """Provide OpenLabeling instructions"""
        logger.info("\n" + "="*60)
        logger.info("OpenLabeling - Fast Annotation Tool")
        logger.info("="*60)
        logger.info(f"\nFound {self.image_count} images to label in {self.train_dir}")
        logger.info("\nInstallation & Usage:")
        logger.info("1. Clone repository:")
        logger.info("   git clone https://github.com/Cartucho/OpenLabeling")
        logger.info("2. Install dependencies:")
        logger.info("   pip install -r requirements.txt")
        logger.info("3. Run the tool:")
        logger.info("   python main.py")
        logger.info("4. Open folder and select:")
        logger.info(f"   {self.train_dir}")
        logger.info("\n✅ Draw bounding boxes with mouse")
        logger.info("📝 Labels are automatically saved in YOLO format")
        logger.info("="*60 + "\n")
        return True


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
    print("\n1. LabelImg - Desktop GUI (Recommended for local work)")
    print("   • Fast and straightforward")
    print("   • Works offline")
    print("   • Native desktop application")
    
    print("\n2. CVAT - Web-based (Best for team collaboration)")
    print("   • Professional annotation platform")
    print("   • Requires Docker")
    print("   • Best for large teams")
    
    print("\n3. Label Studio - Web-based (Easy setup)")
    print("   • Simple web interface")
    print("   • Good for small to medium projects")
    print("   • Easy to install")
    
    print("\n4. OpenLabeling - Fast Desktop Tool")
    print("   • Lightweight and fast")
    print("   • Keyboard shortcuts for speed")
    print("   • Great for bounding box annotation")
    
    print("\n5. Roboflow - Cloud AI-Assisted (Easiest)")
    print("   • AI auto-annotation")
    print("   • No installation needed")
    print("   • Free tier available")
    
    print("\n6. Create Config File Only (Manual setup)")
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
  python label_images.py --train-dir dataset/images/train --tool labelimg
  python label_images.py --train-dir dataset/images/train --tool roboflow
  python label_images.py --train-dir dataset/images/train --config --num-classes 3
        """
    )
    
    parser.add_argument('--train-dir', type=str, default='dataset/images/train',
                       help='Path to training images directory')
    parser.add_argument('--tool', type=str, choices=['labelimg', 'cvat', 'label-studio', 
                                                      'openlabeling', 'roboflow', 'menu'],
                       default='menu', help='Labeling tool to use')
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
        choice = input("\nSelect tool (1-6): ").strip()
        tool_map = {
            '1': 'labelimg',
            '2': 'cvat',
            '3': 'label-studio',
            '4': 'openlabeling',
            '5': 'roboflow',
            '6': 'config'
        }
        args.tool = tool_map.get(choice, 'menu')
    
    # Create and run labeling tool
    tools = {
        'labelimg': LabelImg,
        'cvat': CVATLabeling,
        'label-studio': AnnotationLabeling,
        'openlabeling': OpenLabelImg,
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
