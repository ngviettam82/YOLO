#!/usr/bin/env python3
"""
Convert Label Studio notes.json to YOLO data.yaml format
"""

import json
import yaml
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_notes_to_yaml(raw_dataset_dir: str, output_dir: str) -> bool:
    """
    Convert Label Studio notes.json to YOLO data.yaml
    
    Args:
        raw_dataset_dir: Path to raw_dataset folder (contains notes.json)
        output_dir: Path to dataset folder (where data.yaml will be saved)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        raw_dataset_path = Path(raw_dataset_dir)
        output_path = Path(output_dir)
        notes_file = raw_dataset_path / "notes.json"
        
        # Check if notes.json exists
        if not notes_file.exists():
            logger.warning(f"notes.json not found in {raw_dataset_path}")
            logger.warning("Creating default data.yaml with 2 classes (Fire, Smoke)")
            
            # Create default config if notes.json doesn't exist
            # Use absolute path for compatibility with ultralytics
            default_config = {
                'path': str(output_path.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': 2,
                'names': {0: 'Fire', 1: 'Smoke'}
            }
        else:
            # Read notes.json
            with open(notes_file, 'r') as f:
                notes_data = json.load(f)
            
            logger.info(f"Read notes.json from {notes_file}")
            
            # Extract class information
            categories = notes_data.get('categories', [])
            class_names = {}
            
            for category in categories:
                class_id = category.get('id', 0)
                class_name = category.get('name', f'class_{class_id}')
                class_names[class_id] = class_name
            
            logger.info(f"Found {len(class_names)} classes:")
            for class_id, class_name in sorted(class_names.items()):
                logger.info(f"  - {class_id}: {class_name}")
            
            # Create YOLO data.yaml format
            # Use absolute path for compatibility with ultralytics
            default_config = {
                'path': str(output_path.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(class_names),
                'names': class_names
            }
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write data.yaml
        data_yaml_path = output_path / "data.yaml"
        with open(data_yaml_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"\n✅ Created data.yaml at {data_yaml_path}")
        logger.info(f"\nConfiguration:")
        logger.info(f"  Path: {default_config['path']}")
        logger.info(f"  Train: {default_config['train']}")
        logger.info(f"  Val: {default_config['val']}")
        logger.info(f"  Test: {default_config['test']}")
        logger.info(f"  Classes: {default_config['nc']}")
        logger.info(f"  Names: {default_config['names']}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error converting notes.json: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Label Studio notes.json to YOLO data.yaml')
    parser.add_argument('--raw-dataset', type=str, default='raw_dataset',
                       help='Path to raw_dataset folder (default: raw_dataset)')
    parser.add_argument('--output', type=str, default='dataset',
                       help='Path to output dataset folder (default: dataset)')
    
    args = parser.parse_args()
    
    success = convert_notes_to_yaml(args.raw_dataset, args.output)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
