#!/usr/bin/env python3
"""
Dataset Preparation Utilities
Tools for preparing and managing YOLO datasets
"""

import os
import shutil
from pathlib import Path
import random
import yaml


def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Directory containing images and labels
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        test_ratio: Ratio of test data (default: 0.1)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(list(source_dir.glob(f"**/{ext}")))
        images.extend(list(source_dir.glob(f"**/{ext.upper()}")))
    
    if not images:
        print(f"❌ No images found in {source_dir}")
        return
    
    print(f"📊 Found {len(images)} images")
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate splits
    total = len(images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    print(f"   Training: {len(train_images)} images")
    print(f"   Validation: {len(val_images)} images")
    print(f"   Test: {len(test_images)} images")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_split(images, split_name):
        print(f"\n📁 Copying {split_name} split...")
        for img_path in images:
            # Copy image
            img_dst = output_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, img_dst)
            
            # Copy label if exists
            label_path = img_path.parent / 'labels' / f"{img_path.stem}.txt"
            if not label_path.exists():
                # Try same directory
                label_path = img_path.with_suffix('.txt')
            
            if label_path.exists():
                label_dst = output_dir / split_name / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, label_dst)
    
    copy_split(train_images, 'train')
    copy_split(val_images, 'val')
    if test_images:
        copy_split(test_images, 'test')
    
    print(f"\n✅ Dataset split complete!")
    print(f"📁 Output directory: {output_dir}")


def create_dataset_yaml(output_dir, class_names, dataset_name="custom_dataset"):
    """
    Create dataset YAML configuration file
    
    Args:
        output_dir: Dataset directory
        class_names: List of class names
        dataset_name: Name of the dataset
    """
    output_dir = Path(output_dir)
    
    # Create YAML content
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Save YAML file
    yaml_path = output_dir / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n✅ Dataset YAML created: {yaml_path}")
    print(f"\n📄 Content:")
    print(yaml.dump(yaml_content, default_flow_style=False))


def validate_dataset_structure(dataset_dir):
    """
    Validate dataset structure and report issues
    
    Args:
        dataset_dir: Dataset directory to validate
    """
    dataset_dir = Path(dataset_dir)
    
    print(f"🔍 Validating dataset: {dataset_dir}")
    print("="*80)
    
    issues = []
    
    # Check required directories
    for split in ['train', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            issues.append(f"❌ Missing {split} directory")
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            issues.append(f"❌ Missing {split}/images directory")
        if not labels_dir.exists():
            issues.append(f"❌ Missing {split}/labels directory")
        
        if images_dir.exists() and labels_dir.exists():
            # Count files
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            labels = list(labels_dir.glob('*.txt'))
            
            print(f"\n📊 {split.capitalize()} Split:")
            print(f"   Images: {len(images)}")
            print(f"   Labels: {len(labels)}")
            
            if len(images) == 0:
                issues.append(f"⚠️  No images in {split}/images")
            
            # Check for missing labels
            missing_labels = []
            for img in images:
                label_file = labels_dir / f"{img.stem}.txt"
                if not label_file.exists():
                    missing_labels.append(img.name)
            
            if missing_labels:
                issues.append(f"⚠️  {len(missing_labels)} images without labels in {split}")
                if len(missing_labels) <= 5:
                    for name in missing_labels:
                        print(f"      Missing label: {name}")
    
    # Summary
    print(f"\n{'='*80}")
    if issues:
        print(f"⚠️  Validation completed with {len(issues)} issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"✅ Dataset structure is valid!")
    print(f"{'='*80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Dataset Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--source', type=str, required=True,
                            help='Source directory with images and labels')
    split_parser.add_argument('--output', type=str, required=True,
                            help='Output directory for split dataset')
    split_parser.add_argument('--train', type=float, default=0.8,
                            help='Training ratio (default: 0.8)')
    split_parser.add_argument('--val', type=float, default=0.1,
                            help='Validation ratio (default: 0.1)')
    split_parser.add_argument('--test', type=float, default=0.1,
                            help='Test ratio (default: 0.1)')
    
    # Create YAML command
    yaml_parser = subparsers.add_parser('create-yaml', help='Create dataset YAML file')
    yaml_parser.add_argument('--output', type=str, required=True,
                           help='Dataset directory')
    yaml_parser.add_argument('--classes', nargs='+', required=True,
                           help='List of class names')
    yaml_parser.add_argument('--name', type=str, default='custom_dataset',
                           help='Dataset name (default: custom_dataset)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset structure')
    validate_parser.add_argument('--dataset', type=str, required=True,
                               help='Dataset directory to validate')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test
        )
    
    elif args.command == 'create-yaml':
        create_dataset_yaml(
            output_dir=args.output,
            class_names=args.classes,
            dataset_name=args.name
        )
    
    elif args.command == 'validate':
        validate_dataset_structure(dataset_dir=args.dataset)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
