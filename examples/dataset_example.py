"""
Quick example script demonstrating dataset preparation workflow
"""

import os
from pathlib import Path
from scripts.split_dataset import DatasetSplitter


def example_workflow():
    """
    Example workflow for preparing a YOLO dataset
    """
    
    print("\n" + "="*60)
    print("YOLO Dataset Preparation Example")
    print("="*60 + "\n")
    
    # Step 1: Check raw dataset
    raw_dataset = Path("raw_dataset")
    if not raw_dataset.exists():
        print("❌ raw_dataset folder not found!")
        print("📁 Create the folder and add your images there")
        return False
    
    images = list(raw_dataset.glob("*.jpg")) + list(raw_dataset.glob("*.png"))
    print(f"✅ Found {len(images)} images in raw_dataset/")
    
    if len(images) == 0:
        print("❌ No images found! Add images to raw_dataset/")
        return False
    
    # Step 2: Split dataset with default ratios (70/20/10)
    print("\n" + "-"*60)
    print("Step 1: Splitting dataset...")
    print("-"*60)
    
    splitter = DatasetSplitter(
        raw_dataset_dir="raw_dataset",
        output_dir="dataset",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    if not splitter.run():
        print("❌ Dataset split failed!")
        return False
    
    # Step 3: Show next steps
    print("\n" + "-"*60)
    print("Step 2: Labeling images...")
    print("-"*60)
    print("\n✅ Images have been split into train/val/test folders")
    print("\n📝 Next: Label the training images")
    print("\nRun one of these commands:")
    print("  # Option 1: Interactive menu")
    print("  python scripts/label_images.py")
    print("  # Option 2: Use LabelImg directly")
    print("  labelimg dataset/images/train yolo")
    print("  # Option 3: Use Label Studio")
    print("  label-studio")
    
    print("\n" + "-"*60)
    print("Step 3: Create dataset configuration...")
    print("-"*60)
    print("\nAfter annotation, run:")
    print("  python scripts/label_images.py --config --num-classes 3")
    print("\nUpdate class names in dataset/data.yaml to match your objects")
    
    print("\n" + "-"*60)
    print("Step 4: Train your model...")
    print("-"*60)
    print("\nFinally, start training with:")
    print("  python train_optimized.py --data dataset/data.yaml --epochs 100")
    
    print("\n" + "="*60)
    print("Setup Complete! 🎉")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = example_workflow()
    exit(0 if success else 1)
