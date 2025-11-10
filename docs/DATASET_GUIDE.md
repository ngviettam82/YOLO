````markdown
# Dataset Management Guide

This guide will help you prepare your dataset for YOLO training with automated splitting and labeling.

## Directory Structure

```
YOLO/
├── raw_dataset/           # 📁 Place your raw images here
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
│
├── dataset/               # 📁 Organized dataset (auto-generated)
│   ├── images/
│   │   ├── train/        # Training images (70%)
│   │   ├── val/          # Validation images (20%)
│   │   └── test/         # Test images (10%)
│   ├── labels/
│   │   ├── train/        # Training annotations
│   │   ├── val/          # Validation annotations
│   │   └── test/         # Test annotations
│   └── data.yaml         # Dataset configuration
│
├── scripts/
│   ├── split_dataset.py   # Split raw images into train/val/test
│   └── label_images.py    # Launch annotation tools
```

## Quick Start (3 Steps)

### Step 1: Prepare Raw Images

1. Collect all your images
2. Copy them to the `raw_dataset/` folder
3. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`

```bash
# Example: Copy your images
cp your_images/* raw_dataset/
```

### Step 2: Split Dataset

Split your images into train/val/test sets with optimized ratios (70/20/10):

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Split dataset
python scripts/split_dataset.py
```

**Output:**
- ✅ Train: ~70% of images → `dataset/images/train/`
- ✅ Val:   ~20% of images → `dataset/images/val/`
- ✅ Test:  ~10% of images → `dataset/images/test/`

### Step 3: Label Images

Label the training images using your preferred tool:

```bash
python scripts/label_images.py
```

Then select from 5 annotation tools (or create config directly).

---

## Detailed Usage

### Split Dataset with Custom Ratios

```bash
# Default: 70% train, 20% val, 10% test
python scripts/split_dataset.py

# Custom split: 80% train, 15% val, 5% test
python scripts/split_dataset.py --train 0.8 --val 0.15 --test 0.05

# Move files instead of copying (saves disk space)
python scripts/split_dataset.py --move
```

---

## Annotation Tools

### 1. **LabelImg** (Recommended for beginners) 🌟

Fast, offline desktop tool for bounding box annotation.

```bash
pip install labelimg
python scripts/label_images.py --tool labelimg
```

### 2. **CVAT** (Best for teams) 👥

Professional web-based annotation platform.

```bash
python scripts/label_images.py --tool cvat
```

### 3. **Label Studio** (Easy web-based) 🌐

Simple web interface, minimal setup required.

```bash
pip install label-studio
python scripts/label_images.py --tool label-studio
```

### 4. **OpenLabeling** (Fastest annotation) ⚡

Lightweight desktop tool optimized for speed.

```bash
python scripts/label_images.py --tool openlabeling
```

### 5. **Roboflow** (Cloud AI-assisted) 🤖

AI-powered annotation with minimal manual work.

```bash
python scripts/label_images.py --tool roboflow
```

---

## After Annotation

### Create Dataset Configuration

```bash
python scripts/label_images.py --config --num-classes 3
```

This creates `dataset/data.yaml` with your class names.

---

## YOLO Label Format

Each image needs a corresponding `.txt` file:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `image1.txt`
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

---

## Start Training

Once your dataset is ready:

```bash
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 24
```

---

## Dataset Best Practices

✅ **Do:**
- Use high-quality images
- Include diverse lighting conditions
- Capture objects from multiple angles
- Aim for 100+ images per class minimum
- Balance classes (similar number of each class)

❌ **Don't:**
- Use very low-resolution images
- Have huge class imbalance
- Annotate inconsistently
- Skip difficult examples

---

**Ready to train your model?** 🚀

```bash
python train_optimized.py --data dataset/data.yaml
```

````
