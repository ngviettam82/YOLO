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
.venv\Scripts\activate.bat

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

**Options:**
```
--raw-dir           Path to raw dataset directory (default: raw_dataset)
--output-dir        Path to output directory (default: dataset)
--train             Training set ratio (default: 0.7)
--val               Validation set ratio (default: 0.2)
--test              Test set ratio (default: 0.1)
--seed              Random seed for reproducibility (default: 42)
--move              Move files instead of copying
```

---

## Annotation Tools

### 1. **LabelImg** (Recommended for beginners) 🌟

Fast, offline desktop tool for bounding box annotation.

**Installation:**
```bash
pip install labelimg
```

**Usage:**
```bash
python scripts/label_images.py --tool labelimg
# Or directly:
labelimg dataset/images/train yolo
```

**Features:**
- ✅ Works offline
- ✅ Fast annotation
- ✅ Keyboard shortcuts
- ✅ Automatic YOLO format export

**Steps:**
1. Launch LabelImg
2. Select folder: `dataset/images/train/`
3. Draw bounding boxes around objects
4. Define class names
5. Save in YOLO format

📖 [LabelImg Documentation](https://github.com/heartexlabs/labelImg)

---

### 2. **CVAT** (Best for teams) 👥

Professional web-based annotation platform with collaboration features.

**Installation (requires Docker):**
```bash
# Install Docker: https://www.docker.com/products/docker-desktop

# Run CVAT
docker run -d -p 8080:8080 --name cvat -v cvat_db:/var/lib/postgresql cvat/cvat:latest
```

**Usage:**
```bash
python scripts/label_images.py --tool cvat
```

1. Open http://localhost:8080
2. Create project
3. Upload images from `dataset/images/train/`
4. Annotate with team
5. Export as YOLO v5 PyTorch

**Features:**
- ✅ Team collaboration
- ✅ AI-assisted annotation
- ✅ Version control
- ✅ Quality management

📖 [CVAT Documentation](https://opencv.github.io/cvat/)

---

### 3. **Label Studio** (Easy web-based) 🌐

Simple web interface, minimal setup required.

**Installation:**
```bash
pip install label-studio
```

**Usage:**
```bash
python scripts/label_images.py --tool label-studio
# Or directly:
label-studio
```

1. Go to http://localhost:8080
2. Create project
3. Import images from `dataset/images/train/`
4. Create annotations
5. Export in YOLO v5 PyTorch format

**Features:**
- ✅ Web-based
- ✅ Easy to use
- ✅ Multiple export formats
- ✅ Fast setup

📖 [Label Studio Documentation](https://labelstud.io/)

---

### 4. **OpenLabeling** (Fastest annotation) ⚡

Lightweight desktop tool optimized for speed.

**Installation:**
```bash
git clone https://github.com/Cartucho/OpenLabeling
cd OpenLabeling
pip install -r requirements.txt
```

**Usage:**
```bash
python scripts/label_images.py --tool openlabeling
# Or directly:
python main.py
```

**Features:**
- ✅ Ultra-fast
- ✅ Keyboard shortcuts
- ✅ Lightweight
- ✅ Automatic YOLO format

📖 [OpenLabeling Repository](https://github.com/Cartucho/OpenLabeling)

---

### 5. **Roboflow** (Cloud AI-assisted) 🤖

AI-powered annotation with minimal manual work.

**Usage:**
```bash
python scripts/label_images.py --tool roboflow
```

1. Sign up at https://roboflow.com (free tier)
2. Create dataset
3. Upload images from `dataset/images/train/`
4. Use AI auto-annotation
5. Review and adjust
6. Download YOLO format annotations

**Features:**
- ✅ AI auto-annotation (saves time!)
- ✅ No installation
- ✅ Cloud-based
- ✅ Free tier available

---

## After Annotation

### 1. Move Label Files

Ensure all `.txt` label files are in the correct location:

```
dataset/
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   └── test/
```

### 2. Create Dataset Configuration

```bash
python scripts/label_images.py --config --num-classes 3
```

This creates `dataset/data.yaml`:

```yaml
path: /full/path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # Number of classes
names:
  0: person
  1: car
  2: dog
```

**Edit class names to match your objects!**

---

## Start Training

Once your dataset is ready:

```bash
# Activate virtual environment
.venv\Scripts\activate.bat

# Start training
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 24
```

For more options:
```bash
python train_optimized.py --help
```

---

## YOLO Label Format

Each image needs a corresponding `.txt` file with the same name.

**Format (one line per object):**
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `image1.txt`
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

- `class_id`: 0-indexed class number
- `x_center, y_center`: Normalized (0-1) center coordinates
- `width, height`: Normalized (0-1) bounding box dimensions

Most annotation tools handle this automatically!

---

## Troubleshooting

### Images not found during split
```bash
# Check raw_dataset folder has images
dir raw_dataset/
# Supported formats: .jpg, .jpeg, .png, .bmp, .gif, .webp
```

### Annotation tool won't launch
```bash
# Try installing the tool separately
pip install labelimg

# Or check installation
python -c "import labelimg; print(labelimg.__file__)"
```

### Label files in wrong format
- Ensure `.txt` files are in `dataset/labels/train/` (not in `images/`)
- File names must match image names (e.g., `image1.jpg` → `image1.txt`)
- Use YOLO format (not Pascal VOC, COCO, etc.)

### Training won't start
```bash
# Verify dataset structure
python scripts/split_dataset.py --raw-dir raw_dataset --output-dir dataset

# Create config
python scripts/label_images.py --config --num-classes 3

# Check config file
type dataset/data.yaml
```

---

## Dataset Best Practices

✅ **Do:**
- Use high-quality images
- Include diverse lighting conditions
- Capture objects from multiple angles
- Aim for 100+ images per class minimum
- Balance classes (similar number of each class)
- Remove duplicates

❌ **Don't:**
- Use very low-resolution images
- Mix different cameras/sensors without reason
- Have huge class imbalance
- Annotate inconsistently
- Skip difficult examples

---

## Advanced: Custom Splitting

For more control over dataset preparation:

```python
from scripts.split_dataset import DatasetSplitter

splitter = DatasetSplitter(
    raw_dataset_dir='raw_dataset',
    output_dir='dataset',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

success = splitter.run()
```

---

## Need Help?

- 📖 [YOLO Official Docs](https://docs.ultralytics.com/)
- 💬 [LabelImg Issues](https://github.com/heartexlabs/labelImg/issues)
- 🐞 [OpenLabeling Issues](https://github.com/Cartucho/OpenLabeling/issues)
- 📚 [Computer Vision Basics](https://docs.ultralytics.com/datasets/)

---

**Ready to train your model?** 🚀

```bash
python train_optimized.py --data dataset/data.yaml
```
