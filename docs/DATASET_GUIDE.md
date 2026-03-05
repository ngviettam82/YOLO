# Dataset Guide

Prepare your dataset for YOLO training.

---

## 📋 Overview

This guide explains how to:
1. Organize raw images
2. Split into train/val/test
3. Annotate images with bounding boxes
4. Create dataset configuration

---

## 📁 Step 1: Organize Raw Images

### Create Raw Dataset Folder

```
YOLO/
└── raw_dataset/          # Create this folder
    ├── image1.jpg
    ├── image2.png
    ├── image3.jpg
    └── ...              # Add all your images here
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`

### Add Your Images

1. Create a folder named `raw_dataset` in the project root
2. Copy all your training images to this folder
3. Images can be in any subdirectory structure (script will find them recursively)

---

## 🔄 Step 2: Split Dataset (Auto)

### Option 1: Double-Click (Easiest) ⭐

**Step 2:** Simply **double-click** `2.dataset.bat` in the project root.

The script will:
1. Count images in `raw_dataset/`
2. Split into train (70%) / val (20%) / test (10%)
3. Organize by type: images → labels
4. Create `dataset/data.yaml` config

**Time:** ~1-5 minutes

### Option 2: Command Line

```batch
2.dataset.bat
```

Or manually:
```batch
.venv\Scripts\activate.bat
python scripts\split_dataset.py --train 0.7 --val 0.2 --test 0.1
```

### Custom Split Ratios

```batch
python scripts\split_dataset.py --train 0.8 --val 0.15 --test 0.05
```

---

## Step 3: Label Images

### Option 1: Double-Click (Easiest)

**Step 2:** Simply **double-click** `2.label.bat` in the project root.

This will:
1. Launch Label Studio at http://localhost:8080
2. You create a project and upload/import images
3. Draw bounding boxes and assign labels (fire, smoke)
4. Export in YOLO format

**For large datasets (200+ images):** Use server-based import:
```batch
AutoLabel\import_to_label_studio.bat
```
This avoids the `DATA_UPLOAD_MAX_NUMBER_FILES` upload limit.

### Annotation Tool

**Label Studio (Default - Web-based)**
```batch
python scripts\label_images.py --tool label-studio
```
- Modern web interface
- Server-based import for large datasets
- Export in YOLO format

---

## 📊 Step 4: Verify Dataset

After labeling, verify your dataset structure:

```
dataset/
├── images/
│   ├── train/          # 70% of images
│   ├── val/            # 20% of images
│   └── test/           # 10% of images
├── labels/
│   ├── train/          # Corresponding .txt files
│   ├── val/
│   └── test/
└── data.yaml           # Config file
```

### Edit data.yaml

Open `dataset/data.yaml` and update class names:

```yaml
path: C:/path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['fire', 'smoke']  # class names
```

Custom examples:
```yaml
nc: 1
names: ['fire']
```

Example for vehicle detection:
```yaml
nc: 3
names: ['fire', 'smoke', 'vehicle']
```

---

## ✅ Next Steps

After dataset is ready:

1. **Verify** the data.yaml has correct class names
2. **Train** your model (Step 4):
   ```batch
   4.train.bat
   ```

See `docs/TRAINING_GUIDE.md` for training instructions.

---

## 🆘 Troubleshooting

### No images found in raw_dataset

**Error:** `ERROR: No images found in raw_dataset/`

**Solution:**
1. Create `raw_dataset/` folder in project root
2. Add images to it
3. Supported formats: jpg, jpeg, png, bmp, gif, webp

### No training images found after split

**Error:** `ERROR: No images found in dataset/images/train/`

**Solution:**
1. Run `quickstart_dataset.bat` again
2. Check `raw_dataset/` has images
3. Check image file extensions are valid

### LabelImg won't open

**Error:** LabelImg tool doesn't launch

**Solution:**
1. Ensure PyQt5 is installed (included in requirements.txt)
2. Try manually installing:
   ```batch
   pip install PyQt5 labelimg
   ```
3. Try alternate tool: `python scripts\label_images.py --tool label-studio`

### Dataset splitting errors

**Error:** Dataset splitting fails

**Solution:**
1. Check raw_dataset images are valid (not corrupted)
2. Try with fewer images first
3. Check disk space (need ~2x raw image size)

---

## 📚 Advanced Topics

### Data Augmentation

Training config includes augmentation settings in `configs/train_config.yaml`:
- Mosaic: 1.0 (always)
- Mixup: 0.2 (20% chance)
- Copy-paste: 0.5 (50% chance)
- Rotation: ±20°
- Translation: ±30%
- Scale: 0.85-1.15x

### Class Imbalance

If some classes are rare:
1. Increase copies of rare class images in raw_dataset
2. Increase augmentation in config
3. Use weighted loss (handled automatically)

### Small Dataset

For datasets < 100 images:
1. Increase augmentation
2. Use transfer learning (pre-trained model)
3. Reduce model size (yolo11n.pt instead of yolo11m.pt)

