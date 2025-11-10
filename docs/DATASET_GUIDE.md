````markdown
# Dataset Preparation Guide

Complete guide for preparing your dataset for YOLO training.

---

## 📂 Directory Structure

```
raw_dataset/              ← Your raw images go here
├── image1.jpg
├── image2.png
└── ...

dataset/                  ← Auto-organized dataset (created by split_dataset.py)
├── images/
│   ├── train/  (70%)
│   ├── val/    (20%)
│   └── test/   (10%)
├── labels/
│   ├── train/  (your annotations)
│   ├── val/
│   └── test/
└── data.yaml  (config file)
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Add Images to raw_dataset/

1. Collect your images
2. Place them in `raw_dataset/` folder
3. Supported formats: JPG, PNG, BMP, GIF, WebP

### Step 2: Split Dataset

```powershell
python scripts/split_dataset.py
```

**Default:** 70% train, 20% val, 10% test

**Custom split:**
```powershell
python scripts/split_dataset.py --train 0.8 --val 0.15 --test 0.05
```

### Step 3: Label Images

```powershell
.\quickstart_label.ps1
```

Or with specific tool:
```powershell
.\quickstart_label.ps1 -Tool labelimg
.\quickstart_label.ps1 -Tool cvat
.\quickstart_label.ps1 -Tool label-studio
.\quickstart_label.ps1 -Tool openlabeling
.\quickstart_label.ps1 -Tool roboflow
```

Select your preferred annotation tool from the interactive menu.

---

## 🏷️ Annotation Tools

| Tool | Type | Setup | Speed | Best For |
|------|------|-------|-------|----------|
| **LabelImg** | Desktop | 1 min | Fast | Beginners |
| **CVAT** | Web | 10 min | Medium | Teams |
| **Label Studio** | Web | 2 min | Medium | Web users |
| **OpenLabeling** | Desktop | 3 min | Fast | Speed |
| **Roboflow** | Cloud | 0 min | Slow | AI-assisted |

**Recommended:** LabelImg (offline, fast, easy to use)

---

## 📝 YOLO Label Format

Each image needs a `.txt` file with the same name:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1).

**Example:** `image1.txt`
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

---

## ⚙️ Dataset Configuration

After labeling, create your dataset config:

```powershell
python scripts/label_images.py --config --num-classes 3
```

This creates `dataset/data.yaml`:
```yaml
path: dataset
train: images/train
val: images/val
test: images/test

nc: 3
names: ['class0', 'class1', 'class2']
```

**Update class names in `dataset/data.yaml`** to match your objects.

---

## ✅ Best Practices

✅ **Do:**
- High-quality images (at least 640x640)
- Diverse angles and lighting
- 100+ images per class minimum
- Balanced classes (similar count per class)
- Consistent annotation style

❌ **Don't:**
- Low resolution images
- Extreme class imbalance
- Inconsistent annotations
- Mislabeled images
- Skip difficult examples

---

## 📊 Dataset Statistics

Check your dataset:
```powershell
python scripts/split_dataset.py --stats
```

---

## 🎯 Next Steps

Once dataset is ready:

```powershell
python train_optimized.py --data dataset/data.yaml --epochs 100
```

See `docs/TRAINING_GUIDE.md` for training instructions.

````
