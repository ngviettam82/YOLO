# 🎯 Dataset Management Quick Reference

## 📊 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO DATASET WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: COLLECT IMAGES
├─ Take photos of your objects
├─ Use images from datasets/cameras
└─ Place in: raw_dataset/

Step 2: AUTO-SPLIT (Automated!)
├─ Splits into train/val/test
├─ Default: 70% train, 20% val, 10% test
└─ Result: 
    ├─ dataset/images/train/
    ├─ dataset/images/val/
    └─ dataset/images/test/

Step 3: LABEL IMAGES (Semi-automated)
├─ Choose annotation tool
├─ Annotate training images
└─ Result:
    ├─ dataset/labels/train/*.txt
    ├─ dataset/labels/val/*.txt
    └─ dataset/labels/test/*.txt

Step 4: CREATE CONFIG (Automated!)
├─ Generate data.yaml
├─ Define class names
└─ Result: dataset/data.yaml

Step 5: TRAIN MODEL
└─ python train_optimized.py --data dataset/data.yaml
```

---

## 🚀 Commands Cheat Sheet

### Using Interactive Manager (Easiest)
```bash
# Windows Batch
.\manage_dataset.bat

# PowerShell
.\manage_dataset.ps1
```

### Step-by-Step Commands

```bash
# 1. Split dataset (default 70/20/10)
python scripts/split_dataset.py

# 2. Custom split (80/10/10)
python scripts/split_dataset.py --train 0.8 --val 0.1 --test 0.1

# 3. Launch annotation tools (interactive menu)
python scripts/label_images.py

# 4. Specific annotation tool
python scripts/label_images.py --tool labelimg
python scripts/label_images.py --tool cvat
python scripts/label_images.py --tool label-studio

# 5. Create dataset config (3 classes)
python scripts/label_images.py --config --num-classes 3

# 6. Start training
python train_optimized.py --data dataset/data.yaml
```

---

## 📁 Directory Structure Reference

```
YOLO Project Root
│
├── raw_dataset/                 ← YOUR RAW IMAGES HERE
│   ├── photo1.jpg
│   ├── photo2.png
│   └── ...
│
├── dataset/                     ← AUTO-ORGANIZED
│   ├── images/
│   │   ├── train/              ← 70% images
│   │   ├── val/                ← 20% images
│   │   └── test/               ← 10% images
│   ├── labels/
│   │   ├── train/              ← YOUR ANNOTATIONS
│   │   ├── val/
│   │   └── test/
│   └── data.yaml               ← CONFIG (AUTO-GENERATED)
│
├── scripts/
│   ├── split_dataset.py        ← SPLITTING TOOL
│   ├── label_images.py         ← ANNOTATION LAUNCHER
│   ├── inference.py            ← RUN PREDICTIONS
│   ├── validate_model.py       ← TEST MODEL
│   └── export_model.py         ← EXPORT WEIGHTS
│
├── manage_dataset.bat          ← INTERACTIVE MANAGER
├── manage_dataset.ps1          ← INTERACTIVE MANAGER (PS)
│
├── DATASET_GUIDE.md            ← DETAILED GUIDE
├── DATASET_SETUP_SUMMARY.md    ← THIS SUMMARY
└── README.md                   ← MAIN README
```

---

## 🏷️ Annotation Tools Overview

| Tool | Type | Setup | Speed | Best For |
|------|------|-------|-------|----------|
| **LabelImg** | Desktop | 1 min | ⚡⚡⚡ | Beginners, fast work |
| **CVAT** | Web | 10 min | ⚡⚡ | Teams, large projects |
| **Label Studio** | Web | 2 min | ⚡⚡ | Web preference |
| **OpenLabeling** | Desktop | 3 min | ⚡⚡⚡ | Speed enthusiasts |
| **Roboflow** | Cloud | 0 min | ⚡ | AI-assisted, ease |

**Recommendation:** Start with **LabelImg** if you want the fastest, simplest setup!

---

## 💾 File Format

### YOLO Label Format (`.txt` files)
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `image1.txt`
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

- Line 1: Object 1 (class 0 at center)
- Line 2: Object 2 (class 1 at top-left)
- All coordinates are normalized (0-1 range)

**Most annotation tools handle this automatically!**

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Add images
cp your_images/* raw_dataset/

# 2. Split
python scripts/split_dataset.py

# 3. Label
python scripts/label_images.py
# Choose: LabelImg → Draw boxes → Save

# 4. Config
python scripts/label_images.py --config --num-classes 3
# Edit dataset/data.yaml with your class names

# 5. Train
python train_optimized.py --data dataset/data.yaml
```

**Total Time:** ~5 minutes setup + annotation time!

---

## ✅ Checklist

After each step, verify:

### ✓ After Splitting
```
✓ raw_dataset/ has your images
✓ dataset/images/train/ has 70% of images
✓ dataset/images/val/ has 20% of images
✓ dataset/images/test/ has 10% of images
```

### ✓ After Labeling
```
✓ Opened annotation tool
✓ Drew boxes on train images
✓ Saved labels in YOLO format
✓ .txt files are in dataset/labels/train/
```

### ✓ Before Training
```
✓ dataset/data.yaml exists
✓ Class names are correct in data.yaml
✓ Label files match image names
✓ All paths are correct in data.yaml
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| No images found | Check `raw_dataset/` folder is not empty |
| Tool won't launch | Run `pip install labelimg` first |
| Labels in wrong place | Ensure `.txt` files are in `dataset/labels/` |
| Training won't start | Check `dataset/data.yaml` has correct class count |
| Out of memory | Reduce batch size in train command |

See **DATASET_GUIDE.md** for detailed troubleshooting!

---

## 📚 More Info

- **Quick Guide**: This file (you are here!)
- **Detailed Guide**: `DATASET_GUIDE.md`
- **Training Guide**: `TRAINING_GUIDE.md`
- **Full Docs**: `README.md`

---

## 🎓 Learn by Example

Run the example script:
```bash
python examples/dataset_example.py
```

This shows the complete workflow with explanations!

---

## 🎉 You're Ready!

```
📊 Dataset structure: ✓ Ready
🔧 Tools installed: ✓ Ready  
📖 Documentation: ✓ Ready
🚀 Let's train! ✓ Ready

Next step: .\manage_dataset.bat
```

---

**Happy labeling! 🏷️🚀**
