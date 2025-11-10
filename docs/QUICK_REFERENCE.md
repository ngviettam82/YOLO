# 🎯 Dataset Management Quick Reference

## 📊 Complete Workflow

```
Step 1: COLLECT IMAGES
├─ Place in: raw_dataset/

Step 2: AUTO-SPLIT
├─ Default: 70% train, 20% val, 10% test
└─ python scripts/split_dataset.py

Step 3: LABEL IMAGES
├─ Choose annotation tool
└─ python scripts/label_images.py

Step 4: CREATE CONFIG
└─ python scripts/label_images.py --config --num-classes 3

Step 5: TRAIN MODEL
└─ python train_optimized.py --data dataset/data.yaml
```

---

## 🚀 Commands Cheat Sheet

### Using Interactive Manager (Easiest)
```bash
.\manage_dataset.bat        # Windows Batch
.\manage_dataset.ps1        # PowerShell
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
python scripts/label_images.py --tool label-studio

# 5. Create dataset config (3 classes)
python scripts/label_images.py --config --num-classes 3

# 6. Start training
python train_optimized.py --data dataset/data.yaml
```

---

## 🏷️ Annotation Tools

| Tool | Type | Setup | Speed | Best For |
|------|------|-------|-------|----------|
| **LabelImg** | Desktop | 1 min | ⚡⚡⚡ | Beginners |
| **CVAT** | Web | 10 min | ⚡⚡ | Teams |
| **Label Studio** | Web | 2 min | ⚡⚡ | Web users |
| **OpenLabeling** | Desktop | 3 min | ⚡⚡⚡ | Speed |
| **Roboflow** | Cloud | 0 min | ⚡ | AI-assisted |

---

## 💾 YOLO Label Format

```
<class_id> <x_center> <y_center> <width> <height>
```

Example: `image1.txt`
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Add images
cp your_images/* raw_dataset/

# 2. Split
python scripts/split_dataset.py

# 3. Label
python scripts/label_images.py

# 4. Config
python scripts/label_images.py --config --num-classes 3

# 5. Train
python train_optimized.py --data dataset/data.yaml
```

---

## 📚 Documentation

- **Installation**: `docs/INSTALLATION.md`
- **Dataset Guide**: `docs/DATASET_GUIDE.md`
- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **Main Readme**: `README.md`

---

**Happy training! 🚀**
