# Quick Start - Aerial Fire & Smoke Detection

## 4-Step Pipeline

Double-click these files **in order**:

```
Step 1: 1.install.bat          (Setup environment)
        ↓
Step 2: 2.label.bat            (Label images in Label Studio)
        ↓
Step 3: 3.dataset.bat          (Split into train/val/test)
        ↓
Step 4: 4.train.bat            (Train YOLO model)
```

---

## Before You Start

1. Place your images in `raw_dataset/` folder
2. Make sure you have Python 3.10 installed

---

## Step 1 - `1.install.bat` (Setup)

Creates virtual environment, installs PyTorch + CUDA, all dependencies.

**Time:** ~10-15 minutes

---

## Step 2 - `2.label.bat` (Label Images)

Opens Label Studio at http://localhost:8080 for image annotation.

**What to do:**
1. Create a project in Label Studio
2. Set labeling interface (use fire/smoke rectangle labels)
3. Upload images or import via server
4. Draw bounding boxes around fire and smoke
5. Export in YOLO format

**For large datasets (200+ images):** Use server-based import instead:
```
AutoLabel\run_auto_label.bat       ← Auto-label with pre-trained YOLO
AutoLabel\import_to_label_studio.bat  ← Import to Label Studio via HTTP
```
This avoids the `DATA_UPLOAD_MAX_NUMBER_FILES` Django limit.

**Time:** 30 min - 2 hours depending on dataset size

---

## Step 3 - `3.dataset.bat` (Prepare Dataset)

Splits `raw_dataset/` into train (70%) / val (20%) / test (10%).

**Before running:** Ensure your labeled .txt files are alongside images in `raw_dataset/`.

**Time:** ~1-5 minutes

---

## Step 4 - `4.train.bat` (Train Model)

Trains YOLO11m optimized for aerial fire/smoke detection.

**Default configuration (optimized for drone @ 100m):**
- Image Size: 1280px (preserves small objects)
- Batch Size: 4 (high res = more VRAM)
- Epochs: 800 with early stopping at 120
- Augmentation: 180° rotation, vertical flip, copy-paste (aerial-optimized)

**Time:** 2-8 hours depending on dataset and GPU

---

## After Training

### Standard inference
```batch
python scripts\inference.py --model runs\fire_smoke_xxx\weights\best.pt --source image.jpg
```

### SAHI sliced inference (best for small objects from drone)
```batch
pip install sahi
python scripts\inference_sahi.py --model runs\fire_smoke_xxx\weights\best.pt --source image.jpg
```

---

## Optional: Auto-Label with Pre-trained YOLO

Skip manual labeling for initial labels:

```batch
AutoLabel\run_auto_label.bat          ← Auto-label images
AutoLabel\verify_labels.bat           ← Verify quality
AutoLabel\import_to_label_studio.bat  ← Review/edit in Label Studio
```

See `AutoLabel/README.md` for details.

---

## Troubleshooting

### "DATA_UPLOAD_MAX_NUMBER_FILES" error in Label Studio
Use the server-based import workflow instead of direct file upload:
```
AutoLabel\import_to_label_studio.bat
```

### "There was an issue loading URL" in Label Studio
The image server is not running or started from the wrong directory.
Make sure the image server window is open and serving from the correct project folder.

### CUDA Out of Memory
Reduce batch size: use `--batch 2` or `--batch 1` at imgsz=1280.

### Training NaN / Loss Explodes
Reduce learning rate: use `--lr0 0.0005`.

---

## Documentation

| Topic | File |
|-------|------|
| Project overview | `README.md` |
| Installation details | `docs/INSTALLATION.md` |
| Dataset preparation | `docs/DATASET_GUIDE.md` |
| Training guide | `docs/TRAINING_GUIDE.md` |
| Command reference | `docs/QUICK_REFERENCE.md` |
| Labeling issues | `docs/LABELING_TROUBLESHOOTING.md` |
| Auto-labeling | `AutoLabel/README.md` |

---

---

## 💡 Quick Tips

✅ **Best practice:** Run steps in order (1 → 2 → 3 → 4)

✅ **Before each step:** Make sure the previous step completed successfully

✅ **Dataset quality:** Good labels = better model accuracy

✅ **Training:** Can take 2-8 hours depending on dataset size

✅ **GPU:** Check with `nvidia-smi` command to verify GPU is being used

---

## 🎯 After Training

Once training completes (Step 4):

1. **Find your model:** `runs/train_001/weights/best.pt`
2. **Validate it:** `python scripts\validate_model.py --model runs/train_001/weights/best.pt`
3. **Use it:** `python scripts\inference.py --model runs/train_001/weights/best.pt --source image.jpg`
4. **Export it:** `python scripts\export_model.py --model runs/train_001/weights/best.pt`

---

## 📊 Expected Timeline

| Step | Duration | Activity |
|------|----------|----------|
| 1️⃣ Setup | ~15 min | Install Python, PyTorch, dependencies |
| 2️⃣ Dataset | ~5 min | Split and organize images |
| 3️⃣ Labeling | 30 min - 2 hrs | Draw bounding boxes |
| 4️⃣ Training | 2-8 hrs | Train model |
| **TOTAL** | **~3-11 hrs** | Complete pipeline |

---

## 🚀 You're Ready!

Just follow these 4 simple steps and you'll have a trained YOLO model!

**Start with:** Double-click `1.install.bat` 🎯

---

**Need help?** Check `docs/INSTALLATION.md` or `docs/QUICK_REFERENCE.md`
