# Quick Start - Step-by-Step Guide

**Before you start:** Read `README.md` for project overview

## Follow These 4 Steps In Order

Simply double-click these files **one after another**:

```
Step 1: 1.install.bat          (Setup - ~10 minutes)
        ↓
Step 2: 2.dataset.bat          (Dataset prep - ~5 minutes)
        ↓
Step 3: 3.label.bat            (Labeling - ~30 min - 2 hours)
        ↓
Step 4: 4.train.bat            (Training - 2-8 hours)
```

---

## 📋 What Each Step Does

### Step 1️⃣ - `1.install.bat` (Setup Environment)
**Run this FIRST to set up your system**

✅ Checks for Python 3.10  
✅ Creates virtual environment  
✅ Installs PyTorch with CUDA 12.8  
✅ Installs all dependencies  
✅ Verifies everything works  

**Time:** ~10-15 minutes

---

### Step 2️⃣ - `2.dataset.bat` (Prepare Dataset)
**Run this AFTER Step 1**

✅ Counts images in `raw_dataset/` folder  
✅ Splits into train (70%) / val (20%) / test (10%)  
✅ Creates `dataset/` folder structure  
✅ Generates `data.yaml` config file  

**Before running:** Add your images to `raw_dataset/` folder

**Time:** ~1-5 minutes

---

### Step 3️⃣ - `3.label.bat` (Label Images)
**Run this AFTER Step 2**

✅ Launches LabelImg annotation tool  
✅ Opens training images for labeling  
✅ Saves annotations automatically  

**What to do:** Draw bounding boxes around objects and assign class names

**Time:** Depends on your dataset (usually 30 min - 2 hours)

---

### Step 4️⃣ - `4.train.bat` (Train Model)
**Run this AFTER Step 3**

✅ Verifies dataset is ready  
✅ Starts training with optimized settings  
✅ Saves best model to `runs/train_xxx/weights/best.pt`  
✅ Shows training progress in console  

**Settings:** 1000 epochs, batch size 64, image size 640px

**Time:** 2-8 hours depending on dataset size

---

## 📁 Folder Setup Before Running

### Create `raw_dataset/` Folder

Before running Step 2, create a folder called `raw_dataset/` and add your images:

```
YOLO/
└── raw_dataset/              ← Create this folder
    ├── image1.jpg            ← Add your images here
    ├── image2.jpg
    ├── image3.jpg
    └── ...
```

---

## ✅ Verification Checklist

- [ ] Python 3.10 installed and in PATH
- [ ] Created `raw_dataset/` folder  
- [ ] Added images to `raw_dataset/`
- [ ] Can see `1.install.bat` in file explorer
- [ ] Can see `2.dataset.bat` in file explorer
- [ ] Can see `3.label.bat` in file explorer
- [ ] Can see `4.train.bat` in file explorer

---

## 🆘 Troubleshooting

### Step 1 - Python 3.10 Not Found

**Error:** `ERROR: Python 3.10 not found!`

**Solution:**
1. Download Python 3.10: https://www.python.org/downloads/release/python-3100/
2. Run the installer
3. **IMPORTANT:** Check "Add Python to PATH" during installation
4. Restart your computer
5. Try Step 1 again

### Step 2 - No Images Found

**Error:** `ERROR: No images found in raw_dataset/`

**Solution:**
1. Create `raw_dataset/` folder in project root
2. Add your images (jpg, png, etc.) to this folder
3. Run Step 2 again

### Step 3 - LabelImg Won't Open

**Error:** Annotation tool doesn't launch

**Solution:**
1. Ensure Step 1 and Step 2 completed successfully
2. Check that images exist in `dataset/images/train/`
3. Try running Step 3 again

### Step 4 - CUDA Out of Memory

**Error:** Training crashes with memory error

**Solution:**
1. Reduce batch size in Step 4
2. Or reduce image size
3. Or use smaller model size

See `docs/TRAINING_GUIDE.md` for detailed options

---

## 📚 Detailed Documentation

For more information on any topic:

- **`README.md`** - Project overview & features
- **`docs/INSTALLATION.md`** - Installation troubleshooting
- **`docs/DATASET_GUIDE.md`** - Dataset preparation details
- **`docs/TRAINING_GUIDE.md`** - Training tips & advanced settings
- **`docs/QUICK_REFERENCE.md`** - All commands reference
- **`docs/RTX5080_OPTIMIZED.md`** - GPU optimization
- **`docs/LABELING_TROUBLESHOOTING.md`** - Labeling tool help

---

## 💡 Optional: Auto-Label Images (Skip Step 3)

Don't want to label manually? Use a pre-trained YOLO:

```batch
cd AutoLabel
run_auto_label.bat
```

**See `AutoLabel/QUICKSTART.md` for details.**

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
