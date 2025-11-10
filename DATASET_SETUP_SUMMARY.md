# Dataset Management - Setup Summary

✅ **Complete dataset management system has been added to your YOLO project!**

## 📦 What Was Added

### 1. **Raw Dataset Folder**
```
raw_dataset/  ← Place your images here
```

### 2. **Organized Dataset Structure**
```
dataset/
├── images/
│   ├── train/     (70% of your images)
│   ├── val/       (20% of your images)
│   └── test/      (10% of your images)
├── labels/
│   ├── train/     (YOLO format annotations)
│   ├── val/
│   └── test/
└── data.yaml      (Dataset configuration)
```

### 3. **Automated Dataset Splitting Tool**
📄 **`scripts/split_dataset.py`**
- Automatically splits images into train/val/test
- Default ratio: 70/20/10 (customizable)
- Supports: JPG, PNG, BMP, GIF, WebP
- Command:
  ```bash
  python scripts/split_dataset.py
  python scripts/split_dataset.py --train 0.8 --val 0.15 --test 0.05
  ```

### 4. **Image Labeling Tool Launcher**
📄 **`scripts/label_images.py`**
- Support for 5+ annotation tools
- Tools available:
  - **LabelImg** - Desktop GUI (Recommended) 🌟
  - **CVAT** - Web-based (Best for teams)
  - **Label Studio** - Web-based (Easy setup)
  - **OpenLabeling** - Fast desktop
  - **Roboflow** - Cloud AI-assisted
- Commands:
  ```bash
  python scripts/label_images.py                              # Interactive menu
  python scripts/label_images.py --tool labelimg            # Launch LabelImg
  python scripts/label_images.py --config --num-classes 3   # Create config
  ```

### 5. **Dataset Manager (GUI Helper)**
🪟 **`manage_dataset.bat`** (Windows Batch)
- Interactive menu for all dataset operations
- One-click access to tools
- Dataset information display
- How to use:
  ```batch
  .\manage_dataset.bat
  ```

🐚 **`manage_dataset.ps1`** (PowerShell)
- Same features as batch version
- More colorized output
- How to use:
  ```powershell
  .\manage_dataset.ps1
  ```

### 6. **Comprehensive Dataset Guide**
📖 **`DATASET_GUIDE.md`**
- Step-by-step instructions
- Tool installation guides
- YOLO label format explanation
- Troubleshooting section
- Best practices

### 7. **Example Script**
📝 **`examples/dataset_example.py`**
- Demonstrates the complete workflow
- Can be used as a template

---

## 🚀 Quick Start (3 Steps)

### Step 1: Add Images
```bash
# Copy your images to raw_dataset folder
cp your_images/* raw_dataset/
```

### Step 2: Split Dataset
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Option A: Interactive menu
.\manage_dataset.bat

# Option B: Command line
python scripts/split_dataset.py
```
✅ Images automatically split into train/val/test folders

### Step 3: Label Images
```bash
# Launch annotation tool
python scripts/label_images.py

# Select your preferred tool from the menu
```
✅ Images labeled in YOLO format automatically

---

## 📋 Files Added/Modified

### New Files Created:
- ✅ `scripts/split_dataset.py` - Dataset splitting tool (500+ lines)
- ✅ `scripts/label_images.py` - Annotation tool launcher (600+ lines)
- ✅ `manage_dataset.bat` - Windows batch helper
- ✅ `manage_dataset.ps1` - PowerShell helper
- ✅ `DATASET_GUIDE.md` - Comprehensive guide (400+ lines)
- ✅ `examples/dataset_example.py` - Example workflow
- ✅ `raw_dataset/` - Folder for raw images (empty, ready for your data)
- ✅ `dataset/images/train/` - Training images folder
- ✅ `dataset/images/val/` - Validation images folder
- ✅ `dataset/images/test/` - Test images folder
- ✅ `dataset/labels/train/` - Training labels folder
- ✅ `dataset/labels/val/` - Validation labels folder
- ✅ `dataset/labels/test/` - Test labels folder

### Modified Files:
- ✅ `README.md` - Updated with dataset management section
- ✅ `install.bat` - Updated to use Python 3.10

---

## 🎯 Features

### ✨ Smart Dataset Splitting
- Automatic train/val/test split with configurable ratios
- Randomization with fixed seed for reproducibility
- Move or copy files (copy is safer, move saves disk space)
- Supported formats: JPG, PNG, BMP, GIF, WebP

### 🏷️ Multiple Annotation Tools
1. **LabelImg** (Recommended for beginners)
   - Fast, offline desktop tool
   - Native bounding box annotation
   - Automatic YOLO format export

2. **CVAT** (Best for teams)
   - Professional platform
   - Collaboration features
   - AI-assisted annotation

3. **Label Studio** (Easy web-based)
   - Minimal setup
   - Web interface
   - Multiple export formats

4. **OpenLabeling** (Fastest)
   - Ultra-fast annotation
   - Keyboard shortcuts
   - Lightweight

5. **Roboflow** (Cloud AI-assisted)
   - AI auto-annotation
   - No installation needed
   - Free tier available

### 🛠️ Helper Utilities
- Interactive dataset manager (batch/PowerShell)
- Automatic configuration file generation
- Dataset statistics viewer
- Tool installation helper

---

## 📊 Dataset Format

### Directory Structure
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── data.yaml
```

### YOLO Label Format
Each image needs a corresponding `.txt` file:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id`: 0-indexed class number
- `x_center, y_center`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

Most annotation tools handle this automatically!

---

## 📖 Next Steps

### 1. Prepare Your Dataset
```bash
# Move images to raw_dataset
cp your_images/* raw_dataset/

# Split automatically
python scripts/split_dataset.py
```

### 2. Label Training Images
```bash
# Launch annotation tool
python scripts/label_images.py

# Choose LabelImg, CVAT, Label Studio, OpenLabeling, or Roboflow
```

### 3. Create Dataset Configuration
```bash
python scripts/label_images.py --config --num-classes 3
```

### 4. Start Training
```bash
python train_optimized.py --data dataset/data.yaml --epochs 100 --batch 32
```

---

## 🔧 Advanced Usage

### Custom Split Ratios
```bash
python scripts/split_dataset.py --train 0.8 --val 0.1 --test 0.1
```

### Move Instead of Copy (Saves Disk Space)
```bash
python scripts/split_dataset.py --move
```

### Specific Annotation Tool
```bash
python scripts/label_images.py --tool labelimg
python scripts/label_images.py --tool cvat
python scripts/label_images.py --tool label-studio
```

---

## 💡 Tips & Best Practices

✅ **Do:**
- Use high-quality images
- Include diverse angles and lighting
- Collect 100+ images per class minimum
- Balance classes (similar count per class)
- Review labels for accuracy

❌ **Don't:**
- Use very low-resolution images
- Have huge class imbalance
- Mix different camera sources carelessly
- Skip difficult examples

---

## 🆘 Troubleshooting

### Images not splitting?
```bash
# Check images in raw_dataset
dir raw_dataset/

# Check supported formats: JPG, PNG, BMP, GIF, WebP
```

### Annotation tool won't launch?
```bash
# Install manually
pip install labelimg
labelimg dataset/images/train yolo
```

### Configuration file errors?
```bash
# Regenerate config
python scripts/label_images.py --config --num-classes 3

# Edit data.yaml and update class names manually
```

---

## 📚 Documentation

- **Main Guide**: [README.md](README.md)
- **Dataset Guide**: [DATASET_GUIDE.md](DATASET_GUIDE.md)
- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## 🎓 Learn More

- [YOLO Official Documentation](https://docs.ultralytics.com/)
- [LabelImg Repository](https://github.com/heartexlabs/labelImg)
- [CVAT Documentation](https://opencv.github.io/cvat/)
- [Label Studio Docs](https://labelstud.io/)
- [Roboflow Platform](https://roboflow.com/)

---

**Your dataset management system is ready! 🎉**

Start with:
```bash
.\manage_dataset.bat
```

Happy training! 🚀
