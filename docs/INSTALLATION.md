````markdown
# YOLO Training Installation Guide

Complete installation instructions for your RTX 5080 system.

---

## 🎯 Quick Installation (Recommended)

### Option 1: Automated Installation (Easiest)

```powershell
# Navigate to YOLO directory
cd C:\Users\ADMIN\Documents\Code\YOLO

# Run automated installer (PowerShell)
.\install.ps1

# OR run batch file (CMD)
.\install.bat
```

**The script will:**
1. Create virtual environment
2. Install PyTorch with CUDA 12.8
3. Install all dependencies
4. Verify installation

**Time:** ~10-15 minutes (depends on internet speed)

---

## 📋 Manual Installation

### Step 1: Create Virtual Environment

```powershell
# Navigate to YOLO directory
cd C:\Users\ADMIN\Documents\Code\YOLO

# Create virtual environment
python3.10 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # PowerShell
# OR
.venv\Scripts\activate.bat    # Command Prompt
```

### Step 2: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA 12.8 (CRITICAL!)

```powershell
# For RTX 5080 with CUDA 12.8 (MUST USE THIS)
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**⚠️ Important:** This must be done BEFORE installing other packages!

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Verify Installation

```powershell
python check_setup.py
```

**Expected Output:**
```
✅ Python version is compatible
✅ pip: xx.x.x
✅ PyTorch version: 2.x.x
✅ CUDA available: 12.8
✅ GPU: NVIDIA GeForce RTX 5080
✅ VRAM: 16.0 GB
💡 Recommended batch size: 40
✅ Ultralytics version: x.x.x
✅ OpenCV version: x.x.x
✅ All requirements met! Ready to train.
```

---

## 🔧 Troubleshooting

### Issue 1: PyTorch Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution:**
```powershell
# Check Python version (must be 3.10)
python --version

# Update pip
python -m pip install --upgrade pip

# Try installing PyTorch again
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Issue 2: CUDA Not Detected

**Symptoms:**
```
CUDA available: False
```

**Solutions:**
1. **Check NVIDIA Driver:**
   ```powershell
   nvidia-smi
   ```
   - Should show RTX 5080
   - Driver version should be 560.0+
   - Update if needed: [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)

2. **Reinstall PyTorch with CUDA:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

---

## ✅ Next Steps

After successful installation:

1. **Read Documentation:**
   - `docs/TRAINING_GUIDE.md` - Training instructions
   - `docs/DATASET_GUIDE.md` - Dataset preparation

2. **Prepare Dataset:**
   ```powershell
   python scripts/split_dataset.py
   ```

3. **Start Training:**
   ```powershell
   python train_optimized.py --data dataset/data.yaml
   ```

---

**Installation complete! Ready to train YOLO models! 🎉**

````
