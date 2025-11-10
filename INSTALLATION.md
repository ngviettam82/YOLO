# YOLO Training Installation Guide

Complete installation instructions for your RTX 5080 system.

---

## 🎯 Quick Installation (Recommended)

### Option 1: Automated Installation (Easiest)

```powershell
# Navigate to YOLO directory
cd C:\Users\ADMIN\Documents\Code\SimulationControl\YOLO

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
cd C:\Users\ADMIN\Documents\Code\SimulationControl\YOLO

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # PowerShell
# OR
venv\Scripts\activate.bat    # Command Prompt
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

Choose one of three options:

#### Option A: Standard Installation (Recommended)
```powershell
pip install -r requirements.txt
```
- All essential packages
- ONNX export support
- TensorBoard monitoring
- ~1-2 GB download

#### Option B: Minimal Installation (Fastest)
```powershell
pip install -r requirements-minimal.txt
```
- Only core packages
- No export/monitoring tools
- ~500 MB download

#### Option C: Full Installation (Everything)
```powershell
pip install -r requirements-full.txt
```
- All packages including optional
- Mobile deployment support
- Development tools
- ~3-4 GB download

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

## 🔧 Package Details

### Core Packages (Always Installed)

| Package | Purpose | Size |
|---------|---------|------|
| torch | Deep learning framework | ~2 GB |
| ultralytics | YOLO implementation | ~20 MB |
| opencv-python | Image/video processing | ~50 MB |
| numpy | Numerical computing | ~20 MB |
| PyYAML | Configuration files | ~1 MB |

### Optional Packages

| Package | Purpose | When to Install |
|---------|---------|-----------------|
| onnx | Model export | Deploy to production |
| onnxruntime-gpu | Fast inference | Production deployment |
| tensorboard | Training monitoring | Monitor training progress |
| albumentations | Advanced augmentation | Need more augmentation |
| wandb | Cloud logging | Track experiments online |
| tensorflow | TFLite export | Mobile deployment |

---

## 🎯 Installation by Use Case

### 1. Learning & Training Only
```powershell
pip install -r requirements-minimal.txt
```

### 2. Training & Basic Deployment
```powershell
pip install -r requirements.txt
```

### 3. Production & Mobile Deployment
```powershell
pip install -r requirements-full.txt
```

---

## 🐛 Troubleshooting

### Issue 1: PyTorch Installation Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution:**
```powershell
# Check Python version (must be 3.8-3.11)
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

3. **Check CUDA Installation:**
   - Install CUDA Toolkit 12.8: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Issue 3: Package Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solution:**
```powershell
# Remove virtual environment
deactivate
Remove-Item -Recurse -Force venv

# Start fresh
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install in correct order
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Issue 4: Slow Download Speed

**Solutions:**
1. **Use a mirror (China users):**
   ```powershell
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **Download large packages separately:**
   ```powershell
   # Download PyTorch first (large file)
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   
   # Then install others
   pip install -r requirements.txt
   ```

### Issue 5: Out of Disk Space

**Check space required:**
- Minimal: ~3 GB
- Standard: ~5 GB
- Full: ~8 GB
- Plus dataset space: 1-10 GB
- Plus training outputs: 1-5 GB

**Total recommended: 20 GB free**

---

## ✅ Verification Checklist

After installation, verify:

- [ ] Python 3.8-3.11 installed
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA 12.8 installed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU detected: RTX 5080
- [ ] Ultralytics YOLO installed
- [ ] OpenCV installed
- [ ] `check_setup.py` passes all checks

---

## 🔄 Updating Packages

### Update All Packages
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Update pip
python -m pip install --upgrade pip

# Update PyTorch
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Update other packages
pip install --upgrade -r requirements.txt
```

### Update Specific Package
```powershell
pip install --upgrade ultralytics
pip install --upgrade opencv-python
```

---

## 🗑️ Uninstallation

### Remove Virtual Environment
```powershell
# Deactivate if active
deactivate

# Delete virtual environment
Remove-Item -Recurse -Force venv
```

### Clean Installation
```powershell
# Remove virtual environment
Remove-Item -Recurse -Force venv

# Remove cache
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force *.pyc

# Start fresh installation
python -m venv venv
.\venv\Scripts\Activate.ps1
# ... follow installation steps
```

---

## 📚 Additional Resources

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **NVIDIA Drivers:** https://www.nvidia.com/download/index.aspx
- **Ultralytics Docs:** https://docs.ultralytics.com

---

## 🚀 Next Steps

After successful installation:

1. **Read Documentation:**
   - `SETUP_COMPLETE.md` - Complete workflow
   - `RTX5080_OPTIMIZED.md` - System-specific guide
   - `TRAINING_GUIDE.md` - Training instructions

2. **Prepare Dataset:**
   ```powershell
   python utils/dataset_utils.py validate --dataset dataset
   ```

3. **Start Training:**
   ```powershell
   python train_optimized.py --data dataset/your_data.yaml
   ```

---

**Installation complete! Ready to train YOLO models! 🎉**
