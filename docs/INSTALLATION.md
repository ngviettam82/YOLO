# Installation Guide

Quick setup for your YOLO training environment.

---

## 🚀 Automated Installation (Recommended)

### Option 1: Double-Click (Easiest) ⭐

**Step 1:** Simply **double-click** `1.install.bat` from the project root folder.

The script will:
1. Check for Python 3.10
2. Create virtual environment (`.venv`)
3. Install PyTorch with CUDA 12.8
4. Install all dependencies
5. Verify your setup

**Time:** ~10-15 minutes

### Option 2: Command Line

```batch
1.install.bat
```

Or from PowerShell:
```powershell
cd C:\Users\ADMIN\Documents\Code\YOLO
.\1.install.bat
```

---

## ⚙️ Manual Installation

### Step 1: Create Virtual Environment

```batch
python3.10 -m venv .venv
.venv\Scripts\activate.bat
```

### Step 2: Upgrade pip

```batch
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA 12.8

```batch
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install Dependencies

```batch
pip install -r requirements.txt
```

### Step 5: Verify Setup

```batch
python scripts\check_setup.py
```

---

## 🆘 Troubleshooting

### Python 3.10 Not Found

**Error:** `ERROR: Python 3.10 not found!`

**Solution:**
1. Download Python 3.10 from https://www.python.org/downloads/release/python-3100/
2. During installation, **check "Add Python to PATH"**
3. Restart your computer
4. Try `install.bat` again

Verify Python 3.10 is installed:
```batch
python3.10 --version
```

### PyTorch Installation Fails

**Error:** `ERROR: Failed to install PyTorch!`

**Solution:**
1. Ensure you have internet connection
2. Try manually:
```batch
pip uninstall torch torchvision torchaudio -y
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA Not Detected

**Error:** CUDA not available when running training

**Solution:**
1. Check NVIDIA driver version:
```batch
nvidia-smi
```

2. You need driver version 560.0 or higher
3. Update from: https://www.nvidia.com/download/index.aspx

### Virtual Environment Already Exists

**Prompt:** `Do you want to recreate it? (y/N):`

- Type `y` to delete and recreate (fresh install)
- Type `n` to keep existing (faster, reuses existing packages)

---

## ✅ Next Steps

After installation completes, you're ready to:

1. **Prepare Dataset** (Step 2)
   ```batch
   2.dataset.bat
   ```

2. **Label Images** (Step 3)
   ```batch
   3.label.bat
   ```

3. **Train Model** (Step 4)
   ```batch
   4.train.bat
   ```

See `docs/DATASET_GUIDE.md` for detailed dataset preparation instructions.
