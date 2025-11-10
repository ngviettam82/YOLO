````markdown
# Installation Guide

Quick setup for your YOLO training environment.

---

## 🚀 Automated Installation (Recommended)

```powershell
cd C:\Users\ADMIN\Documents\Code\YOLO
.\install.ps1
```

**This will:**
1. Create virtual environment (`.venv`)
2. Install PyTorch with CUDA 12.8
3. Install all dependencies
4. Verify your setup

**Time:** ~10-15 minutes

---

## ⚙️ Manual Installation

### Step 1: Create Virtual Environment

```powershell
python3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
```

### Step 2: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA 12.8

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Verify Setup

```powershell
python check_setup.py
```

---

## 🆘 Troubleshooting

### PyTorch Installation Fails

```powershell
python --version                    # Check Python 3.10
python -m pip install --upgrade pip # Update pip
# Then retry step 3 above
```

### CUDA Not Detected

```powershell
nvidia-smi  # Check driver (need 560.0+)
```

If needed, [update NVIDIA drivers](https://www.nvidia.com/download/index.aspx)

Reinstall PyTorch:
```powershell
pip uninstall torch torchvision torchaudio
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## ✅ What's Next?

**Next:** Prepare your dataset with `docs/DATASET_GUIDE.md`

```powershell
python scripts/split_dataset.py
python scripts/label_images.py
```

````
