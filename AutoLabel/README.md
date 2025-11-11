# Auto-Labeling Tool

Complete workflow for automatically labeling images with YOLO and reviewing labels in Label Studio.

## Quick Start

Run ONE batch file to do everything:

```batch
AutoLabel\import_to_label_studio.bat
```

It automatically:
1. Imports your auto-labeled data
2. Starts image server (in separate window)
3. Launches Label Studio (in browser)
4. You edit labels
5. Done!

## The 3 Batch Files

**1. Auto-Label Images**
```batch
AutoLabel\run_auto_label.bat
```
- Select YOLO model
- Select image folder
- Select output folder
- Labels generated in YOLO format

**2. Verify Labels**
```batch
AutoLabel\verify_labels.bat
```
- View images with bounding boxes
- Check quality before Label Studio review

**3. Review & Edit in Label Studio** ⭐ (Recommended)
```batch
AutoLabel\import_to_label_studio.bat
```
- Auto-imports data
- Auto-starts image server
- Auto-launches Label Studio
- Edit labels in web browser
- Export corrected labels

## Setup (First Time Only)

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate.ps1

# Install dependencies
cd AutoLabel
pip install -r requirements.txt
```

## How It Works

**Single Batch File Workflow:**
```
import_to_label_studio.bat
    ↓
Import auto-labeled data
    ↓
Start image server (localhost:8000)
    ↓
Launch Label Studio (localhost:8080)
    ↓
Edit labels in browser
    ↓
Export & done
```

## Output Format

YOLO format labels:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.15 0.2
```

## Label Studio Editing

Inside Label Studio browser:
- **Move box** - Click inside, drag, release
- **Resize box** - Drag corner to adjust
- **Delete box** - Right-click, select Delete
- **Add box** - Rectangle tool, draw, select class
- **Change label** - Click box, select new class
- **Submit** - When done with image
- **Export** - Click Export, save file

## Supported Models

- `yolo11n.pt` (Nano - fastest)
- `yolo11s.pt` (Small)
- `yolo11m.pt` (Medium)
- `yolo11l.pt` (Large)
- `yolo11x.pt` (Extra Large - most accurate)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Virtual environment error | `python -m venv .venv` |
| Import fails | Check image & label folders exist |
| Images won't load in Label Studio | Keep image server window open |
| Port 8000/8080 in use | Close other applications |
| Batch file won't run | Check Windows Defender/antivirus not blocking |

## Next Steps

1. Auto-label: `run_auto_label.bat`
2. Verify: `verify_labels.bat`
3. Review & edit: `import_to_label_studio.bat`
4. Export corrected labels
5. Use for training

---

**Start with:** `import_to_label_studio.bat`  

## Workflow

### Step 1: Auto-Label Images

```batch
AutoLabel\run_auto_label.bat
```

- Select YOLO model
- Select image folder
- Select output folder
- Labels generated in YOLO format

### Step 2: Verify Labels (Optional)

```batch
AutoLabel\verify_labels.bat
```

- View images with bounding boxes
- Quick quality check
- Navigate with Previous/Next

### Step 3: Import to Label Studio

```batch
AutoLabel\import_to_label_studio.bat
```

Follow the prompts:
- Select image folder
- Select label folder
- Enter project name
- Opens Label Studio interface
- Edit labels in web browser

## Output Format

Labels are generated in **YOLO format**:

```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.15 0.2
```

- `class_id`: Class index (0-indexed)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

## Configuration

### Model Selection

Supports all pretrained YOLO models:
- `yolo11n.pt` (Nano - fast, small)
- `yolo11s.pt` (Small - balanced)
- `yolo11m.pt` (Medium - accurate)
- `yolo11l.pt` (Large - very accurate)
- `yolo11x.pt` (XLarge - best accuracy)

### Threshold Configuration

Adjust during auto-labeling:
- **Confidence Threshold**: Minimum detection confidence (default: 0.25)
- **IOU Threshold**: Intersection over Union for NMS (default: 0.45)

## GPU Acceleration

The tool automatically detects and uses:
- NVIDIA CUDA 12.1+ with GPU
- CPU fallback if GPU unavailable

Check GPU usage:
```powershell
nvidia-smi  # NVIDIA GPUs
```

## Installation

1. **Virtual Environment** (if not already done)
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate.ps1
   ```

2. **Install Dependencies**
   ```powershell
   cd AutoLabel
   pip install -r requirements.txt
   ```

3. **Run the Tool**
   ```batch
   run_auto_label.bat
   ```

## Troubleshooting

**Virtual environment not found**
- Run: `python -m venv .venv`

**GPU not detected**
- Check: `nvidia-smi`
- Tool will use CPU as fallback

**Label Studio won't import**
- Keep image server running: `scripts\start_image_server.bat`
- Check port 8000 is available

**Images won't display**
- Check browser console (F12)
- Verify image server is running

## Documentation

- `QUICKSTART.md` - 30-second start guide
- `COMPLETE_GUIDE.md` - Full tutorials
- `LABEL_STUDIO_QUICK_START.md` - Label Studio setup
- `FILE_STRUCTURE.md` - Detailed project layout

## Helper Tools

Located in `scripts/`:

- `fix_tasks_urls.py` - Fix Label Studio URLs if needed
- `start_image_server.bat` - Start image server manually
- `test_venv.bat` - Test virtual environment

## Next Steps

1. Run `run_auto_label.bat` to auto-label images
2. Run `verify_labels.bat` to check quality
3. Run `import_to_label_studio.bat` to review/edit in Label Studio
4. Export corrected labels for training

---

**Questions?** See the documentation files or check `QUICKSTART.md`   - Navigate to the folder containing your images
   - Click "OK" to select

4. **Select Output Folder**
   - A folder browser window will open
   - Navigate to where you want to save the labels
   - The folder will be created if it doesn't exist
   - Click "OK" to select

5. **Configure Parameters**
   - A configuration dialog will appear
   - **Confidence Threshold** (0.0-1.0): Minimum confidence to create a detection (default: 0.25)
   - **IOU Threshold** (0.0-1.0): Non-maximum suppression threshold (default: 0.45)
   - Click "Start" to begin labeling or "Cancel" to exit

6. **Monitor Progress**
   - A progress bar shows the processing status
   - Detailed statistics are printed to the console

7. **Review Results**
   - Labels are saved as `.txt` files in the output folder
   - One label file per image with the same name as the image
   - You can now verify and edit with a label tool (like CVAT or LabelImg)

## Parameters Explained

### Confidence Threshold
- **Lower values** (e.g., 0.1): More detections, including weak ones
- **Higher values** (e.g., 0.5): Fewer, more confident detections
- **Default**: 0.25 (good balance)
- **Recommendation**: Start with 0.25 and adjust based on results

### IOU Threshold
- **Lower values** (e.g., 0.3): More aggressive duplicate removal
- **Higher values** (e.g., 0.7): Less aggressive duplicate removal
- **Default**: 0.45 (standard for most applications)
- **Recommendation**: Keep default unless you have overlapping objects

## Performance

With GPU acceleration (CUDA):
- **Images per second**: 10-50 depending on model size and GPU
- **RTX 5080**: ~100+ images/minute with YOLOv11n

With CPU:
- **Images per second**: 1-5 depending on processor
- **Much slower** - GPU is recommended

## Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.gif`
- `.tiff`

## Supported Models

All YOLOv8, YOLOv11 models and variants:
- `yolo11n.pt` (Nano - fastest)
- `yolo11s.pt` (Small)
- `yolo11m.pt` (Medium)
- `yolo11l.pt` (Large)
- `yolo11x.pt` (Extra Large - most accurate)

Or use any custom trained model you have.

## Next Steps: Verification

After auto-labeling, you can verify and correct labels using:

1. **CVAT** (Web-based)
   - Free, powerful, supports batch operations
   - Can load YOLO format labels directly

2. **LabelImg** (Desktop app)
   - Simple, lightweight
   - Can convert between formats

3. **Roboflow Annotate** (Web-based)
   - Good for collaborative editing

## Troubleshooting

### "Virtual environment not found"
```bash
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
```

### "No module named 'ultralytics'"
```bash
.venv\Scripts\activate.ps1
pip install ultralytics opencv-python pillow tqdm
```

### GUI windows not opening
- Try running from command line to see error messages
- Ensure tkinter is installed: `pip install tk`

### CUDA errors
```bash
# Reinstall PyTorch with CUDA support
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Out of memory errors
- Use a smaller model (yolo11n instead of yolo11x)
- Reduce batch processing or process fewer images at once

## Example Workflow

```bash
# 1. Place images in a folder
mkdir dataset/images/custom
# Copy your images here

# 2. Run auto-labeling
AutoLabel\run_auto_label.bat
# Select model, image folder, output folder

# 3. Verify labels using a label tool
# CVAT, LabelImg, or similar

# 4. Use labels for training
python scripts/train_optimized.py --data dataset/custom.yaml
```

## Advanced: Modifying the Script

To add more features, edit `AutoLabel\auto_label.py`:

- **Add command-line arguments** in the `main()` function
- **Change default thresholds** in the `show_config_dialog()` method
- **Add class name filtering** before creating label files
- **Export to different formats** (COCO JSON, Pascal VOC, etc.)

## Support

For issues with:
- **YOLO Model**: Check [Ultralytics Documentation](https://docs.ultralytics.com/)
- **CUDA/GPU**: Check NVIDIA driver version and CUDA toolkit
- **Python dependencies**: Review `requirements.txt` in project root

---

**Happy Labeling! 🎉**
