# YoloLabel - Auto-Labeling with Pre-trained YOLO

Automatically generate YOLO format labels using a pre-trained YOLO11 model, then verify and refine them in Label Studio.

## 🚀 Features

- **Pre-trained YOLO Detection** - Use YOLO11 to automatically detect objects
- **Multiple Model Sizes** - Choose from yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- **Adjustable Confidence** - Fine-tune detection threshold
- **Batch Processing** - Process entire directories of images
- **Visualizations** - Generate sample images with bounding boxes
- **Label Studio Integration** - Import auto-labels for manual verification
- **Fast & Reliable** - GPU accelerated if available

## 📁 Workflow

```
Raw Images (1400+)
    ↓
[1] Auto-Label with Pre-trained YOLO
    ↓
Auto-Generated Labels (YOLO format)
    ↓
[2] Review Visualizations (optional)
    ↓
[3] Verify in Label Studio
    ↓
[4] Manual Corrections
    ↓
Final Verified Labels
    ↓
[5] Start Training
```

---

## ⚡ Quick Start

### Option 1: Easy (Recommended)

Double-click the batch file:
```
YoloLabel/auto_label.bat
```

Or PowerShell:
```
YoloLabel/auto_label.ps1
```

### Option 2: Command Line

Activate environment:
```bash
.venv\Scripts\activate.bat
```

Run auto-labeling:
```bash
python YoloLabel/auto_label.py --images dataset/images/train --output dataset/labels/train --visualize
```

---

## 📊 Input & Output

### Input
```
dataset/
└── images/train/
    ├── fire_001.jpg      (1400+ images)
    ├── fire_002.jpg
    ├── fire_003.png
    └── ...
```

### Output
```
dataset/
├── labels/train/
│   ├── fire_001.txt      (YOLO format)
│   ├── fire_002.txt
│   ├── fire_003.txt
│   └── ...
└── visualizations/
    ├── labeled_fire_001.jpg   (with bounding boxes)
    ├── labeled_fire_002.jpg
    └── ... (first 50 images)
```

---

## 📝 Label Format

Each `.txt` file contains one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `fire_001.txt`
```
0 0.523 0.456 0.234 0.567
0 0.712 0.234 0.156 0.345
```

- `0` = class ID (adjust based on your dataset)
- `0.523` = normalized X center (0-1)
- `0.456` = normalized Y center (0-1)
- `0.234` = normalized width (0-1)
- `0.567` = normalized height (0-1)

---

## 🎛️ Command Line Options

```bash
python auto_label.py --help
```

### Basic Usage

```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train
```

### With Custom Settings

```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --model yolo11l.pt \
    --conf 0.6 \
    --iou 0.5 \
    --include-confidence
```

### Generate Visualizations

```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --visualize \
    --viz-limit 100
```

---

## 🔧 Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n.pt | ~2.6M | ⚡ Fastest | Lower | Mobile/Edge |
| yolo11s.pt | ~6.4M | ⚡⚡ Fast | Medium | Real-time |
| yolo11m.pt | ~14.4M | ⚡⚡⚡ Medium | High | **Default** |
| yolo11l.pt | ~25.9M | ⚡⚡ Slower | ⬆️ Higher | Accuracy |
| yolo11x.pt | ~56.9M | 🐢 Slowest | ⬆️⬆️ Highest | Maximum |

**Recommendation:** Use `yolo11m.pt` for balanced speed/accuracy

---

## 💡 Confidence Threshold

```
Low Threshold (0.3)  → More detections, more false positives
Medium Threshold (0.5) → Balanced (default)
High Threshold (0.8) → Fewer detections, more accurate
```

**Adjust based on:**
- Low: Many small/distant objects
- High: Only clear, obvious objects
- Default (0.5): Good for most cases

---

## 📊 Example Output

```
Auto-labeling 1400 images using yolo11m.pt
Confidence threshold: 0.5
IOU threshold: 0.5

Processing: [████████████████████] 100%

✅ Auto-labeling Complete!
Images processed: 1400
Images with detections: 1320
Total objects detected: 2847
Average objects per image: 2.0
Labels saved to: dataset/labels/train/
```

---

## 🔍 Review Process

### Step 1: Check Visualizations

```
dataset/visualizations/labeled_fire_*.jpg
```

Visual check of auto-generated bounding boxes:
- ✅ Good detections: Move on
- ❌ False positives: Need to correct
- ⚠️ Missed objects: Need to add

### Step 2: Verify in Label Studio

```bash
3.label.bat
```

1. **Import** auto-generated labels
2. **Review** each image
3. **Correct** false positives/negatives
4. **Export** corrected labels

### Step 3: Compare Original vs Corrected

```bash
python compare_labels.py \
    --original dataset/labels/train \
    --corrected dataset/labels/train_corrected
```

---

## ⚠️ Handling False Positives

If auto-labeling generates false positives:

### Option 1: Increase Confidence Threshold

```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --conf 0.7  # Increase from 0.5 to 0.7
```

Higher threshold = fewer but more accurate detections

### Option 2: Manual Correction in Label Studio

1. Review in Label Studio
2. Delete wrong detections
3. Export corrected labels
4. Use for training

### Option 3: Combine with Other Model

```bash
# Try larger model for better accuracy
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --model yolo11l.pt
```

---

## 🚀 Integration with Training

After verification:

```bash
# 1. Auto-label
python auto_label.py --images dataset/images/train --output dataset/labels/train

# 2. Verify in Label Studio
3.label.bat
# ... review and correct labels ...

# 3. Create dataset config
python scripts/label_images.py --config --num-classes 1

# 4. Start training
4.train.bat
```

---

## 📈 Performance Tips

### Speed Up Processing
- Use smaller model: `--model yolo11s.pt`
- Reduce image resolution (not recommended)
- Use GPU with CUDA support

### Improve Accuracy
- Use larger model: `--model yolo11l.pt`
- Adjust confidence: `--conf 0.4` to `--conf 0.6`
- Increase IOU threshold: `--iou 0.6`

### Monitor GPU Usage
```bash
# Windows (NVIDIA)
nvidia-smi -l 1
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"

**Solution:** Use smaller model or reduce batch size
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --model yolo11s.pt
```

### Error: "No images found"

**Solution:** Check image directory and format
```bash
# Check if images exist
dir dataset\images\train\*.jpg
```

### Labels not generated

**Solution:** Check confidence threshold
```bash
# Lower threshold to detect more objects
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --conf 0.3
```

### Models download very slowly

**Solution:** Download manually
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

---

## 📚 YOLO Format Reference

### Single object:
```
0 0.5 0.5 0.3 0.4
```
- Class: 0
- Center: (50%, 50%)
- Size: 30% × 40%

### Multiple objects:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.15 0.2
0 0.8 0.3 0.2 0.25
```

### Conversion formula:
```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width = (x2 - x1) / image_width
height = (y2 - y1) / image_height
```

---

## 🔗 Resources

- **YOLO Documentation:** https://docs.ultralytics.com/
- **Label Studio:** https://labelstud.io/
- **YOLO Format:** https://docs.ultralytics.com/datasets/detect/

---

## ✅ Workflow Checklist

- [ ] 1. Run auto-label: `YoloLabel/auto_label.bat`
- [ ] 2. Review visualizations: `dataset/visualizations/`
- [ ] 3. Check label files: `dataset/labels/train/`
- [ ] 4. Open Label Studio: `3.label.bat`
- [ ] 5. Import and verify labels
- [ ] 6. Export corrected labels
- [ ] 7. Create dataset config: `scripts/label_images.py --config`
- [ ] 8. Start training: `4.train.bat`

---

**Ready to auto-label? Run:** `YoloLabel/auto_label.bat` 🚀
