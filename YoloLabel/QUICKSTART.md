# YoloLabel - Quick Start Guide

## 5-Minute Setup

### Step 1: Run Auto-Labeling (2 minutes)

```bash
YoloLabel/auto_label.bat
```

**What it does:**
- Loads pre-trained YOLO11m model
- Processes all images in `dataset/images/train/`
- Generates labels in `dataset/labels/train/`
- Creates visualizations in `dataset/visualizations/`

### Step 2: Review Visualizations (2 minutes)

```
dataset/visualizations/
├── labeled_image_1.jpg
├── labeled_image_2.jpg
└── ... (first 50 images)
```

Check:
- ✅ Are bounding boxes correct?
- ⚠️ Are there false positives?
- ❓ Are objects missed?

### Step 3: Verify in Label Studio (1 minute)

```bash
3.label.bat
```

Then:
1. Import generated labels
2. Fix any errors
3. Export corrected labels

---

## Expected Output

### After Auto-Labeling:

```
Auto-labeling 1400 images using yolo11m.pt

Processing: [████████████████████] 100%

✅ Auto-labeling Complete!
Images processed: 1400
Images with detections: 1320
Total objects detected: 2847
Average objects per image: 2.0
Labels saved to: dataset/labels/train/
Visualizations saved to: dataset/visualizations/
```

### File Structure:

```
dataset/
├── images/train/
│   ├── fire_001.jpg
│   ├── fire_002.jpg
│   └── ...
├── labels/train/
│   ├── fire_001.txt    (NEW - YOLO format)
│   ├── fire_002.txt
│   └── ...
└── visualizations/
    ├── labeled_fire_001.jpg   (NEW - with boxes)
    ├── labeled_fire_002.jpg
    └── ...
```

---

## Label Format

Each `.txt` file contains one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `fire_001.txt`
```
0 0.523 0.456 0.234 0.567
0 0.712 0.234 0.156 0.345
```

---

## Common Scenarios

### ✅ Scenario 1: Labels Look Good

1. Skip visualization review
2. Directly use in training:
   ```bash
   4.train.bat
   ```

### ⚠️ Scenario 2: Many False Positives

**Solution:** Increase confidence threshold

```bash
python YoloLabel/auto_label.py ^
    --images dataset/images/train ^
    --output dataset/labels/train ^
    --conf 0.7
```

Higher confidence = fewer but more accurate detections

### ❌ Scenario 3: Missing Objects

**Solution:** Lower confidence threshold

```bash
python YoloLabel/auto_label.py ^
    --images dataset/images/train ^
    --output dataset/labels/train ^
    --conf 0.3
```

Lower confidence = more detections but more false positives

### 🚀 Scenario 4: Need Higher Accuracy

**Solution:** Use larger model

```bash
python YoloLabel/auto_label.py ^
    --images dataset/images/train ^
    --output dataset/labels/train ^
    --model yolo11l.pt
```

Larger models are slower but more accurate

---

## Troubleshooting

### "No images found"

Check that images exist:
```bash
dir dataset\images\train\*.jpg
```

If not, run data preparation first:
```bash
2.dataset.bat
```

### "CUDA out of memory"

Use smaller model:
```bash
python YoloLabel/auto_label.py ^
    --images dataset/images/train ^
    --output dataset/labels/train ^
    --model yolo11s.pt
```

### Labels are very bad

Try different confidence:
```bash
# Try 0.3 (more detections)
python YoloLabel/auto_label.py ^
    --images dataset/images/train ^
    --output dataset/labels/train ^
    --conf 0.3
```

---

## Command Line Options

### Basic
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train
```

### With Visualizations
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --visualize
```

### Custom Confidence
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --conf 0.6
```

### Larger Model (Better Accuracy)
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --model yolo11l.pt
```

### Include Confidence Scores
```bash
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --include-confidence
```

---

## Model Comparison

| Model | Speed | Accuracy | Memory | For |
|-------|-------|----------|--------|-----|
| yolo11n | ⚡⚡⚡ | Medium | Low | Edge devices |
| yolo11s | ⚡⚡ | Good | Low | Mobile |
| **yolo11m** | ⚡ | Great | Medium | **Default** |
| yolo11l | 🐢 | Better | High | Better accuracy |
| yolo11x | 🐢🐢 | Best | Very High | Maximum accuracy |

**Default:** yolo11m (good balance)

---

## Confidence Thresholds

```
0.3: Very lenient - many detections, many false positives
0.5: Balanced - default, moderate detections
0.7: Strict - fewer detections, more accurate
0.9: Very strict - only very confident detections
```

**Recommended:** Start at 0.5, adjust based on results

---

## Next Steps After Auto-Labeling

### Option A: Use Directly (Fast)
```bash
4.train.bat
```

Best if auto-labels are already good

### Option B: Manual Verification (Recommended)
```bash
3.label.bat
```

1. Open Label Studio
2. Import auto-generated labels
3. Fix errors
4. Export corrected labels
5. Train with corrected labels

### Option C: Compare Changes (Advanced)
```bash
python YoloLabel/compare_labels.py \
    --original dataset/labels/train \
    --corrected dataset/labels/train_corrected \
    --details
```

See what changed between original and corrected

---

## Tips & Tricks

### Visualize Sample Labels
```bash
# See first 50 images with bounding boxes
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --visualize \
    --viz-limit 50
```

### Include Confidence in Labels
```bash
# Useful for filtering low-confidence detections
python auto_label.py \
    --images dataset/images/train \
    --output dataset/labels/train \
    --include-confidence
```

### Batch Process Multiple Folders
```bash
# Process train
python auto_label.py --images dataset/images/train --output dataset/labels/train

# Process val
python auto_label.py --images dataset/images/val --output dataset/labels/val

# Process test
python auto_label.py --images dataset/images/test --output dataset/labels/test
```

---

## Recommended Workflow

1. **Auto-label** with default settings
   ```bash
   YoloLabel/auto_label.bat
   ```

2. **Review visualizations** (5 min)
   - Check `dataset/visualizations/`
   - Look for obvious errors

3. **Decide next step:**
   - Good results → Go to step 4
   - Bad results → Adjust confidence and repeat step 1

4. **Verify in Label Studio** (optional)
   ```bash
   3.label.bat
   ```

5. **Train**
   ```bash
   4.train.bat
   ```

---

## Files & Directories

```
YoloLabel/
├── auto_label.py              # Main script
├── auto_label.bat             # Batch launcher
├── auto_label.ps1             # PowerShell launcher
├── compare_labels.py          # Compare original vs corrected
└── README.md                  # Full documentation
```

```
dataset/
├── images/train/              # Input images
├── labels/train/              # Output labels (NEW)
└── visualizations/            # Sample visualizations (NEW)
```

---

## Success Checklist

- [ ] Auto-labeling completes without errors
- [ ] Labels generated in `dataset/labels/train/`
- [ ] Visualizations look reasonable
- [ ] All `.txt` files exist for all images
- [ ] Label format is correct (5 values per line)
- [ ] Ready to verify in Label Studio or train

---

**Ready? Run:** `YoloLabel/auto_label.bat` 🚀
