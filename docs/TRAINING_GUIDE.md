# Training Guide

Train your YOLO model with optimized settings for RTX 5080.

---

## 📋 Overview

This guide covers:
1. Starting training
2. Understanding training output
3. Validating your model
4. Exporting for deployment

---

## 🚀 Step 1: Start Training

### Option 1: Double-Click (Easiest) ⭐

**Step 4:** Simply **double-click** `4.train.bat` in the project root.

The script will:
1. Verify dataset exists
2. Activate virtual environment
3. Start training with optimized settings:
   - Model: yolo11m.pt
   - Epochs: 1000
   - Batch: 64
   - Image size: 640px
   - Learning rate: auto-optimized

**Time:** 2-8 hours depending on dataset size

### Option 2: Command Line

```batch
4.train.bat
```

Or manually:
```batch
.venv\Scripts\activate.bat
python scripts\train_optimized.py --data dataset/data.yaml --model yolo11m.pt --epochs 1000 --batch 64 --imgsz 640
```

### Custom Training Parameters

```batch
python scripts\train_optimized.py ^
  --data dataset/data.yaml ^
  --model yolo11m.pt ^
  --epochs 500 ^
  --batch 64 ^
  --imgsz 640 ^
  --patience 150 ^
  --device 0
```

**Common parameters:**
- `--epochs`: Number of epochs (default: 1000)
- `--batch`: Batch size (default: 64, max for RTX 5080)
- `--imgsz`: Image size in pixels (default: 640, options: 416, 512, 640, 768, 1024)
- `--patience`: Early stopping patience (default: 150)
- `--device`: GPU device ID (default: 0, use -1 for CPU)
- `--resume`: Resume from last checkpoint (automatic)

---

## 📊 Understanding Training Output

### Console Output

```
Epoch   1/1000: 100%|████████| 10/10 [00:15<00:00,  0.67s/batch]
        box_loss: 1.234
        cls_loss: 0.567
        dfl_loss: 0.890
        val loss: 2.691
```

**Key metrics:**
- `box_loss`: Bounding box regression loss (lower = better)
- `cls_loss`: Classification loss (lower = better)
- `dfl_loss`: Distribution focal loss (lower = better)
- `val loss`: Validation loss (lower = better)

### Training Folder Structure

```
runs/
└── train_<number>/
    ├── weights/
    │   ├── last.pt          # Last checkpoint
    │   └── best.pt          # Best model (use this)
    ├── plots/
    │   ├── results.png      # Training plots
    │   ├── confusion_matrix.png
    │   └── ...
    └── results.csv          # Training metrics
```

---

## 🎯 Step 2: Validate Model

After training completes, validate your model:

```batch
python scripts\validate_model.py --model runs/train_001/weights/best.pt --data dataset/data.yaml
```

**Output metrics:**
- `Precision`: Correctness of positive predictions
- `Recall`: Ability to find all positives
- `mAP@0.5`: Mean average precision at 0.5 IoU
- `mAP@0.5:0.95`: Mean average precision at 0.5-0.95 IoU

### Interpreting Results

| Metric | Good Range | Meaning |
|--------|-----------|---------|
| Precision | > 0.9 | Few false positives |
| Recall | > 0.85 | Finds most objects |
| mAP@0.5 | > 0.8 | Good detection accuracy |
| mAP@0.5:0.95 | > 0.6 | High quality detections |

---

## 🔍 Step 3: Run Inference

Test your model on new images:

```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source image.jpg
```

**Options:**
```batch
python scripts\inference.py ^
  --model runs/train_001/weights/best.pt ^
  --source image.jpg ^
  --conf 0.5 ^
  --iou 0.5
```

- `--model`: Path to model weights
- `--source`: Image/video/folder path
- `--conf`: Confidence threshold (0-1, default: 0.5)
- `--iou`: IoU threshold (0-1, default: 0.5)

### Batch Inference

Predict on multiple images:
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source dataset/images/test/
```

### Video Inference

Predict on video:
```batch
python scripts\inference.py --model runs/train_001/weights/best.pt --source video.mp4
```

---

## 📦 Step 4: Export Model

Export your trained model for deployment:

### Export to ONNX (Recommended for most uses)

```batch
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats onnx
```

**Output:** `runs/train_001/weights/best.onnx`

### Export to TensorRT (Fastest on NVIDIA GPUs)

```batch
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats engine
```

**Output:** `runs/train_001/weights/best.engine`

### Export Multiple Formats

```batch
python scripts\export_model.py --model runs/train_001/weights/best.pt --formats onnx engine pb
```

**Supported formats:**
- `pt` - PyTorch (native)
- `onnx` - ONNX format (cross-platform)
- `engine` - TensorRT (NVIDIA optimized)
- `pb` - TensorFlow (SavedModel)
- `tflite` - TensorFlow Lite (mobile)
- `torchscript` - TorchScript (C++ compatible)

---

## 🆘 Troubleshooting

### Training Crashes with Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
1. Reduce batch size:
   ```batch
   python scripts\train_optimized.py --batch 32
   ```
2. Reduce image size:
   ```batch
   python scripts\train_optimized.py --imgsz 512
   ```
3. Use smaller model:
   ```batch
   python scripts\train_optimized.py --model yolo11s.pt
   ```

### Training is Very Slow

**Error:** Training takes too long

**Solution:**
1. Reduce image size to 512 or 416
2. Reduce epochs: `--epochs 300`
3. Increase batch size (if memory allows)
4. Check GPU usage with: `nvidia-smi` (should be > 90%)

### Model Not Converging (Loss stuck high)

**Error:** Loss plateaus and doesn't decrease

**Solution:**
1. Increase training time: `--epochs 2000`
2. Reduce learning rate multiplier (in code)
3. Check dataset quality and balance
4. Verify data.yaml is correct

### Best Model Not Improving After 150 Epochs

**Information:** Training stops early (as intended)

**Solution:**
1. Increase patience: `--patience 300`
2. Collect more training data
3. Improve data quality/annotation
4. Check for class imbalance

---

## 📈 Training Tips for Best Results

### Dataset Quality

- ✅ Clean, well-labeled data is most important
- ✅ Balanced classes (similar number of each class)
- ✅ Diverse backgrounds and lighting
- ✅ Multiple angles and distances

### Hyperparameter Tuning

**For higher accuracy:**
- Increase epochs to 1000-2000
- Larger image size: 768 or 1024 (needs more VRAM)
- Smaller batch size (less memory but slower)

**For faster training:**
- Reduce epochs to 300-500
- Smaller image size: 416-512
- Larger batch size (needs more VRAM)

### Early Stopping

Training automatically stops if:
- No improvement for 150 epochs (patience)
- Can extend with `--patience 300`

### Resuming Training

If training is interrupted:
```batch
python scripts\train_optimized.py --data dataset/data.yaml --resume
```

This will:
1. Find the latest checkpoint
2. Resume from that epoch
3. Continue training

---

## RTX 5080 Optimized Settings

**Default configuration for best performance:**

```yaml
# configs/train_config.yaml
epochs: 1000              # Maximum accuracy
batch: 64                 # Optimal for RTX 5080 (16GB VRAM)
imgsz: 640                # Balance speed vs accuracy
workers: 16               # Maximize data loading
device: 0                 # GPU device ID

# Augmentation (recommended)
mosaic: 1.0               # Always mix images
mixup: 0.2                # 20% blend images
copy_paste: 0.5           # 50% copy-paste
```

These settings achieve:
- ✅ Best accuracy
- ✅ Reasonable training time (2-8 hours)
- ✅ Full GPU utilization
- ✅ No memory issues

See `docs/RTX5080_OPTIMIZED.md` for detailed GPU tuning.

---

## ✅ Next Steps

After training and validation:

1. **Deploy model** to your application
2. **Monitor performance** in production
3. **Collect feedback** and retrain periodically
4. **Fine-tune** on new data as needed

