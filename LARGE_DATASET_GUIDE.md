# Large Dataset (1000+ Images) - Complete Annotation Guide

## Problem: Django Streaming Errors with Large Batches

When uploading 1000+ images to Label Studio at once, you get:
```
You cannot access body after reading from request's data stream
```

**Root Cause:** Django middleware reads the request body once, then cannot re-read it for file processing.

**Solution:** Use batch uploads with proper workflow.

---

## 🚀 Recommended Workflow: Batch Upload Strategy

### Setup (5 minutes)

1. **Run Label Studio:**
   ```batch
   3.label.bat
   ```

2. **Browser opens:** http://localhost:8080
3. **Create ONE project** with name: "fire_detection" (or your project name)
4. **Keep browser open** during entire process

---

## 📊 Batch Upload Table (for 1400 images)

| Batch | Image Range | Files | Approx Size |
|-------|-------------|-------|------------|
| 1 | 1-100 | 100 | ~300MB |
| 2 | 101-200 | 100 | ~300MB |
| 3 | 201-300 | 100 | ~300MB |
| 4 | 301-400 | 100 | ~300MB |
| 5 | 401-500 | 100 | ~300MB |
| 6 | 501-600 | 100 | ~300MB |
| 7 | 601-700 | 100 | ~300MB |
| 8 | 701-800 | 100 | ~300MB |
| 9 | 801-900 | 100 | ~300MB |
| 10 | 901-1000 | 100 | ~300MB |
| 11 | 1001-1100 | 100 | ~300MB |
| 12 | 1101-1200 | 100 | ~300MB |
| 13 | 1201-1300 | 100 | ~300MB |
| 14 | 1301-1400 | 100 | ~300MB |

**Total batches:** 14 uploads  
**Estimated time:** 30-60 minutes (depends on internet/disk speed)

---

## 📝 Step-by-Step Upload Instructions

### Batch 1: Images 1-100

1. **Open file explorer** and navigate to:
   ```
   dataset\images\train\
   ```

2. **Select first 100 images:**
   - Sort by name
   - Select images 1-100
   - Copy file paths

3. **In Label Studio browser:**
   - Click **"Data"** tab
   - Click **"Upload Files"**
   - **Drag & Drop** all 100 images into upload area

4. **Monitor upload:**
   - Watch progress bar: `████░░░░░░░░░░░ 30%`
   - Wait until complete: `████████████████ 100%`
   - Should see: "100 images added to project"

5. **Verify:**
   - Check: Top shows "100 images"
   - All images appear in list
   - No error messages

6. **In terminal:** Press Enter to continue

---

### Batch 2: Images 101-200

1. Repeat same process with images 101-200
2. Wait for upload completion
3. Verify: Top shows "200 images" (cumulative)
4. Press Enter to continue

---

### Continue Batches 3-14

Repeat the same process for each batch:
- Select 100 images
- Drag & drop
- Wait for progress
- Verify count increases
- Continue

---

## ⏱️ Timeline Estimate

| Task | Time |
|------|------|
| **Upload Batch 1** | 2-3 min |
| **Upload Batch 2** | 2-3 min |
| ... (batches 3-13) | ... |
| **Upload Batch 14** | 2-3 min |
| **TOTAL UPLOAD TIME** | 30-45 min |
| **Annotation Time** | 4-8 hours (depends on complexity) |
| **Export & Finish** | 5-10 min |

---

## 💡 Pro Tips for Success

### ✅ DO:
- ✅ Use **drag & drop** (more reliable than file picker)
- ✅ Wait for progress bar to finish before continuing
- ✅ Keep browser tab open during entire process
- ✅ Upload during consistent internet connection
- ✅ Use 100 images per batch (optimal for Django)
- ✅ Verify count increases after each batch
- ✅ Take breaks between batches (server needs rest)

### ❌ DON'T:
- ❌ Upload all 1400 images at once (will fail)
- ❌ Close browser during uploads
- ❌ Click "Upload" multiple times
- ❌ Refresh the page during upload
- ❌ Upload while computer is sleeping
- ❌ Upload over unstable WiFi

---

## 🆘 Troubleshooting

### Issue: "Cannot access body after reading from request's data stream"

**Solution 1: Reduce batch size**
- Try 50 images instead of 100
- Or even 25 images

**Solution 2: Restart Label Studio**
```batch
REM Press Ctrl+C in Label Studio terminal
REM Then run again:
3.label.bat
```

**Solution 3: Clear cache**
```batch
.venv\Scripts\activate.bat
pip uninstall label-studio -y
pip install label-studio==1.11.0
label-studio
```

### Issue: "Upload stuck at 50%"

**Solution:**
1. Press Ctrl+C in terminal
2. Wait 30 seconds
3. Restart: `3.label.bat`
4. Try uploading batch again with smaller size

### Issue: "Connection timeout"

**Solution:**
1. Check internet connection
2. Reduce batch size to 50
3. Wait between batches (2-3 min per batch)

---

## 📋 Annotation Workflow

### Once All Images Uploaded:

1. **Create labels** (if not already done):
   - Go to project settings
   - Create label: "fire" (or your class name)
   - Save

2. **Start annotation:**
   - Click first image
   - Draw bounding box around fire
   - Select label "fire"
   - Repeat for each fire in image
   - Move to next image

3. **Export annotations:**
   - Go to **"Export"** button
   - Select format: **"YOLO"**
   - Download as ZIP
   - Extract to `dataset/labels/train/`

4. **Verify export:**
   - Check that `.txt` files exist in `dataset/labels/train/`
   - Each `.txt` file matches an image
   - Format: `class_id x_center y_center width height`

---

## 📊 YOLO Format Reference

### File Structure:
```
dataset/
├── images/train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (1400 images)
└── labels/train/
    ├── image1.txt      ← MUST exist for each image
    ├── image2.txt
    └── ... (1400 files)
```

### Label File Format (image1.txt):
```
0 0.523 0.456 0.234 0.567
0 0.712 0.234 0.156 0.345
```

**Each line:**
- `0` = class ID (0=fire, 1=smoke, etc.)
- `0.523` = center X (normalized 0-1)
- `0.456` = center Y (normalized 0-1)
- `0.234` = width (normalized 0-1)
- `0.567` = height (normalized 0-1)

### Example with comments:
```
0 0.5 0.5 0.3 0.4   ← Fire at center, 30% width, 40% height
0 0.2 0.8 0.1 0.15  ← Smaller fire at bottom-left
```

---

## 🎯 After Annotation: Next Steps

1. **Verify labels:**
   ```batch
   python scripts\validate_model.py --data dataset/data.yaml
   ```

2. **Update dataset config:**
   - Edit `dataset/data.yaml`
   - Set correct number of classes
   - Update class names

3. **Start training:**
   ```batch
   4.train.bat
   ```

---

## 📞 Additional Resources

- **Label Studio Docs:** https://labelstud.io/guide/
- **YOLO Format:** https://docs.ultralytics.com/datasets/detect/
- **Roboflow Alternative:** https://roboflow.com (cloud-based, for 1000+ images)

---

## ✅ Checklist

Before starting training, verify:
- ✅ All 1400 images uploaded to Label Studio
- ✅ All images have bounding box annotations
- ✅ Annotations exported in YOLO format
- ✅ 1400 `.txt` files in `dataset/labels/train/`
- ✅ Each `.txt` file has format: `class_id x y w h`
- ✅ `dataset/data.yaml` updated with correct classes
- ✅ No images without corresponding `.txt` files

**Once verified, ready to train! 🚀**

