# Label Studio Upload Issues - Troubleshooting Guide

## Error: "You cannot access body after reading from request's data stream"

**This happens when uploading very large image batches (500+ images).** Label Studio's Django server cannot handle the request stream properly.

### ⚠️ **For 1000+ Images: Mandatory Batch Upload**

If you have 1000-1400 images, **you MUST use batch uploads**. Here are solutions in order of effectiveness:

---

## ✅ Solution 1: Upload Images in Smaller Batches (Recommended for 1000+ images)

**This is the most reliable approach for large datasets.**

### Batch Upload Strategy:

1. **Open Label Studio** at `http://localhost:8080`
2. **Create ONE project** (don't create multiple projects)
3. **Upload images in batches:**

#### For 1400 images:
- **Batch 1:** Images 1-100 (drag & drop all at once)
- Wait for upload progress bar to complete (shows ✅)
- **Batch 2:** Images 101-200
- Wait for completion
- Continue until all batches uploaded

### Step-by-Step Upload Process:

```
Batch 1: Images 1-100
├─ Drag all 100 images to Label Studio upload area
├─ Wait for progress bar: ████████████████ 100%
├─ Should show "100 images added"
└─ Press Enter in terminal to continue

Batch 2: Images 101-200
├─ Drag next 100 images
├─ Wait for progress
├─ Should now show "200 images total"
└─ Continue...

(Repeat until all 1400 images uploaded)
```

### Why Batch Upload Works:

✅ Each upload is smaller (100 images = ~300MB)  
✅ Django can process request stream properly  
✅ No "cannot access body" errors  
✅ Reliable and consistent  

### 💡 Pro Tips:

- **Use drag-and-drop** (more reliable than file picker)
- **Upload 100 images at a time** (sweet spot for Django)
- **Wait between batches** (let server finish processing)
- **Don't close browser** during upload
- **Check upload progress bar** before continuing

---

## ✅ Solution 2: Use Smaller Batch Size (If 100 fails)

If you still get errors with 100 images per batch, reduce to 50:

---

## ✅ Solution 2: Use Smaller Batch Size (If 100 fails)

If you still get errors with 100 images per batch, reduce to 50:

1. Upload 50 images at a time
2. Monitor browser console for errors
3. If still failing, try 25 at a time
4. Continue reducing until uploads work

---

## ✅ Solution 3: Clear Browser Cache & Restart

Sometimes browser cache causes issues:

1. **Close Label Studio browser tab**
2. **Clear browser cache** (Ctrl+Shift+Delete)
3. **Restart Label Studio:**
   ```batch
   .venv\Scripts\activate.bat
   pip uninstall label-studio -y
   pip install label-studio==1.11.0
   label-studio
   ```
4. **Try uploading again** with smaller batch size

---

## ✅ Solution 4: Use Roboflow (Cloud-Based - No Upload Limits)

```batch
python scripts\label_images.py --tool roboflow
```

**Advantages:**
- ✅ Cloud-based (no local Django issues)
- ✅ AI auto-annotation available
- ✅ Handles large batches easily
- ✅ Free tier available
- ✅ No streaming errors

**Steps:**
1. Visit https://roboflow.com
2. Sign up (free tier)
3. Create new dataset
4. Upload images (can upload all at once)
5. Use AI auto-annotation if desired
6. Download annotations in YOLO format
7. Place `.txt` files in `dataset/labels/train/`

---

## ✅ Solution 4: Manual Annotation (If UI Tools Fail)

If you need to skip the UI tools entirely, create annotations manually:

### Format:
Each image needs a corresponding `.txt` file with the same name:

```
dataset/images/train/
├── image1.jpg
├── image2.jpg
└── ...

dataset/labels/train/
├── image1.txt          ← Create for each image
├── image2.txt
└── ...
```

### Text File Format:
Each line represents one object:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `image1.txt`
```
0 0.5 0.4 0.3 0.5
1 0.2 0.7 0.15 0.2
```

Where:
- `class_id`: 0, 1, 2, ... (integer for each class)
- `x_center`, `y_center`, `width`, `height`: Normalized to 0-1 range

### Calculation:
```
x_center = pixel_x_center / image_width
y_center = pixel_y_center / image_height
width = box_width / image_width
height = box_height / image_height
```

---

## ✅ Solution 5: Reinstall Label Studio Fresh

If errors persist, try a clean reinstall:

```batch
.venv\Scripts\activate.bat
pip uninstall label-studio -y
pip install label-studio
label-studio
```

Then try uploading again in smaller batches.

---

## 🔍 Why This Error Occurs

The error "cannot access body after reading from request's data stream" happens because:

1. Django middleware reads the request body to parse files
2. Multiple middleware components may read the stream
3. Once read, the stream cannot be accessed again
4. Large batch uploads trigger this more often

**This is a known Django/Label Studio issue** - not a problem with your setup.

---

## 📊 Recommended Workflow

**Fastest & Most Reliable:**

1. Use **Roboflow** for AI-assisted annotation
   - Upload all images at once
   - Let AI auto-annotate
   - Manual review
   - Download YOLO format

2. Or upload to **Label Studio in batches**
   - 50-100 images per batch
   - Use drag-and-drop
   - Wait between batches

3. As last resort: **Manual annotation**
   - For small datasets
   - When tools are unavailable

---

## 💡 Quick Reference

| Issue | Solution |
|-------|----------|
| Upload error | Try batch of 50 images |
| Still fails | Use Roboflow (cloud-based) |
| Want local tool | Upgrade Label Studio: `pip install --upgrade label-studio` |
| Need auto-labeling | Use Roboflow AI (free tier) |
| All tools fail | Manual annotation format |

---

## 📞 Support

- **Label Studio Docs:** https://labelstud.io/guide/
- **Roboflow Docs:** https://docs.roboflow.com
- **YOLO Format:** https://docs.ultralytics.com/datasets/detect/

