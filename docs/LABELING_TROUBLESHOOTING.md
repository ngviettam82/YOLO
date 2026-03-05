# Image Labeling Troubleshooting Guide

## Common Issues & Solutions

---

## Error: "DATA_UPLOAD_MAX_NUMBER_FILES exceeded"

### Problem
When uploading images directly to Label Studio via `2.label.bat`, Django rejects the upload:
```
The number of files exceeded settings.DATA_UPLOAD_MAX_NUMBER_FILES
```

### Cause
Label Studio uses Django which limits file uploads to ~1000 files by default. Your dataset has more files than this limit.

### Solution (Recommended): Use server-based import
Instead of uploading files directly, use the HTTP server-based import:
```batch
AutoLabel\import_to_label_studio.bat
```
This serves images via HTTP (no upload limit) and generates a `tasks.json` that Label Studio imports by URL reference.

### Alternative: Upload in small batches
If you must use direct upload:
1. Open Label Studio at http://localhost:8080
2. Create a project
3. Upload images in batches of 100 (drag & drop)
4. Wait for each batch to complete before uploading the next

---

## Error: "There was an issue loading URL from $image value"

### Problem
After importing `tasks.json`, Label Studio cannot load images from `http://localhost:8000/images/...`

### Causes & Solutions

**1. Image server not running**
The HTTP image server must be running in a separate window while using Label Studio.
```batch
cd label_studio_YOUR_PROJECT_NAME
python serve_images.py
```

**2. Image server started from wrong directory**
The server must run from inside the `label_studio_*` project folder (where the `images/` subfolder is), not from the YOLO project root.
```batch
cd label_studio_YOLO_Review_raw_dataset
python serve_images.py
```

**3. CORS blocked**
Label Studio requires the image server to send CORS headers. The `serve_images.py` included in each project folder already has CORS headers. If you use a different server, ensure it sends:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, OPTIONS
```

**4. URL mismatch (http vs https)**
Label Studio's scheme must match the image server scheme. If Label Studio runs on `http://localhost:8080`, image URLs must also use `http://` (not `https://`).

**5. Test the image server directly**
Open your browser and navigate to an image URL from tasks.json:
```
http://localhost:8000/images/YOUR_IMAGE_NAME.jpg
```
If the image loads in the browser but not in Label Studio, it's a CORS issue.

---

## Error: "cannot import name 'APP' from 'labelImg.labelImg'"

### Problem
LabelImg fails with error: `cannot import name 'APP' from 'labelImg.labelImg'`

This occurs due to compatibility issues with the LabelImg package versions.

### ✅ Solution 1: Reinstall LabelImg (Recommended)

```batch
.venv\Scripts\activate.bat
pip uninstall labelimg -y
pip install labelimg --upgrade
```

Then run `3.label.bat` again.

### ✅ Solution 2: Manual Command Line Approach

If reinstalling doesn't work, launch LabelImg directly:

```batch
.venv\Scripts\activate.bat
labelImg dataset\images\train yolo
```

This bypasses the Python module import issue entirely.

### ✅ Solution 3: Use Alternative Annotation Tools

If LabelImg still won't work, use a different tool (all are free):

#### Option A: Roboflow (Easiest - AI-Assisted)
- **Website:** https://roboflow.com
- **Features:** Free tier, AI auto-annotation, no installation
- **Steps:**
  1. Sign up for free account
  2. Create new dataset
  3. Upload images from `dataset/images/train/`
  4. AI auto-annotates (optional)
  5. Download in YOLO format
  6. Place `.txt` files in `dataset/labels/train/`

#### Option B: Label Studio (Easy - Web-based)
- **Installation:**
  ```batch
  .venv\Scripts\activate.bat
  pip install label-studio
  label-studio
  ```
- **Usage:**
  1. Open browser to `http://localhost:8080`
  2. Create new project
  3. Upload images from `dataset/images/train/`
  4. Create bounding box annotations
  5. Export in YOLO format
  6. Place `.txt` files in `dataset/labels/train/`

#### Option C: CVAT (Professional - Web-based)
- **Website:** https://github.com/opencv/cvat
- **Requirements:** Docker
- **Features:** Team collaboration, advanced features
- **Setup:**
  1. Install Docker: https://www.docker.com/products/docker-desktop
  2. Run: `docker run -d -p 8080:8080 --name cvat cvat/cvat:latest`
  3. Open browser to `http://localhost:8080`
  4. Create project and upload images
  5. Annotate and export as YOLO format

#### Option D: OpenLabeling (Fast - Desktop)
- **GitHub:** https://github.com/Cartucho/OpenLabeling
- **Features:** Lightweight, keyboard shortcuts, fast
- **Setup:**
  ```batch
  git clone https://github.com/Cartucho/OpenLabeling
  cd OpenLabeling
  pip install -r requirements.txt
  python main.py
  ```
- **Usage:** Select `dataset/images/train/` and draw boxes

### ✅ Solution 4: Manual Annotation Format

If you want to manually create annotations without a tool:

**Create `.txt` files matching image names:**

```
dataset/
├── images/
│   └── train/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    └── train/
        ├── image1.txt          ← Create these files
        ├── image2.txt          ← One per image
        └── ...
```

**File format for each `.txt` file:**

Each line represents one object:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `image1.txt` with 2 objects:
```
0 0.5 0.4 0.3 0.5
1 0.2 0.7 0.15 0.2
```

Where:
- `class_id`: 0, 1, 2, ... (integer for each class)
- `x_center`, `y_center`, `width`, `height`: All normalized to 0-1 range
- **Calculation:**
  - `x_center = pixel_x_center / image_width`
  - `y_center = pixel_y_center / image_height`
  - `width = box_width / image_width`
  - `height = box_height / image_height`

**Example in Python:**
```python
# If image is 640x480 and object is at pixels (200, 150) with size 100x200
image_width, image_height = 640, 480
pixel_x, pixel_y = 200, 150
box_width, box_height = 100, 200

x_center = pixel_x / image_width  # = 0.3125
y_center = pixel_y / image_height  # = 0.3125
width = box_width / image_width    # = 0.15625
height = box_height / image_height # = 0.41667

# Write to file
with open('image1.txt', 'w') as f:
    f.write(f"0 {x_center} {y_center} {width} {height}\n")
```

---

## 🟡 Error: "No training images folder found"

### Problem
Error: `No training images folder found!` or `No images found in dataset/images/train/`

### ✅ Solution

1. **Run dataset preparation first:**
   ```batch
   2.dataset.bat
   ```

2. **Add images to `raw_dataset/`:**
   ```
   YOLO/
   └── raw_dataset/          ← Create this if doesn't exist
       ├── image1.jpg
       ├── image2.png
       └── ...
   ```

3. **Run dataset prep again:**
   ```batch
   2.dataset.bat
   ```

4. **Verify structure created:**
   ```
   YOLO/
   └── dataset/
       ├── images/
       │   ├── train/         ← Should have images here
       │   ├── val/
       │   └── test/
       └── labels/
   ```

---

## 🟡 Warning: "Tool did not complete successfully"

### Problem
Labeling tool exits without completing.

### ✅ Solution

1. **Check Python version:**
   ```batch
   python --version
   ```
   Should be 3.10.x

2. **Verify PyQt5 (required for GUI):**
   ```batch
   .venv\Scripts\activate.bat
   pip install PyQt5
   ```

3. **Try alternative tool:**
   - Roboflow (no installation needed)
   - Label Studio
   - Manual annotation

4. **Check GPU status (if applicable):**
   ```batch
   nvidia-smi
   ```

---

## 🟡 LabelImg Window Won't Open

### Problem
LabelImg command runs but no window appears.

### ✅ Solutions

1. **Ensure display access (for remote/SSH):**
   If running remotely, ensure X11 forwarding or use web-based tool instead.

2. **Try with explicit display:**
   ```batch
   set QT_QPA_PLATFORM=offscreen
   labelImg dataset\images\train yolo
   ```

3. **Use alternative tool instead:**
   Roboflow or Label Studio don't require GUI access.

---

## ✅ Verification Checklist

After successful annotation:

- [ ] Visited `dataset/images/train/` - contains images
- [ ] Visited `dataset/labels/train/` - contains `.txt` files
- [ ] Number of `.txt` files matches number of images
- [ ] Each `.txt` file has annotation lines
- [ ] Class IDs in `.txt` files are 0, 1, 2, ... (sequential)

**Verify with script:**
```batch
.venv\Scripts\activate.bat
python -c "import os; train_imgs = len([f for f in os.listdir('dataset/images/train') if f.endswith(('.jpg','.png'))]); train_labels = len([f for f in os.listdir('dataset/labels/train') if f.endswith('.txt')]); print(f'Images: {train_imgs}, Labels: {train_labels}')"
```

Should show equal counts.

---

## 📊 Next Steps After Annotation

Once annotations are complete:

1. **Create/Update `dataset/data.yaml`:**
   ```batch
   .venv\Scripts\activate.bat
   python scripts\label_images.py --config --num-classes 3
   ```

2. **Edit class names in `dataset/data.yaml`:**
   ```yaml
   nc: 3
   names:
     0: person
     1: car
     2: dog
   ```

3. **Start training:**
   ```batch
   4.train.bat
   ```

---

## 🆘 Still Having Issues?

### Debug Steps

1. **Check virtual environment:**
   ```batch
   .venv\Scripts\activate.bat
   pip list | findstr labelimg
   ```

2. **Test LabelImg directly:**
   ```batch
   .venv\Scripts\activate.bat
   python -c "import labelImg; print('LabelImg OK')"
   ```

3. **Check image files:**
   ```batch
   dir dataset\images\train /s
   ```

4. **Check permissions:**
   - Right-click `3.label.bat` → Run as Administrator

### Contact & Resources

- **LabelImg GitHub:** https://github.com/heartexlabs/labelImg
- **Roboflow Docs:** https://docs.roboflow.com
- **Label Studio Docs:** https://labelstud.io/docs
- **YOLO Format Docs:** https://docs.ultralytics.com/datasets/detect/

---

## 💡 Pro Tips

✅ **Keyboard shortcuts in LabelImg:**
- `W` - Draw bounding box
- `D` - Next image
- `A` - Previous image
- `Ctrl+S` - Save
- `Ctrl+R` - Change directory

✅ **Batch annotation with Roboflow:**
- Use AI auto-annotation to label all images at once
- Manual review 50+ images for quality
- Download and use for training

✅ **Speed up manual annotation:**
- Start with `raw_dataset/` organized by class
- Use keyboard shortcuts
- Set reasonable confidence threshold
- Use tools that support batch operations

