# Quick Start Guide

## One Command (30 seconds)

```batch
AutoLabel\import_to_label_studio.bat
```

Automatically:
- Imports auto-labeled data
- Starts image server
- Opens Label Studio
- You edit labels
- Done!

## Setup (First Time - 2 minutes)

```powershell
cd AutoLabel
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
```

## The 3 Batch Files

| File | Purpose | Use When |
|------|---------|----------|
| `run_auto_label.bat` | Auto-label images | Starting from scratch |
| `verify_labels.bat` | View labels quickly | Quick quality check |
| `import_to_label_studio.bat` | Edit in web browser | ⭐ **Use this one** |

## Label Studio Editing (In Browser)

After batch file opens browser:

**Move/resize boxes:**
- Click inside box to move
- Drag corner to resize
- Right-click to delete

**Add/change labels:**
- Rectangle tool → draw new box → select class
- Click existing box → select new class

**Navigate:**
- Click Submit when done with image
- Auto-loads next image

**Export:**
- Click Export button
- Select YOLO format
- Download corrected labels

## YOLO Label Format

```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.15 0.2
```

All values are normalized (0-1).

## Complete Workflow

```
1. Auto-Label
   run_auto_label.bat
        ↓
2. Verify (optional)
   verify_labels.bat
        ↓
3. Review & Edit
   import_to_label_studio.bat
        ↓
4. Export labels
   Use for training
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Batch file won't run | Windows Defender blocking? Right-click → Run anyway |
| Virtual env error | `python -m venv .venv` then activate |
| Import fails | Check images & labels folders exist |
| Images won't load in Label Studio | Keep image server window open |
| Port already in use | Close other apps or restart |

---

**See `README.md` for full details**