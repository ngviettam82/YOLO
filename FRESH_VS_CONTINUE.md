# ⚡ Quick Reference - Fresh vs Continue Training

## Two Training Options

### Option 1: Fresh Start ✅
```
Training Mode Selection:
  1. Fresh start ← SELECT THIS
  2. Continue training

Result:
  - Starts from epoch 1
  - Resets all training state
  - New training run
```

### Option 2: Continue Training ✅
```
Training Mode Selection:
  1. Fresh start
  2. Continue training ← SELECT THIS

Prompt:
  → File dialog appears
  → Select checkpoint file (.pt)

Result:
  - Resumes from selected checkpoint
  - Continues improving model
  - Preserves training progress
```

---

## Which Checkpoint to Resume From?

Located in: `runs/train_YYYYMMDD_HHMMSS/weights/`

| File | Use When |
|------|----------|
| `best.pt` | Want best accuracy (recommended) |
| `last.pt` | Want to continue from last epoch |
| `custom.pt` | Using specific checkpoint |

---

## Complete Workflow

### First Training Run
```
4.train.bat
→ Select model: 1 (Pretrained)
→ Select mode: 1 (Fresh start)
→ Training starts from epoch 1
... train for a while ...
→ Stop with Ctrl+C
→ Best model saved to: runs/train_20251113_145830/weights/best.pt
```

### Resume Training
```
4.train.bat
→ Select model: 1 (Pretrained)
→ Select mode: 2 (Continue training)
→ File dialog opens
→ Navigate to: runs/train_20251113_145830/weights/
→ Select: best.pt
→ Training resumes from epoch 150
```

---

## Key Points

✅ Fresh start: Every time restart from epoch 1
✅ Continue: Load checkpoint, resume from saved epoch
✅ File dialog: Automatically appears for checkpoint selection
✅ Flexible: Any .pt file can be used as checkpoint

---

## Common Issues & Fixes

### Can't find checkpoint
- Look in: `runs/train_YYYYMMDD_HHMMSS/weights/`
- Files: `best.pt` or `last.pt`

### Wrong checkpoint selected
- Just select again when prompted
- File dialog will appear again

### Want to force fresh start
- Select option 1: "Fresh start"
- Won't ask for checkpoint file

---

## Status: ✅ READY

Run:
```bash
4.train.bat
```

Answer 2 questions and training begins! 🚀
