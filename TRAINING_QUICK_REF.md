# ⚡ Quick Reference - Flexible Training

## New Two-Step Selection

The training script now asks:

### Question 1: **Which model?**
```
1. Pretrained (yolo11m)
2. Load from file (your saved model)
```

### Question 2: **How to train?**
```
1. Fresh start (epoch 1)
2. Resume (from checkpoint)
```

---

## Common Scenarios

### 🚀 Fresh Training (First Time)
```bash
4.train.bat
→ Select 1 (Pretrained)
→ Select 1 (Fresh start)
✅ Trains from epoch 1
```

### 📊 Resume Training (After stopping at epoch 50)
```bash
4.train.bat
→ Select 2 (Load file) → select your .pt file
→ Select 2 (Resume)
✅ Continues from epoch 50
```

### 🔄 Retrain Model From Scratch
```bash
4.train.bat
→ Select 2 (Load file) → select your .pt file
→ Select 1 (Fresh start)
✅ Loads model but trains from epoch 1
```

### ⚙️ Resume via Command Line
```bash
python scripts/train_optimized.py --data dataset/data.yaml --resume
→ Select model
→ Training mode skipped → resumes
✅ Resumes training
```

---

## Workflow

```
START
  ↓
Choose Model:
  1. Pretrained yolo11m.pt
  2. Load your_model.pt
  ↓
Choose Training Mode:
  1. Fresh start (epoch 1→500)
  2. Resume (epoch N→500)
  ↓
TRAIN
```

---

## Key Points

✅ Flexible: Any model can be trained fresh OR resumed
✅ Simple: Just two questions
✅ Smart: Supports all scenarios
✅ Fast: One batch file or command

---

## Status: ✅ READY

Just run: `4.train.bat`

That's it! 🎉
