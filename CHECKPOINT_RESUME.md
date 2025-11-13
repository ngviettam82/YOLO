# ✅ Checkpoint File Selection for Resume Training

## New Feature: Interactive Checkpoint Selection

When you choose **"Continue training"** (Option 2), the script now prompts you to select a checkpoint file to resume from.

---

## Training Workflow

```
Run: 4.train.bat
    ↓
Step 1: Select Model
  1. Pretrained (yolo11m)
  2. Load from file
    ↓
Step 2: Select Training Mode
  1. Fresh start (epoch 1)
  2. Continue training
    ↓
IF Continue training selected:
  → File dialog opens
  → Select checkpoint file (.pt)
  → Training resumes from selected checkpoint
    ↓
Training starts
```

---

## Example Usage

### Scenario 1: Fresh Training

```
4.train.bat

Choose model source:
  1. Use pretrained model (yolo11m - fresh download)
  2. Load from file (previously trained model)
Enter your choice (1 or 2): 1
✓ Selected: yolo11m.pt (pretrained)

Choose training mode:
  1. Fresh start (start training from epoch 1)
  2. Continue training (resume from checkpoint)
Enter your choice (1 or 2): 1
✓ Selected: Fresh start training

📌 Training mode: FRESH START (starting from epoch 1)
Epoch 1/5000...
```

### Scenario 2: Resume Training

```
4.train.bat

Choose model source:
  1. Use pretrained model (yolo11m - fresh download)
  2. Load from file (previously trained model)
Enter your choice (1 or 2): 1
✓ Selected: yolo11m.pt (pretrained)

Choose training mode:
  1. Fresh start (start training from epoch 1)
  2. Continue training (resume from checkpoint)
Enter your choice (1 or 2): 2
✓ Selected: Continue training

Select checkpoint file to resume from...
(File dialog opens)
→ Navigate to: runs/train_20251113_145830/
→ Select: best.pt or last.pt
✓ Selected checkpoint: best.pt

📌 Training mode: RESUME (continuing from checkpoint)
   Checkpoint: best.pt

Epoch 150/5000...  (resumes from epoch 150)
```

---

## How to Resume Training

### Step-by-Step

1. **Stop current training** (Ctrl+C)
   - Model saves automatically to `runs/train_YYYYMMDD_HHMMSS/`
   - Files created:
     - `best.pt` - Best performing model so far
     - `last.pt` - Last epoch model

2. **Run training again**
   ```bash
   4.train.bat
   ```

3. **Select model:**
   ```
   Choose model source:
     1. Use pretrained model
     2. Load from file
   Enter: 1 (or 2 if using previously trained model)
   ```

4. **Select training mode:**
   ```
   Choose training mode:
     1. Fresh start
     2. Continue training
   Enter: 2
   ```

5. **File dialog opens**
   - Navigate to your checkpoint directory
   - Select a `.pt` file (typically `best.pt` or `last.pt`)
   - Click "Open"

6. **Training resumes**
   - Continues from the saved epoch
   - Same hyperparameters
   - Improves on existing weights

---

## Checkpoint Files

### What Gets Saved

When training runs, models are saved to:
```
runs/train_YYYYMMDD_HHMMSS/
  ├── weights/
  │   ├── best.pt    ← Best model (highest mAP50)
  │   └── last.pt    ← Last epoch trained
  ├── args.yaml      ← Training configuration
  ├── results.csv    ← Metrics per epoch
  └── plots/         ← Training plots
```

### Which File to Resume From

- **`best.pt`** (recommended)
  - Best performing model during training
  - Highest validation accuracy
  - Prevents overfitting

- **`last.pt`**
  - Last completed epoch
  - Continues from exact stop point
  - Useful for fine-tuning

---

## Implementation Details

### New Methods

```python
select_training_mode()
  └─ Returns: (resume: bool, checkpoint_path: str or None)
     - Fresh start: (False, None)
     - Continue: (True, "/path/to/checkpoint.pt")

_select_checkpoint_file()
  └─ Opens file dialog to select checkpoint
     Returns: checkpoint_path (or None if canceled)

train(dataset_yaml, resume=False, checkpoint_path=None)
  └─ Accepts checkpoint_path parameter
     - If resume=True and checkpoint_path provided:
       Loads checkpoint and resumes training
     - Otherwise: Starts fresh with base model
```

### Code Flow

```python
def main():
    trainer = SimpleYOLOTrainer()
    trainer.select_model()
    
    # Returns tuple (resume, checkpoint_path)
    resume, checkpoint_path = trainer.select_training_mode()
    
    # Pass checkpoint path to train
    trainer.train(
        dataset_yaml=args.data,
        resume=resume,
        checkpoint_path=checkpoint_path
    )
```

---

## Features

✅ **Interactive Selection** - File dialog appears automatically when resuming
✅ **Flexible** - Select any checkpoint file (best.pt, last.pt, custom.pt)
✅ **Intuitive** - Clear prompts guide the user
✅ **Reliable** - YOLO handles resume internally
✅ **Safe** - Original model preserved, checkpoint loaded for resume

---

## Common Scenarios

### Extend Training Duration
```
Initial run: 5000 epochs
Stop at: Epoch 1500
Resume training: Continues from epoch 1500 to 5000
Result: Total 5000 epochs completed
```

### Switch Checkpoint Mid-Training
```
Trained: best.pt (mAP50: 0.75)
Trained: last.pt (mAP50: 0.73, overfitting detected)
Resume from: best.pt (to avoid overfitting)
Result: Continue with best model
```

### Fine-Tune with Different Data
```
Trained: model.pt (on dataset A)
Resume from: model.pt
Train on: dataset B
Result: Transfer learning / fine-tuning
```

---

## Git Commit

```
Commit: 0e2855a
Message: Add checkpoint file selection for resume training

- select_training_mode() now returns (resume, checkpoint_path) tuple
- Added _select_checkpoint_file() to prompt file selection for resuming
- train() method updated to accept and use checkpoint_path
- When user selects 'continue training', file dialog appears to select checkpoint
- Checkpoint is loaded for resuming training from saved state
```

---

## Next: Ready to Train!

Just run:
```bash
4.train.bat
```

Answer the prompts, and training will start or resume! 🚀
