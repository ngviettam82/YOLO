# 🔄 Flexible Training Mode - Model + Training Mode Selection

## New Training Workflow

The training script now asks two separate questions:

### Step 1️⃣: Select Model
```
Choose model source:
  1. Use pretrained model (yolo11m - fresh download)
  2. Load from file (previously trained model)
```

### Step 2️⃣: Select Training Mode
```
Choose training mode:
  1. Fresh start (start training from epoch 1)
  2. Resume training (continue from last checkpoint)
```

---

## Scenarios

### Scenario 1: Fresh Training with Pretrained Model
```
Step 1: Select 1 (Pretrained)
Step 2: Select 1 (Fresh start)
Result: ✅ Trains yolo11m.pt from epoch 1 on your fire dataset
```

**Use case:** First-time training, starting from scratch

---

### Scenario 2: Resume Previous Training

**First session:**
```
Step 1: Select 1 (Pretrained)
Step 2: Select 1 (Fresh start)
Wait for training...
Epoch 50/500 - Press Ctrl+C to stop
Model saves to: runs/train_20251113_145830/weights/best.pt
```

**Later session - Resume:**
```
Run 4.train.bat again
Step 1: Select 2 (Load from file)
        → Select: runs/train_20251113_145830/weights/best.pt
Step 2: Select 2 (Resume training)
Result: ✅ Resumes from epoch 50, continues to epoch 500
```

**Use case:** Extend training that was interrupted

---

### Scenario 3: Fresh Start with Previously Trained Model

If you want to retrain a previously trained model from scratch:

```
Step 1: Select 2 (Load from file)
        → Select: runs/train_20251113_145830/weights/best.pt
Step 2: Select 1 (Fresh start)
Result: ✅ Loads model but starts new training from epoch 1
```

**Use case:** Reset a model and train again

---

### Scenario 4: Resume via Command Line

For automation or batch scripts:

```bash
python scripts/train_optimized.py --data dataset/data.yaml --resume
```

Then select:
```
Step 1: Model source (1 or 2)
Step 2: Skipped - uses --resume flag → Resume training
```

---

## Implementation Details

### Three Methods

```python
select_model()
  └─ Asks: Pretrained or file?
     Returns: model_path (string)

select_training_mode()
  └─ Asks: Fresh start or resume?
     Returns: resume (True/False)

train(dataset_yaml, resume)
  └─ Trains model with given resume setting
```

### Flow in main()

```python
def main():
    trainer = SimpleYOLOTrainer()
    
    # Step 1: Get model
    trainer.select_model()
    
    # Step 2: Get training mode (unless --resume provided)
    if args.resume:
        resume = True  # Skip prompt
    else:
        resume = trainer.select_training_mode()
    
    # Train
    trainer.train(dataset_yaml=args.data, resume=resume)
```

---

## Key Features

✅ **Flexible:** Any model (pretrained or trained) can be trained fresh or resumed
✅ **Simple:** Only two decisions, clear prompts
✅ **Powerful:** Supports both interactive and command-line modes
✅ **Intuitive:** Separates "what model" from "how to train"

---

## Usage Examples

### Example 1: Interactive (Default)
```bash
4.train.bat
```

### Example 2: Command Line - Resume Mode
```bash
python scripts/train_optimized.py --data dataset/data.yaml --resume
```

### Example 3: Script Automation
```bash
python scripts/train_optimized.py --data dataset/data.yaml
```

---

## Decision Matrix

| Model Type | Training Mode | Command | Result |
|-----------|--------------|---------|--------|
| Pretrained | Fresh start | Normal | ✅ Train from epoch 1 |
| Pretrained | Resume | Normal | ✅ Resume if checkpoint exists |
| Trained file | Fresh start | Normal | ✅ Load file, train from epoch 1 |
| Trained file | Resume | Normal | ✅ Load file, resume from checkpoint |
| Any | Resume | `--resume` flag | ✅ Skip mode selection, resume |

---

## Example Run

```
4.train.bat
↓
================================================================================
  YOLO Training Quickstart
================================================================================

[1/3] Checking dataset configuration...
Dataset config found: dataset/data.yaml

[2/3] Training Configuration:
  Model Selection: Pretrained (yolo11m) or Load from file
  Training Mode: Fresh start or Resume from checkpoint
  ...

[3/3] Starting training...

🚀 GPU Detected: NVIDIA GeForce RTX 5080
💾 VRAM Available: 15.9 GB

================================================================================
🏷️  Model Selection
================================================================================

Choose model source:
  1. Use pretrained model (yolo11m - fresh download)
  2. Load from file (previously trained model)

Enter your choice (1 or 2): 1
✓ Selected: yolo11m.pt (pretrained)

================================================================================
⚙️  Training Mode Selection
================================================================================

Choose training mode:
  1. Fresh start (start training from epoch 1)
  2. Resume training (continue from last checkpoint)

Enter your choice (1 or 2): 1
✓ Selected: Fresh start training

📊 Dataset Statistics:
   Training images: 228
   Validation images: 65

🔄 Loading model: yolo11m.pt
📌 Training mode: FRESH START (starting from epoch 1)

🏋️  Starting training...

Epoch 1/500   - loss_box: 2.34, loss_cls: 1.23, loss_dfl: 0.56
Epoch 2/500   - loss_box: 2.12, loss_cls: 1.01, loss_dfl: 0.45
...
```

---

## Benefits Over Previous Approach

**Old:**
- Model selection and training mode were coupled
- Pretrained only → fresh training
- Trained only → could resume
- Less flexible

**New:**
- Model selection and training mode are independent
- Any model can be trained fresh or resumed
- More intuitive: separate decisions
- More powerful: supports all scenarios

---

## Git Commit

```
Commit: 81d9228
Message: Refactor: Separate model selection from training mode

- Users now choose model first (pretrained or file)
- Then choose training mode (fresh start or resume)
- More flexible: any model can be trained fresh or resumed
- Simplified: removed model_type dependency
```

---

## Next: Ready to Train! 🚀

Just run:
```bash
4.train.bat
```

And answer two simple questions!
