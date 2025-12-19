# Getting Started - FOL Symbolic Transformer

## ⚠️ GPU Detection Issue

Your system shows: **"No GPU detected"** but you have AMD Radeon 8060S with 96GB VRAM.

### Why This Happens

PyTorch was installed with CUDA support, but you have an AMD GPU (needs ROCm).

### Fix GPU Support

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with ROCm 6.0 support (for your Radeon)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify GPU is detected
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
AMD Radeon Graphics
```

---

## ✅ Quick Start (Works Now - CPU Mode)

Even without GPU, you can test everything on CPU (just slower):

### Step 1: Test Vocabulary (30 seconds)

```bash
cd /home/claude  # or wherever you extracted files
python3 quickstart.py
# Choose option 1
```

This verifies:
- ✅ Vocabulary loading works (662 symbols)
- ✅ Encoding/decoding works
- ✅ Compositional structure works

### Step 2: Generate Test Dataset (5 minutes)

```bash
python3 quickstart.py
# Choose option 2
```

Generates:
- 100 training formulas (~500 samples)
- 20 validation formulas
- 20 test formulas

### Step 3: Train Tiny Model (30 minutes on CPU)

```bash
python3 quickstart.py
# Choose option 3
```

Trains a tiny model (500K params) to verify:
- ✅ Model architecture works
- ✅ Training loop works
- ✅ Checkpointing works

**Note:** This is just for testing. For real training, you want the GPU.

---

## 🚀 Once GPU is Working

After fixing GPU support:

### Quick Test

```bash
python3 quickstart.py
# Choose option 4 (Full pipeline)
```

### Or Manual Control

```bash
# 1. Generate full dataset (10K formulas)
cd fol_transformer/data
python3 dataset_generator.py

# 2. Train base model (~5 hours on GPU)
cd ../training
python3 train.py

# 3. Evaluate
cd ../evaluation
python3 evaluate.py
```

---

## 📁 Files You Have

```
/home/claude/
├── unified_vocabulary.json       ✅ Created (662 symbols)
├── numerals_decode_table.json   ✅ Extracted from zip
├── decode_table_symbols.json    ✅ Extracted from zip
├── merge_vocabulary.py          ✅ Working
├── quickstart.py                ✅ New - easy testing
├── run_pipeline.py              ✅ Full automation
└── fol_transformer/
    ├── data/
    │   └── dataset_generator.py
    ├── models/
    │   └── transformer.py
    ├── training/
    │   └── train.py
    ├── evaluation/
    │   └── evaluate.py
    └── utils/
        └── vocabulary.py
```

---

## 🎯 What Each Script Does

### `quickstart.py` (NEW - Use This!)

Interactive menu for:
1. Test vocabulary (30 sec)
2. Generate small dataset (5 min)
3. Train tiny model on CPU (30 min)
4. Full pipeline (hours, needs GPU)

### `merge_vocabulary.py`

Already run! Created `unified_vocabulary.json` with 662 symbols.

### `run_pipeline.py`

Full automation (needs GPU for reasonable speed):
- Merges vocab
- Generates 10K formulas
- Trains model
- Evaluates

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named X"

```bash
cd fol_transformer
pip install -r requirements.txt
```

### "FileNotFoundError: unified_vocabulary.json"

Already fixed! File is in `/home/claude/unified_vocabulary.json`

### Training is slow

That's normal on CPU. Options:
1. Fix GPU support (recommended)
2. Use smaller model (tiny)
3. Use fewer epochs
4. Use smaller dataset

### Out of memory

Reduce batch size in `training/train.py`:
```python
batch_size = 8  # or even 4
```

---

## 📊 Expected Performance

### On CPU (for testing only)

| Model | Training Time | Memory |
|-------|--------------|--------|
| Tiny  | 30 min       | ~2GB   |
| Small | 2 hours      | ~4GB   |

### On GPU (AMD Radeon 8060S, 96GB)

| Model | Training Time | Memory |
|-------|--------------|--------|
| Tiny  | 5 min        | ~2GB   |
| Small | 20 min       | ~4GB   |
| Base  | 5 hours      | ~8GB   |
| Large | 12 hours     | ~16GB  |

---

## 🎓 Understanding Your Symbols

Your unified vocabulary has:

**625 numerals (IDs 0-624)**
- Encoded as 25×25 grid patterns
- Row-major counting (0=empty, 624=full)
- Used for variable/predicate indices

**37 FOL symbols (IDs 625-661)**
- Logical operators: ¬, ∧, ∨, →, ↔
- Quantifiers: ∀, ∃
- Relations: =, ≠, <, >, ≤, ≥
- Structural: (, ), ,
- Category tokens: VAR, CONST, PRED, FUNC, SORT
- Meta-logic: ⊢, ⊨, ≔, ≡

**Compositional encoding:**
```
∀x P₅(x) = [FORALL, VAR, 1, PRED, 5, LPAREN, VAR, 1, RPAREN]
```

---

## 💡 Next Steps

1. **Fix GPU** (recommended for speed)
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```

2. **Test vocabulary** (quick verification)
   ```bash
   python3 quickstart.py  # option 1
   ```

3. **Generate dataset** (5 min)
   ```bash
   python3 quickstart.py  # option 2
   ```

4. **Train model** (30 min CPU or 5 hours GPU)
   ```bash
   python3 quickstart.py  # option 3 (CPU) or 4 (GPU)
   ```

5. **Review results**
   - Check `checkpoints/best_model.pt`
   - Run evaluation script
   - Try interactive generation

---

## 📖 Documentation

- **SETUP_GUIDE.md** - Comprehensive setup instructions
- **TECHNICAL_ARCHITECTURE.md** - Deep dive into architecture
- **fol_transformer/README.md** - Full project documentation

---

## 🆘 Need Help?

Common issues:

**"No GPU detected"** → Install ROCm PyTorch (see above)

**"Missing vocabulary"** → Already created! At `/home/claude/unified_vocabulary.json`

**Training too slow** → Use GPU or reduce model size

**Out of memory** → Reduce batch_size in `training/train.py`

---

## ✅ Success Checklist

- [x] Vocabulary merged (662 symbols)
- [ ] GPU support working (optional but recommended)
- [ ] Test dataset generated
- [ ] Model trained (even tiny model on CPU counts!)
- [ ] Evaluation metrics computed
- [ ] Sample generations reviewed

---

**Ready to start? Run:**

```bash
python3 quickstart.py
```

Choose option 1 to verify everything works! 🚀
