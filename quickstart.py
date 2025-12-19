#!/usr/bin/env python3
"""
Quick-start script for FOL Symbolic Transformer.
Handles setup and runs a small test to verify everything works.
"""

import os
import sys
import json
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════════╗2
║                                                           ║
║           FOL SYMBOLIC TRANSFORMER - QUICK START         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")

# Check prerequisites
print("\n🔍 Checking prerequisites...")

# Python version
if sys.version_info < (3, 10):
    print("❌ Python 3.10+ required")
    sys.exit(1)
print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        device_str = "cuda"
    else:
        print("⚠️  No GPU detected - will use CPU (slower but works)")
        print("   For GPU support: pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
        device_str = "cpu"
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch")
    sys.exit(1)

# Check for unified vocabulary
if not Path("unified_vocabulary.json").exists():
    print("\n❌ unified_vocabulary.json not found")
    print("   Run: python3 merge_vocabulary.py")
    sys.exit(1)

print("✅ Unified vocabulary found")

print("\n" + "="*60)
print("✅ All prerequisites met!")
print("="*60)

# Offer options
print("\nWhat would you like to do?")
print("1. Test vocabulary loading (quick)")
print("2. Generate small test dataset (5 min)")
print("3. Train tiny model on CPU (30 min)")
print("4. Full pipeline (requires GPU, several hours)")
print("5. Exit")

choice = input("\nEnter choice (1-5): ").strip()

if choice == "1":
    print("
" + "="*60)
    print("Testing vocabulary loading...")
    print("="*60)
    
    sys.path.append('fol_transformer')
    from utils.vocabulary import Vocabulary
    
    vocab = Vocabulary("unified_vocabulary.json")
    
    # Test encoding
    print("
✅ Testing numeral encoding:")
    for val in [0, 1, 24, 100]:
        token_id = vocab.encode_numeral(val)
        print(f"  {val} → token {token_id}")
    
    print("
✅ Testing symbol encoding:")
    for label in ['FORALL', 'EXISTS', 'AND', 'IMPLIES']:
        token_id = vocab.encode_label(label)
        print(f"  {label} → token {token_id}")
    
    print("
✅ Testing compositional encoding:")
    tokens = vocab.encode_compositional('VAR', 12)
    print(f"  VAR 12 → tokens {tokens}")
    
    print("
✅ Vocabulary system working correctly!")

elif choice == "2":
    print("
" + "="*60)
    print("Generating test dataset...")
    print("="*60)
    
    sys.path.append('fol_transformer')
    from data.dataset_generator import generate_training_data
    
    generate_training_data(
        vocab_path="unified_vocabulary.json",
        output_dir="datasets/fol_next_symbol",
        n_train=1000,   # More formulas (~7k training samples)
        n_val=200,
        n_test=200,
        complexity_distribution=[
            (1, 0.2),  # 20% simple
            (2, 0.4),  # 40% medium
            (3, 0.3),  # 30% complex
            (4, 0.1),  # 10% very complex
        ]
    )
    
    print("
✅ Test dataset generated!")
    print("   Location: datasets/fol_next_symbol/")

elif choice == "3":
    print("
" + "="*60)
    print("Training tiny model on CPU...")
    print("="*60)
    print("⚠️  This will take ~30 minutes on CPU")
    print("   For faster training, use GPU version")
    
    confirm = input("
Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Check dataset exists
    if not Path("datasets/fol_next_symbol/train.json").exists():
        print("
❌ Dataset not found. Run option 2 first.")
        sys.exit(1)

    with open("datasets/fol_next_symbol/train.json", 'r') as f:
        data = json.load(f)
    if len(data.get('samples', [])) > 50000:
        print(f"⚠️  Large dataset detected ({len(data['samples']):,} samples)")
        print("   This will take longer to train.")
        print("   Consider using GPU for faster training.")
    
    sys.path.append('fol_transformer')
    from training.train import main, TrainingConfig
    
    # Override config for CPU test
    import training.train as train_module
    original_config = train_module.TrainingConfig
    
    class CPUTestConfig(original_config):
        def __init__(self, **kwargs):
            # Accept kwargs so train.main() can still pass defaults.
            super().__init__(**kwargs)
            self.model_size = 'tiny'
            self.batch_size = 8
            self.num_epochs = 20
            self.device = 'cpu'
            self.vocab_size = 662
    
    train_module.TrainingConfig = CPUTestConfig
    
    print("
Starting training...")
    main()
    
    print("
✅ Training complete!")
    print("   Checkpoints saved to: checkpoints/")

elif choice == "4":
    print("\n" + "="*60)
    print("Full pipeline requires:")
    print("="*60)
    print("- AMD GPU with ROCm")
    print("- Several hours training time")
    print("- ~10GB disk space for datasets")
    
    if device_str == "cpu":
        print("\n⚠️  No GPU detected!")
        print("   This will be VERY slow on CPU.")
        confirm = input("\nContinue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled. Install GPU support first.")
            sys.exit(0)
    
    print("\nRunning full pipeline...")
    os.system("python3 run_pipeline.py")

else:
    print("\nExiting.")
    sys.exit(0)

print("\n" + "="*60)
print("Next steps:")
print("="*60)
print("- See README.md for full documentation")
print("- See TECHNICAL_ARCHITECTURE.md for details")
print("- Run with GPU for faster training")
print("- Adjust hyperparameters in training/train.py")
print("="*60)
