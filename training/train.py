"""
Training script for Symbolic FOL Transformer.
Optimized for AMD Radeon GPU with ROCm backend.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import sys
import argparse
import re

sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import create_model
from utils.vocabulary import Vocabulary


def get_training_config(model_size='tiny', num_epochs=50, batch_size=None):
    """Helper to get training config for different model sizes."""

    # Auto-adjust batch size based on model
    if batch_size is None:
        batch_size = {
            'tiny': 64,
            'small': 32,
            'base': 16,
            'large': 8,
        }[model_size]

    return TrainingConfig(
        model_size=model_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        warmup_steps=2000,
        checkpoint_dir='checkpoints',
    )


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_size: str = 'base'
    vocab_size: int = 663
    
    # Training
    batch_size: int = 64  # Larger batches for larger dataset
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000  # More warmup for larger dataset
    max_grad_norm: float = 1.0
    
    # Data
    max_seq_len: int = 128
    train_data_path: str = "datasets/fol_next_symbol/train.json"
    val_data_path: str = "datasets/fol_next_symbol/val.json"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save every N epochs
    
    # Logging
    log_every: int = 100  # Log every N batches
    
    # Device
    device: str = "cuda"  # Will use ROCm if available
    mixed_precision: bool = False  # AMD GPU may not support all AMP ops


class FOLDataset(Dataset):
    """PyTorch Dataset for FOL next-symbol prediction."""
    
    def __init__(self, data_path: str, max_seq_len: int = 128):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.max_seq_len = max_seq_len
        
        print(f"✓ Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get context and target
        context = sample['context']
        target = sample['target']
        
        # Pad context to max_seq_len
        context_len = len(context)
        if context_len < self.max_seq_len:
            context = context + [0] * (self.max_seq_len - context_len)
        else:
            context = context[:self.max_seq_len]
            context_len = self.max_seq_len
        
        return {
            'input_ids': torch.tensor(context, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'length': torch.tensor(context_len, dtype=torch.long)
        }


class Trainer:
    """Trainer for Symbolic Transformer."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        vocab: Vocabulary
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab = vocab
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(
                (step - config.warmup_steps) / (len(train_loader) * config.num_epochs)
            ) * 3.14159)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Compute loss on last position (next-token prediction)
            loss = self.criterion(logits[:, -1, :], targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target'].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(logits[:, -1, :], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model and optimizer state from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        print(f"✓ Resumed from checkpoint: {checkpoint_path}")

    def train(self, start_epoch: int = 0):
        """Main training loop."""
        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Model: {self.config.model_size}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            
            print(f"Epoch {self.epoch}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} {'[BEST]' if is_best else ''}")
            print(f"  Time:       {epoch_time:.1f}s\n")
            
            # Save checkpoint
            if self.epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best)
        
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final checkpoint: {self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'}")
        print("="*60)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train the Symbolic FOL Transformer")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from checkpoint path or 'latest' in checkpoint_dir"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Total number of epochs to train (overrides config)"
    )
    parser.add_argument(
        "--model-size",
        default="tiny",
        choices=["tiny", "small", "base", "large"],
        help="Model size preset"
    )
    args = parser.parse_args()

    def resolve_checkpoint_path(resume_arg: str, checkpoint_dir: Path) -> str:
        if resume_arg == "latest":
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            def checkpoint_epoch(path: Path) -> int:
                match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
                return int(match.group(1)) if match else -1
            latest_checkpoint = max(checkpoints, key=checkpoint_epoch)
            return str(latest_checkpoint)
        return resume_arg

    # Configuration
    config = get_training_config(
        model_size=args.model_size,
        num_epochs=args.num_epochs or 50,
    )
    
    # Load vocabulary
    vocab_path = "unified_vocabulary.json"
    vocab = Vocabulary(vocab_path)
    config.vocab_size = vocab.vocab_size
    
    print(f"\n✓ Loaded vocabulary: {vocab.vocab_size} tokens")
    
    resume_path: Optional[str] = None
    if args.resume:
        resume_path = resolve_checkpoint_path(args.resume, Path(config.checkpoint_dir))
        checkpoint_meta = torch.load(resume_path, map_location="cpu")
        ckpt_config = checkpoint_meta.get('config', {})
        if ckpt_config.get('model_size'):
            config.model_size = ckpt_config['model_size']
        if ckpt_config.get('vocab_size'):
            config.vocab_size = ckpt_config['vocab_size']
        if ckpt_config.get('max_seq_len'):
            config.max_seq_len = ckpt_config['max_seq_len']
        if ckpt_config.get('num_epochs') and args.num_epochs is None:
            config.num_epochs = ckpt_config['num_epochs']

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = FOLDataset(config.train_data_path, config.max_seq_len)
    val_dataset = FOLDataset(config.val_data_path, config.max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config.vocab_size, config.model_size)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, vocab)

    start_epoch = 0
    if resume_path:
        trainer.load_checkpoint(resume_path)
        start_epoch = trainer.epoch

    # Train
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
