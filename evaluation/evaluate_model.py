#!/usr/bin/env python3
"""
Evaluate trained FOL model with batch processing.
Optimized for speed using DataLoader.
CORRECTED: Now handles padding properly by using sequence lengths.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json
import argparse
from tqdm import tqdm

# Add fol_transformer to path to ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fol_transformer'))

class FOLDataset(Dataset):
    """Dataset for FOL formulas."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if 'input_ids' in sample:
            context = sample['input_ids']
            target = sample['target']
        elif 'context' in sample:
            context = sample['context']
            target = sample['target']
        elif 'input' in sample:
            context = sample['input']
            target = sample['target']
        else:
            context = []
            target = 0
        
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def collate_fn(batch):
    """Pad sequences in batch and return lengths."""
    contexts, targets = zip(*batch)
    
    # CRITICAL FIX 1: Capture lengths before padding
    lengths = torch.tensor([len(x) for x in contexts], dtype=torch.long)
    
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    
    # CRITICAL FIX 2: Return lengths
    return padded_contexts, targets, lengths

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def load_model_checkpoint(checkpoint_path, vocab_size, device='cpu'):
    # Import here to avoid issues if paths aren't set up at module level
    from models.transformer import create_model
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(vocab_size, config['model_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"✓ Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model

def load_test_data(data_path):
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if 'samples' in data:
        samples = data['samples']
    else:
        samples = data
    
    return samples

def evaluate(model, dataloader, vocab_size, device):
    """Evaluate model accuracy."""
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        # CRITICAL FIX 3: Unpack lengths
        for contexts, targets, lengths in tqdm(dataloader, desc="Evaluating"):
            contexts = contexts.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            logits = model(contexts) # [batch, max_len, vocab]
            
            # CRITICAL FIX 4: Select logits at the last REAL token position
            # shape: [batch_size]
            batch_indices = torch.arange(contexts.size(0), device=device)
            # We want the logit at index (length - 1)
            next_token_logits = logits[batch_indices, lengths - 1, :]
            
            # Top-k predictions
            _, top10_preds = torch.topk(next_token_logits, k=10, dim=1)
            
            # Check accuracy
            targets_expanded = targets.unsqueeze(1)
            matches = (top10_preds == targets_expanded)
            
            correct_top1 += matches[:, 0].sum().item()
            correct_top5 += matches[:, :5].any(dim=1).sum().item()
            correct_top10 += matches[:, :10].any(dim=1).sum().item()
            total += targets.size(0)
            
    return {
        'top1': correct_top1 / total,
        'top5': correct_top5 / total,
        'top10': correct_top10 / total,
        'total': total
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate FOL Transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--vocab', type=str, default='unified_vocabulary.json', help='Path to vocabulary')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Check files
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    if not os.path.exists(args.test_data):
        print(f"Error: Test data not found at {args.test_data}")
        return
        
    # Load vocab
    vocab = load_vocabulary(args.vocab)
    vocab_size = vocab['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    
    # Load data
    samples = load_test_data(args.test_data)
    dataset = FOLDataset(samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2 if use_cuda else 0
    )
    print(f"Test samples: {len(samples)}")
    
    # Load model
    model = load_model_checkpoint(args.checkpoint, vocab_size, device)
    
    # Evaluate
    print("\n" + "="*50)
    print("STARTING EVALUATION")
    print("="*50)
    
    metrics = evaluate(model, dataloader, vocab_size, device)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy:  {metrics['top1']:.2%} ({metrics['top1']*100:.1f}%)")
    print(f"Top-5 Accuracy:  {metrics['top5']:.2%} ({metrics['top5']*100:.1f}%)")
    print(f"Top-10 Accuracy: {metrics['top10']:.2%} ({metrics['top10']*100:.1f}%)")
    
    # Random baseline comparison
    random_acc = 1.0 / vocab_size
    improvement = metrics['top1'] / random_acc
    print(f"\nImprovement over random ({random_acc:.2%}): {improvement:.1f}x")

if __name__ == '__main__':
    main()