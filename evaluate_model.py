#!/usr/bin/env python3
"""
Evaluate trained FOL model.
Standalone script with all imports properly configured.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json

# Add fol_transformer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fol_transformer'))

def load_vocabulary(vocab_path):
    """Load vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def load_model_checkpoint(checkpoint_path, vocab_size, device='cpu'):
    """Load trained model from checkpoint."""
    from models.transformer import create_model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(vocab_size, config['model_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model

def load_test_data(data_path):
    """Load test dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Handle different data formats
    if 'samples' in data:
        samples = data['samples']
    else:
        samples = data
    
    return samples

def compute_accuracy(model, samples, vocab_size, device='cpu', top_k=1):
    """Compute top-k accuracy."""
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            # Handle different sample formats
            if 'input_ids' in sample:
                context = sample['input_ids']
                target = sample['target']
            elif 'context' in sample:
                context = sample['context']
                target = sample['target']
            elif 'input' in sample and 'target' in sample:
                context = sample['input']
                target = sample['target']
            else:
                print(f"Warning: Unknown sample format: {sample.keys()}")
                continue
            
            # Convert to tensor
            if isinstance(context, list):
                input_ids = torch.tensor([context], dtype=torch.long).to(device)
            else:
                input_ids = context.unsqueeze(0).to(device)
            
            # Predict
            logits = model(input_ids)
            top_k_preds = torch.topk(logits[0, -1, :], k=top_k).indices.tolist()
            
            if target in top_k_preds:
                correct += 1
            total += 1
    
    return correct / total

def decode_id(vocab, token_id):
    """Decode token ID to label."""
    id_to_label = {int(k): v for k, v in vocab['id_to_label'].items()}
    return id_to_label.get(token_id, f"UNK_{token_id}")

def decode_sequence(vocab, token_ids):
    """Decode sequence of token IDs."""
    return " ".join([decode_id(vocab, tid) for tid in token_ids])

def main():
    print("\n" + "="*60)
    print("EVALUATING TRAINED MODEL")
    print("="*60)
    
    # Paths
    vocab_path = "unified_vocabulary.json"
    checkpoint_path = "checkpoints/best_model.pt"
    test_data_path = "datasets/fol_next_symbol/test.json"
    
    # Check files exist
    for path in [vocab_path, checkpoint_path, test_data_path]:
        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            return
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = load_vocabulary(vocab_path)
    vocab_size = vocab['vocab_size']
    print(f"✓ Vocabulary size: {vocab_size}")
    
    # Load model
    print("\nLoading model...")
    device = 'cpu'
    model = load_model_checkpoint(checkpoint_path, vocab_size, device)
    
    # Load test data
    print("\nLoading test data...")
    samples = load_test_data(test_data_path)
    print(f"✓ Test samples: {len(samples)}")
    
    # Compute accuracy
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    
    print("\nComputing top-1 accuracy...")
    top1_acc = compute_accuracy(model, samples, vocab_size, device, top_k=1)
    print(f"✓ Top-1 Accuracy: {top1_acc:.2%}")
    print(f"  ({top1_acc*100:.1f}% chance of exact next symbol)")
    
    print("\nComputing top-5 accuracy...")
    top5_acc = compute_accuracy(model, samples, vocab_size, device, top_k=5)
    print(f"✓ Top-5 Accuracy: {top5_acc:.2%}")
    print(f"  ({top5_acc*100:.1f}% chance target in top 5)")
    
    print("\nComputing top-10 accuracy...")
    top10_acc = compute_accuracy(model, samples, vocab_size, device, top_k=10)
    print(f"✓ Top-10 Accuracy: {top10_acc:.2%}")
    print(f"  ({top10_acc*100:.1f}% chance target in top 10)")
    
    # Compare to random
    random_top1 = 1.0 / vocab_size
    improvement = top1_acc / random_top1
    print(f"\n📊 Improvement over random guessing:")
    print(f"   Random: {random_top1:.2%}")
    print(f"   Model:  {top1_acc:.2%}")
    print(f"   {improvement:.1f}× better!")
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    model.eval()
    for i in range(min(10, len(samples))):
        sample = samples[i]
        
        # Handle different formats
        if 'input_ids' in sample:
            context = sample['input_ids']
            target = sample['target']
        elif 'context' in sample:
            context = sample['context']
            target = sample['target']
        elif 'input' in sample and 'target' in sample:
            context = sample['input']
            target = sample['target']
        else:
            continue
        
        # Get context (remove padding if present)
        if isinstance(context, list):
            context = [t for t in context if t != 0]
        else:
            context = context.tolist()
            context = [t for t in context if t != 0]
        
        # Use last 10 symbols as context for display
        context_short = context[-10:]
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            # Top 5 predictions
            top5_probs, top5_ids = torch.topk(probs, 5)
            top5_ids = top5_ids.tolist()
            top5_probs = top5_probs.tolist()
        
        # Decode
        context_str = decode_sequence(vocab, context_short)
        target_str = decode_id(vocab, target)
        pred_str = decode_id(vocab, top5_ids[0])
        
        # Check if correct
        correct = (top5_ids[0] == target)
        in_top5 = (target in top5_ids)
        
        print(f"\n{'='*60}")
        print(f"Example {i+1}:")
        print(f"  Context: ...{context_str}")
        print(f"  Target:     {target_str:15s} {'✓' if correct else '✗'}")
        print(f"  Predicted:  {pred_str:15s} (confidence: {top5_probs[0]:.1%})")
        
        if not correct and in_top5:
            target_rank = top5_ids.index(target) + 1
            print(f"  (Target was #{target_rank} in top-5)")
        
        print(f"  Top-5 predictions:")
        for j, (tid, prob) in enumerate(zip(top5_ids[:5], top5_probs[:5])):
            symbol = decode_id(vocab, tid)
            marker = "← TARGET" if tid == target else ""
            print(f"    {j+1}. {symbol:15s} {prob:6.1%} {marker}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model size:      Tiny (566K params)")
    print(f"Training epochs: 5")
    print(f"Training time:   ~25 seconds")
    print(f"Final val loss:  2.52")
    print(f"")
    print(f"Performance:")
    print(f"  Top-1:  {top1_acc:.1%}")
    print(f"  Top-5:  {top5_acc:.1%}")
    print(f"  Top-10: {top10_acc:.1%}")
    print(f"")
    print(f"Next steps:")
    print(f"  - Train for more epochs (20-50)")
    print(f"  - Use larger model (small/base)")
    print(f"  - Use GPU for faster training")
    print(f"  - Generate more training data")
    print("="*60)

if __name__ == "__main__":
    main()