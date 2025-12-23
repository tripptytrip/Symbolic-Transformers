#!/usr/bin/env python3
"""
Complete vocabulary update script - adds all Phase 1 and Phase 2 special tokens.
Run this once to add MESSAGE_START, MESSAGE_END, PAD, UNK, SEP, QUERY, ANSWER, END, UNKNOWN.
"""

import json
import os
from pathlib import Path

def update_vocabulary_complete():
    """Add all special tokens to unified_vocabulary.json in one go."""
    
    vocab_path = Path("unified_vocabulary.json")
    
    if not vocab_path.exists():
        print(f"❌ Error: {vocab_path} not found.")
        print("   Make sure you're running this from the project root directory.")
        return False
    
    # Load existing vocabulary
    print("="*60)
    print("VOCABULARY UPDATE - ADDING ALL SPECIAL TOKENS")
    print("="*60)
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"\nCurrent vocab size: {vocab['vocab_size']}")
    print(f"Numeral range: {vocab['numeral_range']}")
    print(f"Symbol range: {vocab['symbol_range']}")
    
    # Define all special tokens
    # Phase 1: Message boundaries and padding
    # Phase 2: Reasoning task structure
    special_tokens_to_add = {
        # Phase 1 - Core special tokens
        "MESSAGE_START": {
            "description": "Marks beginning of formula",
            "phase": 1,
            "type": "special"
        },
        "MESSAGE_END": {
            "description": "Marks end of complete formula",
            "phase": 1,
            "type": "special"
        },
        "PAD": {
            "description": "Padding token for batching",
            "phase": 1,
            "type": "special"
        },
        "UNK": {
            "description": "Unknown token fallback",
            "phase": 1,
            "type": "special"
        },
        # Phase 2 - Reasoning task tokens
        "SEP": {
            "description": "Separates premises in reasoning tasks",
            "phase": 2,
            "type": "special"
        },
        "QUERY": {
            "description": "Marks start of query in reasoning tasks",
            "phase": 2,
            "type": "special"
        },
        "ANSWER": {
            "description": "Marks start of answer in reasoning tasks",
            "phase": 2,
            "type": "special"
        },
        "END": {
            "description": "Marks end of reasoning sequence",
            "phase": 2,
            "type": "special"
        },
        "UNKNOWN": {
            "description": "Placeholder for unanswerable queries",
            "phase": 2,
            "type": "special"
        }
    }
    
    # Track what we're adding
    added_tokens = []
    next_id = vocab['vocab_size']
    
    # Add tokens that don't already exist
    print("\nAdding tokens:")
    for token_name, token_info in special_tokens_to_add.items():
        if token_name in vocab['label_to_id']:
            print(f"  ⚠ Skipping {token_name} (already exists at ID {vocab['label_to_id'][token_name]})")
        else:
            print(f"  + Adding {token_name} at ID {next_id} (Phase {token_info['phase']})")
            
            # Add to label mappings
            vocab['label_to_id'][token_name] = next_id
            vocab['id_to_label'][str(next_id)] = token_name
            
            # Add to tokens array with metadata
            vocab['tokens'].append({
                'id': next_id,
                'type': token_info['type'],
                'label': token_name,
                'description': token_info['description'],
                'phase': token_info['phase'],
                'filename': None,
                'path': None
            })
            
            added_tokens.append((token_name, next_id))
            next_id += 1
    
    if not added_tokens:
        print("\n✓ Vocabulary already up to date - no tokens added.")
        return True
    
    # Update vocab size
    old_size = vocab['vocab_size']
    vocab['vocab_size'] = next_id
    
    # Add/update special_tokens dictionary
    if 'special_tokens' not in vocab:
        vocab['special_tokens'] = {}
    
    # Add Phase 1 tokens to special_tokens dict (for easy access)
    phase1_start = None
    phase1_end = None
    for token_name, token_id in added_tokens:
        if special_tokens_to_add[token_name]['phase'] == 1:
            vocab['special_tokens'][token_name] = token_id
            if phase1_start is None:
                phase1_start = token_id
            phase1_end = token_id
    
    # Add special_token_range if we added Phase 1 tokens
    if phase1_start is not None:
        # Expand range if it already exists, otherwise create new
        if 'special_token_range' in vocab:
            existing_start, existing_end = vocab['special_token_range']
            vocab['special_token_range'] = [
                min(existing_start, phase1_start),
                max(existing_end, phase1_end)
            ]
        else:
            vocab['special_token_range'] = [phase1_start, phase1_end]
    
    # Save updated vocabulary
    print(f"\nUpdating vocabulary file...")
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("UPDATE COMPLETE")
    print("="*60)
    print(f"Vocab size: {old_size} → {vocab['vocab_size']} (+{len(added_tokens)})")
    print(f"\nAdded tokens:")
    for token_name, token_id in added_tokens:
        phase = special_tokens_to_add[token_name]['phase']
        print(f"  [{token_id}] {token_name} (Phase {phase})")
    
    if 'special_token_range' in vocab:
        print(f"\nSpecial token range: {vocab['special_token_range']}")
    
    print(f"\n✓ Updated {vocab_path}")
    print("\nNext steps:")
    print("  1. Test vocabulary: python utils/vocabulary.py")
    print("  2. Update data generators to use MESSAGE_START/MESSAGE_END")
    print("  3. Retrain model with new vocabulary")
    
    return True


if __name__ == "__main__":
    success = update_vocabulary_complete()
    exit(0 if success else 1)