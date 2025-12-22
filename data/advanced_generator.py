#!/usr/bin/env python3
"""
Advanced FOL Formula Generator (Depth-Balanced Edition)
replaces previous generator with a robust, target-depth strategy
to cure "depth blindness" in symbolic transformers.
"""

import random
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.vocabulary import Vocabulary

# Increase recursion limit for deep formula generation
sys.setrecursionlimit(2000)

@dataclass
class FormulaConfig:
    """Configuration for formula generation."""
    min_depth: int = 1
    max_depth: int = 10          # Target max depth
    max_variables: int = 100
    max_predicates: int = 50
    max_functions: int = 4
    max_constants: int = 5
    
    # Structural Toggles
    use_quantifiers: bool = True
    use_connectives: bool = True
    use_equality: bool = True
    
    # Probability tunings
    prob_shadowing: float = 0.05
    prob_vacuous: float = 0.05  

class AdvancedFormulaGenerator:
    """
    Generates FOL formulas with forced depth targeting to ensure
    uniform distribution of recursive structures.
    """
    def __init__(self, vocab: Vocabulary, config: FormulaConfig, seed: Optional[int] = None):
        self.vocab = vocab
        self.config = config
        
        if seed is not None:
            random.seed(seed)
            
        # Cache token IDs
        self.tokens = {
            'LPAREN': vocab.encode_label('LPAREN'),
            'RPAREN': vocab.encode_label('RPAREN'),
            'COMMA': vocab.encode_label('COMMA'),
            'NOT': vocab.encode_label('NOT'),
            'AND': vocab.encode_label('AND'),
            'OR': vocab.encode_label('OR'),
            'IMPLIES': vocab.encode_label('IMPLIES'),
            'IFF': vocab.encode_label('IFF'),
            'FORALL': vocab.encode_label('FORALL'),
            'EXISTS': vocab.encode_label('EXISTS'),
            'EQUALS': vocab.encode_label('EQUALS'),
            'NOT_EQUALS': vocab.encode_label('NOT_EQUALS'),
        }

        # Capability checks
        self.has_funcs = 'FUNC' in vocab.compositional_tokens
        self.has_consts = 'CONST' in vocab.compositional_tokens
        
        # Initialize Fixed Signatures (Arity Maps)
        self.pred_arities = {i: random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0] 
                             for i in range(config.max_predicates)}
        
        self.func_arities = {i: random.choices([1, 2], weights=[0.8, 0.2])[0] 
                             for i in range(config.max_functions)}
        
    def get_signature_info(self) -> Dict:
        """Return arity maps for logging."""
        return {
            'predicate_arities': self.pred_arities,
            'function_arities': self.func_arities,
        }

    def _generate_variable(self, bound_vars: List[int]) -> List[int]:
        if bound_vars and random.random() < 0.90:
            return self.vocab.encode_compositional('VAR', random.choice(bound_vars))
        return self.vocab.encode_compositional('VAR', random.randint(0, self.config.max_variables - 1))

    def _generate_term(self, current_depth: int, max_term_depth: int, bound_vars: List[int]) -> List[int]:
        if current_depth >= max_term_depth or not self.has_funcs or random.random() < 0.4:
            if self.has_consts and random.random() < 0.2:
                idx = random.randint(0, self.config.max_constants - 1)
                return self.vocab.encode_compositional('CONST', idx)
            return self._generate_variable(bound_vars)

        func_idx = random.randint(0, self.config.max_functions - 1)
        arity = self.func_arities[func_idx]
        
        tokens = self.vocab.encode_compositional('FUNC', func_idx)
        tokens.append(self.tokens['LPAREN'])
        for i in range(arity):
            tokens.extend(self._generate_term(current_depth + 1, max_term_depth, bound_vars))
            if i < arity - 1:
                tokens.append(self.tokens['COMMA'])
        tokens.append(self.tokens['RPAREN'])
        return tokens

    def _generate_atomic(self, bound_vars: List[int]) -> List[int]:
        if self.config.use_equality and random.random() < 0.2:
            left = self._generate_term(0, 2, bound_vars)
            op = self.tokens['EQUALS'] if random.random() > 0.1 else self.tokens['NOT_EQUALS']
            right = self._generate_term(0, 2, bound_vars)
            return left + [op] + right

        pred_idx = random.randint(0, self.config.max_predicates - 1)
        arity = self.pred_arities[pred_idx]
        
        tokens = self.vocab.encode_compositional('PRED', pred_idx)
        tokens.append(self.tokens['LPAREN'])
        for i in range(arity):
            tokens.extend(self._generate_term(0, 2, bound_vars))
            if i < arity - 1:
                tokens.append(self.tokens['COMMA'])
        tokens.append(self.tokens['RPAREN'])
        return tokens

    def generate_formula(self, current_depth: int, target_depth: int, bound_vars: List[int]) -> List[int]:
        """Recursive generation attempting to hit target_depth."""
        if current_depth >= target_depth:
            return self._generate_atomic(bound_vars)

        options = ['QUANT', 'BINARY', 'NOT']
        weights = [0.35, 0.50, 0.15]
        choice = random.choices(options, weights=weights)[0]

        if choice == 'NOT':
            tokens = [self.tokens['NOT']]
            tokens.extend(self.generate_formula(current_depth + 1, target_depth, bound_vars))
            return tokens

        elif choice == 'QUANT':
            quant = self.tokens['FORALL'] if random.random() < 0.5 else self.tokens['EXISTS']
            if bound_vars and random.random() < self.config.prob_shadowing:
                var_idx = random.choice(bound_vars)
            else:
                var_idx = random.randint(0, self.config.max_variables - 1)
                
            tokens = [quant] + self.vocab.encode_compositional('VAR', var_idx)
            new_bound = bound_vars
            if random.random() > self.config.prob_vacuous:
                new_bound = bound_vars + [var_idx]
                
            tokens.extend(self.generate_formula(current_depth + 1, target_depth, new_bound))
            return tokens

        elif choice == 'BINARY':
            op = random.choice([self.tokens['AND'], self.tokens['OR'], self.tokens['IMPLIES'], self.tokens['IFF']])
            
            # One branch must handle the depth growth
            deep_branch = self.generate_formula(current_depth + 1, target_depth, bound_vars)
            
            # The other can be random length (up to target)
            other_target = random.randint(current_depth + 1, target_depth)
            other_branch = self.generate_formula(current_depth + 1, other_target, bound_vars)
            
            if random.random() < 0.5:
                left, right = deep_branch, other_branch
            else:
                left, right = other_branch, deep_branch
                
            return [self.tokens['LPAREN']] + left + [op] + right + [self.tokens['RPAREN']]

        return self._generate_atomic(bound_vars)

    def generate_batch(self, n_samples: int, complexity: int = 0) -> List[Dict]:
        """
        Generates a balanced batch.
        Args:
            n_samples: Total samples to generate
            complexity: (Ignored in this version, uses config.max_depth range)
        """
        samples = []
        depth_range = range(self.config.min_depth, self.config.max_depth + 1)
        samples_per_depth = int(n_samples / len(depth_range))
        
        for d in depth_range:
            for _ in range(samples_per_depth):
                tokens = self.generate_formula(0, d, [])
                samples.append({
                    'tokens': tokens,
                    'meta_depth': d
                })
        
        # Fill remainder
        while len(samples) < n_samples:
            d = random.choice(depth_range)
            tokens = self.generate_formula(0, d, [])
            samples.append({'tokens': tokens, 'meta_depth': d})
            
        random.shuffle(samples)
        return samples


class NextSymbolDataset:
    """Wrapper to format data for training."""
    def __init__(self, formulas_with_meta: List[Dict], vocab: Vocabulary, max_seq_len: int = 128):
        self.samples = []
        
        for entry in formulas_with_meta:
            formula = entry['tokens']
            meta_depth = entry['meta_depth']
            
            if len(formula) > max_seq_len + 1:
                continue
                
            for i in range(1, len(formula)):
                context = formula[:i]
                target = formula[i]
                
                self.samples.append({
                    'context': context,
                    'target': target,
                    'depth': meta_depth # Added metadata key
                })
                
    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump({'samples': self.samples}, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")


def generate_advanced_training_data(
    vocab_path: str = "unified_vocabulary.json",
    output_dir: str = "datasets/fol_next_symbol", # Original output dir
    n_train: int = 15000,
    n_val: int = 2000,
    n_test: int = 1000,
    seed: int = 42,
):
    print("="*60)
    print("BALANCED DEPTH FOL GENERATION")
    print("="*60)
    
    try:
        vocab = Vocabulary(vocab_path)
    except FileNotFoundError:
        print(f"❌ Vocabulary not found at {vocab_path}")
        return

    # Use max_depth=10 to force the model to learn counting
    config = FormulaConfig(max_depth=10) 
    generator = AdvancedFormulaGenerator(vocab, config, seed)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate Splits
    print(f"\nGenerating TRAIN ({n_train})...")
    train_raw = generator.generate_batch(n_train)
    train_ds = NextSymbolDataset(train_raw, vocab, max_seq_len=256)
    train_ds.to_json(out_path / "train.json")
    
    print(f"\nGenerating VAL ({n_val})...")
    val_raw = generator.generate_batch(n_val)
    val_ds = NextSymbolDataset(val_raw, vocab, max_seq_len=256)
    val_ds.to_json(out_path / "val.json")
    
    print(f"\nGenerating TEST ({n_test})...")
    test_raw = generator.generate_batch(n_test)
    test_ds = NextSymbolDataset(test_raw, vocab, max_seq_len=256)
    test_ds.to_json(out_path / "test.json")
    
    # Save Metadata
    metadata = {
        'generator': 'balanced_depth',
        'config': asdict(config),
        'signatures': generator.get_signature_info(),
        'n_train': len(train_ds.samples),
        'n_val': len(val_ds.samples)
    }
    with open(out_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\n✓ Generation Complete.")

if __name__ == "__main__":
    generate_advanced_training_data()