#!/usr/bin/env python3
"""
Advanced FOL formula generator with richer structure.
CORRECTED: Fixed context length safety check and vocabulary validation.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.vocabulary import Vocabulary


class FormulaConfig:
    """Configuration for formula generation."""
    def __init__(
        self,
        max_depth: int = 4,
        max_variables: int = 100,
        max_predicates: int = 100,
        max_functions: int = 4,
        max_constants: int = 5,
        use_quantifiers: bool = True,
        use_connectives: bool = True,
        use_equality: bool = True,
    ):
        self.max_depth = max_depth
        self.max_variables = max_variables
        self.max_predicates = max_predicates
        self.max_functions = max_functions
        self.max_constants = max_constants
        self.use_quantifiers = use_quantifiers
        self.use_connectives = use_connectives
        self.use_equality = use_equality


class AdvancedFormulaGenerator:
    """
    Generates FOL formulas with functions, fixed signatures, and structural patterns.
    """
    
    def __init__(self, vocab: Vocabulary, config: FormulaConfig, seed: Optional[int] = None):
        self.vocab = vocab
        self.config = config
        
        if seed is not None:
            random.seed(seed)
        
        # Cache common token IDs
        self.lparen = vocab.encode_label('LPAREN')
        self.rparen = vocab.encode_label('RPAREN')
        self.comma = vocab.encode_label('COMMA')

        # Check for optional tokens in vocabulary to prevent crashes
        self.has_funcs = 'FUNC' in vocab.compositional_tokens
        self.has_consts = 'CONST' in vocab.compositional_tokens
        
        if not self.has_funcs and config.max_functions > 0:
            print("⚠️ Warning: FUNC tokens not in vocabulary. Disabling functions.")
            self.config.max_functions = 0
            
        if not self.has_consts and config.max_constants > 0:
            print("⚠️ Warning: CONST tokens not in vocabulary. Disabling constants.")
            self.config.max_constants = 0
        
        # 1. Fixed Signature Consistency
        self.pred_arities = {}
        for i in range(config.max_predicates):
            self.pred_arities[i] = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
        
        self.func_arities = {}
        for i in range(config.max_functions):
            self.func_arities[i] = random.choices([1, 2], weights=[0.7, 0.3])[0]
        
        # Diversity probabilities
        self.prob_use_function = 0.25
        self.prob_vacuous = 0.05
        self.prob_shadowing = 0.05
        self.prob_horn_clause = 0.20
        self.prob_use_constant = 0.15
    
    def generate_formula(self, depth: int = 0, bound_vars: Optional[List[int]] = None) -> List[int]:
        if bound_vars is None:
            bound_vars = []
        
        if depth >= self.config.max_depth:
            return self.generate_atomic(bound_vars)
        
        r = random.random()
        
        if depth == 0 and r < self.prob_horn_clause:
            return self.generate_horn_clause(depth, bound_vars)
        
        choices = ['quantifier', 'binary', 'unary', 'atomic']
        weights = [0.30, 0.40, 0.15, 0.15]
        
        mode = random.choices(choices, weights=weights)[0]
        
        if mode == 'quantifier' and self.config.use_quantifiers:
            return self.generate_quantified(depth, bound_vars)
        elif mode == 'binary' and self.config.use_connectives:
            return self.generate_binary(depth, bound_vars)
        elif mode == 'unary' and self.config.use_connectives:
            return self.generate_unary(depth, bound_vars)
        else:
            return self.generate_atomic(bound_vars)
    
    def generate_term(self, depth: int, bound_vars: List[int]) -> List[int]:
        # Base case: too deep, or random chance, or no functions
        if depth > 1 or random.random() > self.prob_use_function or self.config.max_functions == 0:
            if bound_vars and random.random() > self.prob_use_constant:
                var_idx = random.choice(bound_vars)
                return self.vocab.encode_compositional('VAR', var_idx)
            elif self.config.max_constants > 0:
                const_idx = random.randint(0, self.config.max_constants - 1)
                return self.vocab.encode_compositional('CONST', const_idx)
            else:
                var_idx = random.randint(0, self.config.max_variables - 1)
                return self.vocab.encode_compositional('VAR', var_idx)
        
        func_idx = random.randint(0, self.config.max_functions - 1)
        arity = self.func_arities[func_idx]
        
        tokens = self.vocab.encode_compositional('FUNC', func_idx)
        tokens.append(self.lparen)
        
        for i in range(arity):
            tokens.extend(self.generate_term(depth + 1, bound_vars))
            if i < arity - 1:
                tokens.append(self.comma)
        
        tokens.append(self.rparen)
        return tokens
    
    def generate_atomic(self, bound_vars: List[int]) -> List[int]:
        if self.config.use_equality and random.random() < 0.20:
            tokens = self.generate_term(0, bound_vars)
            op = 'EQUALS' if random.random() < 0.8 else 'NOT_EQUALS'
            tokens.append(self.vocab.encode_label(op))
            tokens.extend(self.generate_term(0, bound_vars))
            return tokens
        else:
            pred_idx = random.randint(0, self.config.max_predicates - 1)
            arity = self.pred_arities[pred_idx]
            
            tokens = self.vocab.encode_compositional('PRED', pred_idx)
            tokens.append(self.lparen)
            
            for i in range(arity):
                tokens.extend(self.generate_term(0, bound_vars))
                if i < arity - 1:
                    tokens.append(self.comma)
            
            tokens.append(self.rparen)
            return tokens
    
    def generate_quantified(self, depth: int, bound_vars: List[int]) -> List[int]:
        quantifier = random.choice(['FORALL', 'EXISTS'])
        
        if bound_vars and random.random() < self.prob_shadowing:
            var_idx = random.choice(bound_vars)
        else:
            var_idx = random.randint(0, self.config.max_variables - 1)
        
        tokens = [self.vocab.encode_label(quantifier)]
        tokens.extend(self.vocab.encode_compositional('VAR', var_idx))
        
        if random.random() < self.prob_vacuous:
            new_bound = bound_vars
        else:
            new_bound = bound_vars + [var_idx]
        
        tokens.extend(self.generate_formula(depth + 1, new_bound))
        return tokens
    
    def generate_horn_clause(self, depth: int, bound_vars: List[int]) -> List[int]:
        num_conditions = random.randint(1, 3)
        tokens = []
        
        if num_conditions > 1:
            tokens.append(self.lparen)
        
        for i in range(num_conditions):
            tokens.extend(self.generate_atomic(bound_vars))
            if i < num_conditions - 1:
                tokens.append(self.vocab.encode_label('AND'))
        
        if num_conditions > 1:
            tokens.append(self.rparen)
        
        tokens.append(self.vocab.encode_label('IMPLIES'))
        tokens.extend(self.generate_atomic(bound_vars))
        
        return tokens
    
    def generate_binary(self, depth: int, bound_vars: List[int]) -> List[int]:
        connective = random.choice(['AND', 'OR', 'IMPLIES', 'IFF'])
        tokens = [self.lparen]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.vocab.encode_label(connective))
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.rparen)
        return tokens
    
    def generate_unary(self, depth: int, bound_vars: List[int]) -> List[int]:
        tokens = [self.vocab.encode_label('NOT')]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        return tokens
    
    def generate_batch(self, n: int, complexity: int = 2) -> List[List[int]]:
        old_depth = self.config.max_depth
        self.config.max_depth = min(complexity + 1, 5)
        
        formulas = []
        for _ in range(n):
            formulas.append(self.generate_formula())
        
        self.config.max_depth = old_depth
        return formulas
    
    def get_signature_info(self) -> Dict:
        return {
            'predicate_arities': self.pred_arities.copy(),
            'function_arities': self.func_arities.copy(),
        }


class NextSymbolDataset:
    """Creates next-symbol prediction samples from formulas."""
    
    def __init__(self, formulas: List[List[int]], vocab: Vocabulary, max_seq_len: int = 128):
        self.vocab = vocab
        self.samples = []
        
        for formula in formulas:
            # SAFETY FIX: Ensure we don't generate samples where context exceeds max_seq_len
            # because train.py will truncate the context but keep the remote target,
            # creating a gap in logic.
            limit = min(len(formula) - 1, max_seq_len)
            
            for i in range(limit):
                prefix = formula[:i + 1]
                target = formula[i + 1]
                self.samples.append({
                    'context': prefix,
                    'target': target
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump({'samples': self.samples}, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")


def generate_advanced_training_data(
    vocab_path: str = "unified_vocabulary.json",
    output_dir: str = "datasets/fol_next_symbol",
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 500,
    seed: int = 42,
):
    print("=" * 60)
    print("ADVANCED FOL DATASET GENERATION")
    print("=" * 60)
    
    try:
        vocab = Vocabulary(vocab_path)
    except FileNotFoundError:
        print(f"❌ Error: Vocabulary file not found at {vocab_path}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = FormulaConfig(
        max_depth=4,
        max_variables=100,
        max_predicates=100,
        max_functions=4,
        max_constants=5,
        use_quantifiers=True,
        use_connectives=True,
        use_equality=True,
    )
    
    generator = AdvancedFormulaGenerator(vocab, config, seed=seed)
    
    print("\nℹ️ Fixed Signatures:")
    sig_info = generator.get_signature_info()
    print(f"   Predicates: {sig_info['predicate_arities']}")
    print(f"   Functions:  {sig_info['function_arities']}")
    
    complexity_dist = {1: 0.20, 2: 0.40, 3: 0.30, 4: 0.10}
    
    def generate_split(n: int, name: str) -> NextSymbolDataset:
        print(f"\nGenerating {name} set ({n} formulas)...")
        all_formulas = []
        
        for complexity, fraction in complexity_dist.items():
            count = int(n * fraction)
            print(f"  Complexity {complexity}: {count} formulas")
            formulas = generator.generate_batch(count, complexity=complexity)
            all_formulas.extend(formulas)
        
        dataset = NextSymbolDataset(all_formulas, vocab, max_seq_len=128)
        return dataset
    
    train_dataset = generate_split(n_train, "train")
    val_dataset = generate_split(n_val, "val")
    test_dataset = generate_split(n_test, "test")
    
    train_dataset.to_json(output_path / "train.json")
    val_dataset.to_json(output_path / "val.json")
    test_dataset.to_json(output_path / "test.json")
    
    metadata = {
        'generator': 'advanced',
        'config': {
            'max_depth': config.max_depth,
            'max_variables': config.max_variables,
            'max_predicates': config.max_predicates,
        },
        'signatures': sig_info,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'seed': seed,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ Advanced dataset generation complete!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Train samples: {len(train_dataset)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_advanced_training_data(
        n_train=10000,
        n_val=1000,
        n_test=1000,
    )