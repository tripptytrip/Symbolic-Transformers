#!/usr/bin/env python3
"""
Advanced FOL formula generator with richer structure.

Improvements over basic generator:
1. Function Symbols: P(f(x), g(y)) instead of just P(x, y)
2. Fixed Signatures: PRED_1 is *always* arity 2 (model must learn consistency)
3. Structural Archetypes: Horn Clauses (A ∧ B ∧ C) → D
4. Vacuous Quantification: ∀x P(y) (unused variable)
5. Variable Shadowing: ∀x (∃x ...) (nested scope)
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
        max_variables: int = 8,
        max_predicates: int = 10,
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
        
        # 1. Fixed Signature Consistency
        # Each predicate/function has a fixed arity across ALL formulas
        self.pred_arities = {}
        for i in range(config.max_predicates):
            # Weighted: mostly 1 or 2, rarely 3
            self.pred_arities[i] = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
        
        self.func_arities = {}
        for i in range(config.max_functions):
            # Functions usually unary or binary
            self.func_arities[i] = random.choices([1, 2], weights=[0.7, 0.3])[0]
        
        # Diversity probabilities
        self.prob_use_function = 0.25      # Chance to use f(x) instead of x
        self.prob_vacuous = 0.05           # Chance to quantify but not use var
        self.prob_shadowing = 0.05         # Chance to reuse var name
        self.prob_horn_clause = 0.20       # Chance to generate Horn clause
        self.prob_use_constant = 0.15      # Chance to use constant instead of var
    
    def generate_formula(self, depth: int = 0, bound_vars: Optional[List[int]] = None) -> List[int]:
        """Generate a complete FOL formula as token IDs."""
        if bound_vars is None:
            bound_vars = []
        
        # Stop condition: max depth reached
        if depth >= self.config.max_depth:
            return self.generate_atomic(bound_vars)
        
        r = random.random()
        
        # Special structures at top level
        if depth == 0 and r < self.prob_horn_clause:
            return self.generate_horn_clause(depth, bound_vars)
        
        # Standard recursive generation
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
        """
        Generate a term: variable, constant, or function application.
        Recursive for nested functions like f(g(x), y).
        """
        # Base case: too deep, or random chance, or no functions
        if depth > 1 or random.random() > self.prob_use_function or self.config.max_functions == 0:
            # Return variable or constant
            if random.random() < self.prob_use_constant and self.config.max_constants > 0:
                # Constant
                const_idx = random.randint(0, self.config.max_constants - 1)
                return self.vocab.encode_compositional('CONST', const_idx)
            else:
                # Variable
                if bound_vars:
                    var_idx = random.choice(bound_vars)
                else:
                    var_idx = random.randint(0, self.config.max_variables - 1)
                return self.vocab.encode_compositional('VAR', var_idx)
        
        # Recursive: function application
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
        """Generate atomic formula: PRED(t1, t2...) or t1 = t2."""
        if self.config.use_equality and random.random() < 0.20:
            # Equality: f(x) = g(y)
            tokens = self.generate_term(0, bound_vars)
            op = 'EQUALS' if random.random() < 0.8 else 'NOT_EQUALS'
            tokens.append(self.vocab.encode_label(op))
            tokens.extend(self.generate_term(0, bound_vars))
            return tokens
        else:
            # Predicate with fixed arity
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
        """Generate quantified formula with optional shadowing/vacuous quantification."""
        quantifier = random.choice(['FORALL', 'EXISTS'])
        
        # Variable selection (with possible shadowing)
        if bound_vars and random.random() < self.prob_shadowing:
            var_idx = random.choice(bound_vars)  # Reuse existing var
        else:
            var_idx = random.randint(0, self.config.max_variables - 1)
        
        tokens = [self.vocab.encode_label(quantifier)]
        tokens.extend(self.vocab.encode_compositional('VAR', var_idx))
        
        # Vacuous quantification: don't add to bound_vars
        if random.random() < self.prob_vacuous:
            new_bound = bound_vars  # Body might not use this var
        else:
            new_bound = bound_vars + [var_idx]
        
        tokens.extend(self.generate_formula(depth + 1, new_bound))
        return tokens
    
    def generate_horn_clause(self, depth: int, bound_vars: List[int]) -> List[int]:
        """
        Generate Horn-like clause: (A ∧ B ∧ C) → D
        Common in logic programming (Prolog-style rules).
        """
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
        """Generate binary connective formula."""
        connective = random.choice(['AND', 'OR', 'IMPLIES', 'IFF'])
        tokens = [self.lparen]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.vocab.encode_label(connective))
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.rparen)
        return tokens
    
    def generate_unary(self, depth: int, bound_vars: List[int]) -> List[int]:
        """Generate negated formula."""
        tokens = [self.vocab.encode_label('NOT')]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        return tokens
    
    def generate_batch(self, n: int, complexity: int = 2) -> List[List[int]]:
        """Generate n formulas with given complexity (max_depth)."""
        old_depth = self.config.max_depth
        self.config.max_depth = min(complexity + 1, 5)
        
        formulas = []
        for _ in range(n):
            formulas.append(self.generate_formula())
        
        self.config.max_depth = old_depth
        return formulas
    
    def get_signature_info(self) -> Dict:
        """Return the fixed signatures for documentation/debugging."""
        return {
            'predicate_arities': self.pred_arities.copy(),
            'function_arities': self.func_arities.copy(),
        }


class NextSymbolDataset:
    """Creates next-symbol prediction samples from formulas."""
    
    def __init__(self, formulas: List[List[int]], vocab: Vocabulary):
        self.vocab = vocab
        self.samples = []
        
        for formula in formulas:
            # Create (prefix, next_token) pairs
            for i in range(len(formula) - 1):
                prefix = formula[:i + 1]
                target = formula[i + 1]
                self.samples.append({
                    'input': prefix,
                    'target': target
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def to_json(self, path: str):
        """Save samples to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.samples, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")


def generate_advanced_training_data(
    vocab_path: str = "unified_vocabulary.json",
    output_dir: str = "datasets/fol_next_symbol",
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 500,
    seed: int = 42,
):
    """
    Generate training data using the advanced formula generator.
    """
    print("=" * 60)
    print("ADVANCED FOL DATASET GENERATION")
    print("=" * 60)
    
    # Load vocabulary
    vocab = Vocabulary(vocab_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = FormulaConfig(
        max_depth=4,
        max_variables=8,
        max_predicates=10,
        max_functions=4,
        max_constants=5,
        use_quantifiers=True,
        use_connectives=True,
        use_equality=True,
    )
    
    # Create generator with fixed seed for reproducibility
    generator = AdvancedFormulaGenerator(vocab, config, seed=seed)
    
    # Print signature info
    print("\n📋 Fixed Signatures (consistent across all formulas):")
    sig_info = generator.get_signature_info()
    print(f"   Predicates: {sig_info['predicate_arities']}")
    print(f"   Functions:  {sig_info['function_arities']}")
    
    # Complexity distribution (same as basic generator)
    complexity_dist = {1: 0.20, 2: 0.40, 3: 0.30, 4: 0.10}
    
    def generate_split(n: int, name: str) -> List[Dict]:
        print(f"\nGenerating {name} set ({n} formulas)...")
        all_formulas = []
        
        for complexity, fraction in complexity_dist.items():
            count = int(n * fraction)
            print(f"  Complexity {complexity}: {count} formulas")
            formulas = generator.generate_batch(count, complexity=complexity)
            all_formulas.extend(formulas)
        
        # Create dataset
        dataset = NextSymbolDataset(all_formulas, vocab)
        return dataset.samples
    
    # Generate splits
    train_samples = generate_split(n_train, "train")
    val_samples = generate_split(n_val, "val")
    test_samples = generate_split(n_test, "test")
    
    # Save to JSON (wrapped in dict for train.py compatibility)
    with open(output_path / "train.json", 'w') as f:
        json.dump({'samples': train_samples}, f)
    print(f"✓ Saved {len(train_samples)} samples to {output_path}/train.json")
    
    with open(output_path / "val.json", 'w') as f:
        json.dump({'samples': val_samples}, f)
    print(f"✓ Saved {len(val_samples)} samples to {output_path}/val.json")
    
    with open(output_path / "test.json", 'w') as f:
        json.dump({'samples': test_samples}, f)
    print(f"✓ Saved {len(test_samples)} samples to {output_path}/test.json")
    
    # Save metadata
    metadata = {
        'generator': 'advanced',
        'config': {
            'max_depth': config.max_depth,
            'max_variables': config.max_variables,
            'max_predicates': config.max_predicates,
            'max_functions': config.max_functions,
            'max_constants': config.max_constants,
        },
        'signatures': sig_info,
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test': len(test_samples),
        'seed': seed,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ Advanced dataset generation complete!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Train samples: {len(train_samples)}")
    print(f"✓ Val samples: {len(val_samples)}")
    print(f"✓ Test samples: {len(test_samples)}")
    print("=" * 60)


if __name__ == "__main__":
    # Test generation
    generate_advanced_training_data(
        n_train=1000,
        n_val=100,
        n_test=50,
    )
