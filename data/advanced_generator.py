#!/usr/bin/env python3
"""
Advanced FOL Formula Generator (Depth-Balanced Edition v2)
Updated drop-in replacement with configurable term depth, 
smart variable locality, and improved recursion safety.
"""

import random
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from utils.vocabulary import Vocabulary
except ImportError:
    # Fallback for standalone testing if utils aren't present
    class Vocabulary:
        def __init__(self, path): self.compositional_tokens = {'VAR', 'FUNC', 'CONST', 'PRED'}
        def encode_label(self, label): return label
        def encode_compositional(self, kind, idx): return [f"{kind}_{idx}"]

# Increase recursion limit for deep formula generation
sys.setrecursionlimit(5000)

@dataclass
class FormulaConfig:
    """Configuration for formula generation."""
    min_depth: int = 1
    max_depth: int = 12          # Increased default max depth
    max_term_depth: int = 3      # New: Controls depth of f(g(x)) trees
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
    prob_local_var: float = 0.70 # Favor recently bound variables

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
        self.has_funcs = hasattr(vocab, 'compositional_tokens') and 'FUNC' in vocab.compositional_tokens
        self.has_consts = hasattr(vocab, 'compositional_tokens') and 'CONST' in vocab.compositional_tokens
        
        # Initialize Fixed Signatures (Arity Maps)
        # Predicates: biased slightly towards arity 1 and 2
        self.pred_arities = {i: random.choices([1, 2, 3], weights=[0.45, 0.45, 0.1])[0] 
                             for i in range(config.max_predicates)}
        
        # Functions: biased towards arity 1 (unary) to prevent exponential explosion
        self.func_arities = {i: random.choices([1, 2], weights=[0.85, 0.15])[0] 
                             for i in range(config.max_functions)}
        
    def get_signature_info(self) -> Dict:
        """Return arity maps for logging."""
        return {
            'predicate_arities': self.pred_arities,
            'function_arities': self.func_arities,
        }

    def _generate_variable(self, bound_vars: List[int]) -> List[int]:
        """Selects a variable with a bias towards locally bound ones."""
        if bound_vars:
            # 70% chance to pick from the most recent 3 variables (Locality Bias)
            if random.random() < self.config.prob_local_var:
                # Slice the last 3 variables, or fewer if not enough exist
                recent_window = bound_vars[-min(len(bound_vars), 3):]
                var_idx = random.choice(recent_window)
            else:
                var_idx = random.choice(bound_vars)
            return self.vocab.encode_compositional('VAR', var_idx)
        
        # Fallback for free variables
        return self.vocab.encode_compositional('VAR', random.randint(0, self.config.max_variables - 1))

    def _generate_term(self, current_depth: int, max_depth: int, bound_vars: List[int]) -> List[int]:
        """
        Recursively generates a term (variable, constant, or function).
        Respects max_depth to prevent term trees from becoming larger than formula trees.
        """
        # Base case: Hit depth limit, or random chance to stop, or no functions available
        if current_depth >= max_depth or not self.has_funcs or random.random() < 0.4:
            if self.has_consts and random.random() < 0.2:
                idx = random.randint(0, self.config.max_constants - 1)
                return self.vocab.encode_compositional('CONST', idx)
            return self._generate_variable(bound_vars)

        # Recursive case: Function application
        func_idx = random.randint(0, self.config.max_functions - 1)
        arity = self.func_arities[func_idx]
        
        tokens = self.vocab.encode_compositional('FUNC', func_idx)
        tokens.append(self.tokens['LPAREN'])
        for i in range(arity):
            # Recurse with incremented depth
            tokens.extend(self._generate_term(current_depth + 1, max_depth, bound_vars))
            if i < arity - 1:
                tokens.append(self.tokens['COMMA'])
        tokens.append(self.tokens['RPAREN'])
        return tokens

    def _generate_atomic(self, bound_vars: List[int]) -> List[int]:
        """Generates an atomic formula (Equality or Predicate)."""
        # Use configured max_term_depth instead of hardcoded 2
        term_depth_limit = self.config.max_term_depth

        if self.config.use_equality and random.random() < 0.25:
            left = self._generate_term(0, term_depth_limit, bound_vars)
            op = self.tokens['EQUALS'] if random.random() > 0.1 else self.tokens['NOT_EQUALS']
            right = self._generate_term(0, term_depth_limit, bound_vars)
            return left + [op] + right

        pred_idx = random.randint(0, self.config.max_predicates - 1)
        arity = self.pred_arities[pred_idx]
        
        tokens = self.vocab.encode_compositional('PRED', pred_idx)
        tokens.append(self.tokens['LPAREN'])
        for i in range(arity):
            tokens.extend(self._generate_term(0, term_depth_limit, bound_vars))
            if i < arity - 1:
                tokens.append(self.tokens['COMMA'])
        tokens.append(self.tokens['RPAREN'])
        return tokens

    def generate_formula(self, current_depth: int, target_depth: int, bound_vars: List[int]) -> List[int]:
        """
        Recursive generation attempting to hit target_depth exactly.
        Ensures at least one branch of the tree reaches the target.
        """
        # Base case: We reached the target depth, must collapse to an atom
        if current_depth >= target_depth:
            return self._generate_atomic(bound_vars)

        # Recursive step: Choose a logical operator
        options = []
        weights = []
        
        if self.config.use_connectives:
            options.append('BINARY')
            weights.append(0.50)
            options.append('NOT')
            weights.append(0.15)
        
        if self.config.use_quantifiers:
            options.append('QUANT')
            weights.append(0.35)

        # Normalize weights if some options are disabled
        if not options:
            return self._generate_atomic(bound_vars)
            
        choice = random.choices(options, weights=weights, k=1)[0]

        if choice == 'NOT':
            tokens = [self.tokens['NOT']]
            # Unary operator must carry the depth burden
            tokens.extend(self.generate_formula(current_depth + 1, target_depth, bound_vars))
            return tokens

        elif choice == 'QUANT':
            quant = self.tokens['FORALL'] if random.random() < 0.5 else self.tokens['EXISTS']
            
            # Variable handling (Shadowing vs New)
            if bound_vars and random.random() < self.config.prob_shadowing:
                var_idx = random.choice(bound_vars)
            else:
                var_idx = random.randint(0, self.config.max_variables - 1)
                
            tokens = [quant] + self.vocab.encode_compositional('VAR', var_idx)
            
            # Vacuous Quantification check
            new_bound = bound_vars
            if random.random() > self.config.prob_vacuous:
                new_bound = bound_vars + [var_idx]
                
            # Quantifier must carry the depth burden
            tokens.extend(self.generate_formula(current_depth + 1, target_depth, new_bound))
            return tokens

        elif choice == 'BINARY':
            op = random.choice([self.tokens['AND'], self.tokens['OR'], self.tokens['IMPLIES'], self.tokens['IFF']])
            
            # Critical Logic: One branch MUST be deep, the other CAN be shallow.
            # This ensures we hit target_depth without exceeding it.
            
            # Branch 1: Forced to go to target
            deep_branch = self.generate_formula(current_depth + 1, target_depth, bound_vars)
            
            # Branch 2: Can stop anywhere between current+1 and target
            # We skew this slightly so the tree isn't always perfectly balanced (more natural)
            other_target = random.randint(current_depth + 1, target_depth)
            other_branch = self.generate_formula(current_depth + 1, other_target, bound_vars)
            
            # Randomize left/right position of the deep branch
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
            complexity: If > 0, sets the exact depth target (Curriculum Learning)
        """
        samples = []
        
        # Determine depth strategy
        if complexity > 0:
            # Curriculum mode: Force specific depth
            depth_range = [complexity]
        else:
            # Balanced mode: Uniform distribution over allowed depths
            depth_range = range(self.config.min_depth, self.config.max_depth + 1)
        
        samples_per_depth = int(n_samples / len(depth_range))
        
        print(f"Generating ~{samples_per_depth} formulas per depth level {list(depth_range)}...")

        for d in depth_range:
            for _ in range(samples_per_depth):
                tokens = self.generate_formula(0, d, [])
                samples.append({
                    'tokens': tokens,
                    'meta_depth': d
                })
        
        # Fill remainder to match exact n_samples
        while len(samples) < n_samples:
            d = random.choice(depth_range)
            tokens = self.generate_formula(0, d, [])
            samples.append({'tokens': tokens, 'meta_depth': d})
            
        random.shuffle(samples)
        return samples


class NextSymbolDataset:
    """Wrapper to format data for Next Token Prediction training."""
    def __init__(self, formulas_with_meta: List[Dict], vocab: Vocabulary, max_seq_len: int = 256):
        self.samples = []
        skipped_count = 0
        
        for entry in formulas_with_meta:
            formula = entry['tokens']
            meta_depth = entry['meta_depth']
            
            # Strictly filter out sequences that are too long
            if len(formula) > max_seq_len:
                skipped_count += 1
                continue
                
            # Create autoregressive pairs
            # Context: [A, B, C] -> Target: D
            for i in range(1, len(formula)):
                context = formula[:i]
                target = formula[i]
                
                self.samples.append({
                    'context': context,
                    'target': target,
                    'depth': meta_depth
                })
        
        if skipped_count > 0:
            print(f"⚠ Warning: Skipped {skipped_count} formulas exceeding max_seq_len ({max_seq_len})")
                
    def to_json(self, path: Path):
        # Using a generator/list write might be better for huge files, 
        # but sticking to standard dump for drop-in compatibility.
        with open(path, 'w') as f:
            json.dump({'samples': self.samples}, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")


def generate_advanced_training_data(
    vocab_path: str = "unified_vocabulary.json",
    output_dir: str = "datasets/fol_next_symbol",
    n_train: int = 15000,
    n_val: int = 2000,
    n_test: int = 1000,
    seed: int = 42,
):
    print("="*60)
    print("BALANCED DEPTH FOL GENERATION (v2)")
    print("="*60)
    
    try:
        vocab = Vocabulary(vocab_path)
    except FileNotFoundError:
        print(f"❌ Vocabulary not found at {vocab_path}")
        # For testing without existing vocab file, uncomment below:
        # vocab = Vocabulary("dummy") 
        return

    # Use max_depth=10 to force the model to learn counting and structure
    # Increased term_depth to 3 for richer atoms
    config = FormulaConfig(max_depth=10, max_term_depth=3) 
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
        'generator': 'balanced_depth_v2',
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