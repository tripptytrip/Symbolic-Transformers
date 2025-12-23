#!/usr/bin/env python3
"""
Advanced FOL Formula Generator with MESSAGE_START/MESSAGE_END support.
Generates formulas with proper message boundaries for training.
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
    # Fallback for standalone testing
    class Vocabulary:
        def __init__(self, path): 
            self.compositional_tokens = {'VAR', 'FUNC', 'CONST', 'PRED'}
            self.message_start = None
            self.message_end = None
        def encode_label(self, label): return label
        def encode_compositional(self, kind, idx): return [f"{kind}_{idx}"]
        def has_special_token(self, name): return False

# Increase recursion limit for deep formula generation
sys.setrecursionlimit(5000)

@dataclass
class FormulaConfig:
    """Configuration for formula generation."""
    min_depth: int = 1
    max_depth: int = 12
    max_term_depth: int = 3
    max_variables: int = 100
    max_predicates: int = 50
    max_functions: int = 4
    max_constants: int = 5
    
    # Structural Toggles
    use_quantifiers: bool = True
    use_connectives: bool = True
    use_equality: bool = True
    use_message_boundaries: bool = True  # NEW: Control MESSAGE_START/END
    
    # Probability tunings
    prob_shadowing: float = 0.05
    prob_vacuous: float = 0.05
    prob_local_var: float = 0.70
    
    # NEW: Structural diversity parameters
    prob_standalone_atom: float = 0.15
    prob_early_termination: float = 0.20
    prob_wrapped_atom: float = 0.10

class AdvancedFormulaGenerator:
    """
    Generates FOL formulas with message boundaries and structural diversity.
    """
    def __init__(self, vocab: Vocabulary, config: FormulaConfig, seed: Optional[int] = None):
        self.vocab = vocab
        self.config = config
        
        if seed is not None:
            random.seed(seed)
        
        # Check if MESSAGE_START/MESSAGE_END are available
        self.has_boundaries = (
            vocab.has_special_token('MESSAGE_START') and 
            vocab.has_special_token('MESSAGE_END')
        )
        
        if config.use_message_boundaries and not self.has_boundaries:
            print("⚠ Warning: MESSAGE_START/MESSAGE_END not in vocabulary, disabling boundaries")
            config.use_message_boundaries = False
        
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
        
        # Add special tokens if available
        if self.has_boundaries:
            self.tokens['MESSAGE_START'] = vocab.message_start
            self.tokens['MESSAGE_END'] = vocab.message_end

        # Capability checks
        self.has_funcs = hasattr(vocab, 'compositional_tokens') and 'FUNC' in vocab.compositional_tokens
        self.has_consts = hasattr(vocab, 'compositional_tokens') and 'CONST' in vocab.compositional_tokens
        
        # Initialize Fixed Signatures (Arity Maps)
        self.pred_arities = {i: random.choices([1, 2, 3], weights=[0.45, 0.45, 0.1])[0] 
                             for i in range(config.max_predicates)}
        
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
            if random.random() < self.config.prob_local_var:
                recent_window = bound_vars[-min(len(bound_vars), 3):]
                var_idx = random.choice(recent_window)
            else:
                var_idx = random.choice(bound_vars)
            return self.vocab.encode_compositional('VAR', var_idx)
        
        return self.vocab.encode_compositional('VAR', random.randint(0, self.config.max_variables - 1))

    def _generate_term(self, current_depth: int, max_depth: int, bound_vars: List[int]) -> List[int]:
        """Recursively generates a term (variable, constant, or function)."""
        if current_depth >= max_depth or not self.has_funcs or random.random() < 0.4:
            if self.has_consts and random.random() < 0.2:
                idx = random.randint(0, self.config.max_constants - 1)
                return self.vocab.encode_compositional('CONST', idx)
            return self._generate_variable(bound_vars)

        func_idx = random.randint(0, self.config.max_functions - 1)
        arity = self.func_arities[func_idx]
        
        tokens = self.vocab.encode_compositional('FUNC', func_idx)
        tokens.append(self.tokens['LPAREN'])
        for i in range(arity):
            tokens.extend(self._generate_term(current_depth + 1, max_depth, bound_vars))
            if i < arity - 1:
                tokens.append(self.tokens['COMMA'])
        tokens.append(self.tokens['RPAREN'])
        return tokens

    def _generate_atomic(self, bound_vars: List[int]) -> List[int]:
        """Generates an atomic formula (Equality or Predicate)."""
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
        """Recursive generation attempting to hit target_depth exactly."""
        if current_depth >= target_depth:
            return self._generate_atomic(bound_vars)

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

        if not options:
            return self._generate_atomic(bound_vars)
            
        choice = random.choices(options, weights=weights, k=1)[0]

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
            op = random.choice([self.tokens['AND'], self.tokens['OR'], 
                               self.tokens['IMPLIES'], self.tokens['IFF']])
            
            deep_branch = self.generate_formula(current_depth + 1, target_depth, bound_vars)
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
        Generates a balanced batch with structural diversity.
        """
        samples = []
        
        # Generate standalone atoms for termination learning
        n_standalone = int(n_samples * self.config.prob_standalone_atom)
        for _ in range(n_standalone):
            tokens = self._generate_atomic([])
            samples.append({
                'tokens': tokens,
                'meta_depth': 0,
                'structural_type': 'standalone_atom'
            })
        
        # Generate wrapped atoms (P(x)) for parenthesis learning
        n_wrapped = int(n_samples * self.config.prob_wrapped_atom)
        for _ in range(n_wrapped):
            atom = self._generate_atomic([])
            tokens = [self.tokens['LPAREN']] + atom + [self.tokens['RPAREN']]
            samples.append({
                'tokens': tokens,
                'meta_depth': 1,
                'structural_type': 'wrapped_atom'
            })
        
        # Remaining samples use depth-balanced generation
        remaining = n_samples - len(samples)
        
        if complexity > 0:
            depth_range = [complexity]
        else:
            depth_range = range(self.config.min_depth, self.config.max_depth + 1)
        
        samples_per_depth = remaining // len(depth_range)
        
        print(f"Generating formulas:")
        print(f"  - Standalone atoms: {n_standalone}")
        print(f"  - Wrapped atoms: {n_wrapped}")
        print(f"  - Complex formulas: ~{samples_per_depth} per depth {list(depth_range)}")

        for d in depth_range:
            for _ in range(samples_per_depth):
                # Sometimes stop early for diversity
                if random.random() < self.config.prob_early_termination:
                    actual_depth = random.randint(self.config.min_depth, d)
                else:
                    actual_depth = d
                    
                tokens = self.generate_formula(0, actual_depth, [])
                samples.append({
                    'tokens': tokens,
                    'meta_depth': d,
                    'actual_depth': actual_depth,
                    'structural_type': 'complex'
                })
        
        # Fill to exact count
        while len(samples) < n_samples:
            d = random.choice(depth_range)
            tokens = self.generate_formula(0, d, [])
            samples.append({
                'tokens': tokens,
                'meta_depth': d,
                'structural_type': 'complex'
            })
            
        random.shuffle(samples)
        return samples


class NextSymbolDataset:
    """Wrapper to format data for Next Token Prediction training with message boundaries."""
    def __init__(self, formulas_with_meta: List[Dict], vocab: Vocabulary, 
                 max_seq_len: int = 512, use_boundaries: bool = True):
        self.samples = []
        skipped_count = 0
        
        # Check if boundaries are available
        has_boundaries = (
            vocab.has_special_token('MESSAGE_START') and 
            vocab.has_special_token('MESSAGE_END')
        )
        
        if use_boundaries and not has_boundaries:
            print("⚠ Warning: MESSAGE_START/MESSAGE_END not available, proceeding without boundaries")
            use_boundaries = False
        
        for entry in formulas_with_meta:
            formula = entry['tokens']
            meta_depth = entry['meta_depth']
            structural_type = entry.get('structural_type', 'complex')
            
            # Wrap with message boundaries if enabled
            if use_boundaries:
                full_formula = [vocab.message_start] + formula + [vocab.message_end]
            else:
                full_formula = formula
            
            # Filter out sequences that are too long
            if len(full_formula) > max_seq_len:
                skipped_count += 1
                continue
            
            # Create autoregressive pairs INCLUDING MESSAGE_END as a target
            for i in range(1, len(full_formula)):
                context = full_formula[:i]
                target = full_formula[i]
                
                self.samples.append({
                    'context': context,
                    'target': target,
                    'depth': meta_depth,
                    'structural_type': structural_type
                })
        
        if skipped_count > 0:
            print(f"⚠ Warning: Skipped {skipped_count} formulas exceeding max_seq_len ({max_seq_len})")
        
        # Report boundary usage
        if use_boundaries and self.samples:
            boundary_samples = sum(1 for s in self.samples 
                                 if s['target'] in [vocab.message_start, vocab.message_end])
            print(f"✓ Generated {len(self.samples)} samples ({boundary_samples} boundary targets)")
                
    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump({'samples': self.samples}, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")


def generate_advanced_training_data(
    vocab_path: str = "unified_vocabulary.json",
    output_dir: str = "datasets/fol_next_symbol",
    n_train: int = 50000,
    n_val: int = 5000,
    n_test: int = 5000,
    seed: int = 42,
    use_boundaries: bool = True,
):
    print("="*60)
    print("ADVANCED FOL GENERATION WITH MESSAGE BOUNDARIES")
    print("="*60)
    
    try:
        vocab = Vocabulary(vocab_path)
    except FileNotFoundError:
        print(f"❌ Vocabulary not found at {vocab_path}")
        return

    # Check boundary availability
    has_boundaries = (
        vocab.has_special_token('MESSAGE_START') and 
        vocab.has_special_token('MESSAGE_END')
    )
    
    if use_boundaries and not has_boundaries:
        print("⚠ Warning: MESSAGE_START/MESSAGE_END not in vocabulary")
        print("   Run update_vocabulary_complete.py first!")
        print("   Proceeding without boundaries...")
        use_boundaries = False
    elif use_boundaries:
        print(f"✓ Using MESSAGE_START (ID {vocab.message_start}) and MESSAGE_END (ID {vocab.message_end})")

    config = FormulaConfig(
        max_depth=10, 
        max_term_depth=3,
        use_message_boundaries=use_boundaries,
        prob_standalone_atom=0.15,
        prob_early_termination=0.20,
        prob_wrapped_atom=0.10
    )
    generator = AdvancedFormulaGenerator(vocab, config, seed)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate Splits
    print(f"\nGenerating TRAIN ({n_train})...")
    train_raw = generator.generate_batch(n_train)
    train_ds = NextSymbolDataset(train_raw, vocab, max_seq_len=512, use_boundaries=use_boundaries)
    train_ds.to_json(out_path / "train.json")
    
    print(f"\nGenerating VAL ({n_val})...")
    val_raw = generator.generate_batch(n_val)
    val_ds = NextSymbolDataset(val_raw, vocab, max_seq_len=512, use_boundaries=use_boundaries)
    val_ds.to_json(out_path / "val.json")
    
    print(f"\nGenerating TEST ({n_test})...")
    test_raw = generator.generate_batch(n_test)
    test_ds = NextSymbolDataset(test_raw, vocab, max_seq_len=512, use_boundaries=use_boundaries)
    test_ds.to_json(out_path / "test.json")
    
    # Save Metadata
    metadata = {
        'generator': 'advanced_with_boundaries_v3',
        'config': asdict(config),
        'signatures': generator.get_signature_info(),
        'n_train': len(train_ds.samples),
        'n_val': len(val_ds.samples),
        'n_test': len(test_ds.samples),
        'vocab_size': vocab.vocab_size,
        'uses_boundaries': use_boundaries
    }
    with open(out_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\n✓ Generation Complete.")
    print(f"✓ Total training samples: {len(train_ds.samples)}")

if __name__ == "__main__":
    generate_advanced_training_data()
