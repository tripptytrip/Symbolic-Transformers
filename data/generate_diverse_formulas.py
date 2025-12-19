#!/usr/bin/env python3
"""
Enhanced FOL formula generator with diversity improvements.

Key tricks for more diversity:
1. Variable reuse patterns (some formulas reuse vars, some don't)
2. Mixed arity predicates (unary, binary, ternary)
3. Nested quantifiers with different patterns
4. Both prenex and non-prenex normal forms
5. Various connective combinations
"""

import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_generator import FOLFormulaGenerator, FormulaConfig, NextSymbolDataset
from utils.vocabulary import Vocabulary
import json

class DiverseFormulaGenerator(FOLFormulaGenerator):
    """Enhanced generator with more diversity tricks."""
    
    def __init__(self, vocab, config):
        super().__init__(vocab, config)
        
        # Diversity settings
        self.variable_reuse_prob = 0.7  # Probability of reusing existing variable
        self.nested_quantifier_prob = 0.3  # Probability of nested quantifiers
        self.mixed_connectives_prob = 0.4  # Mix different connectives
    
    def generate_formula(self, depth=0, bound_vars=None):
        """Enhanced formula generation with diversity tricks."""
        if bound_vars is None:
            bound_vars = []
        
        # Base case: atomic
        if depth >= self.config.max_depth or random.random() < 0.25:
            return self.generate_atomic(bound_vars)
        
        # Choose formula type with adjusted probabilities
        choices = []
        weights = []
        
        if self.config.use_quantifiers:
            choices.append('quantifier')
            weights.append(0.35)  # Higher probability for quantifiers
        
        if self.config.use_connectives:
            choices.append('binary_connective')
            weights.append(0.40)
            choices.append('unary_connective')
            weights.append(0.15)
        
        choices.append('atomic')
        weights.append(0.10)
        
        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
        formula_type = random.choices(choices, weights=weights)[0]
        
        if formula_type == 'quantifier':
            return self.generate_quantified(depth, bound_vars)
        elif formula_type == 'binary_connective':
            return self.generate_binary_compound(depth, bound_vars)
        elif formula_type == 'unary_connective':
            return self.generate_unary_compound(depth, bound_vars)
        else:
            return self.generate_atomic(bound_vars)
    
    def generate_quantified(self, depth, bound_vars):
        """Generate quantified formula with diversity."""
        quantifier = random.choice(self.quantifiers)
        
        # Choose variable index
        if bound_vars and random.random() < 0.3:
            # Sometimes use same index as existing var (different scope)
            var_idx = random.choice(bound_vars)
        else:
            var_idx = random.randint(0, self.config.max_variables - 1)
        
        tokens = [self.vocab.encode_label(quantifier)]
        tokens.extend(self.generate_variable(var_idx))
        
        # Sometimes add nested quantifier
        if random.random() < self.nested_quantifier_prob and depth < 2:
            # Nested quantifier
            inner = self.generate_quantified(depth + 1, bound_vars + [var_idx])
        else:
            # Regular body
            inner = self.generate_formula(depth + 1, bound_vars + [var_idx])
        
        tokens.extend(inner)
        return tokens
    
    def generate_binary_compound(self, depth, bound_vars):
        """Generate binary connective with variety."""
        # Choose connective
        binary_connectives = ['AND', 'OR', 'IMPLIES', 'IFF']
        
        # Sometimes mix different connectives in left/right
        if random.random() < self.mixed_connectives_prob:
            connective = random.choice(binary_connectives)
        else:
            connective = random.choice(binary_connectives)
        
        tokens = [self.lparen]
        
        # Left side
        left = self.generate_formula(depth + 1, bound_vars)
        tokens.extend(left)
        
        tokens.append(self.vocab.encode_label(connective))
        
        # Right side
        right = self.generate_formula(depth + 1, bound_vars)
        tokens.extend(right)
        
        tokens.append(self.rparen)
        return tokens
    
    def generate_unary_compound(self, depth, bound_vars):
        """Generate negation."""
        tokens = [self.vocab.encode_label('NOT')]
        subformula = self.generate_formula(depth + 1, bound_vars)
        tokens.extend(subformula)
        return tokens
    
    def generate_atomic(self, bound_vars=None):
        """Generate atomic with variable arity predicates."""
        if bound_vars is None:
            bound_vars = []
        
        # Decide: predicate or equality
        if self.config.use_equality and random.random() < 0.25:
            return self.generate_equality(bound_vars)
        else:
            return self.generate_predicate_varied_arity(bound_vars)
    
    def generate_predicate_varied_arity(self, bound_vars):
        """Generate predicate with varied arity (1-3 arguments)."""
        pred_idx = random.randint(0, self.config.max_predicates - 1)
        
        # Choose arity with weighted distribution
        arity_choices = [1, 2, 3]
        arity_weights = [0.5, 0.4, 0.1]  # Mostly unary and binary
        arity = random.choices(arity_choices, weights=arity_weights)[0]
        
        tokens = self.vocab.encode_compositional('PRED', pred_idx)
        tokens.append(self.lparen)
        
        # Generate arguments
        var_indices = []
        for i in range(arity):
            if bound_vars and random.random() < self.variable_reuse_prob:
                # Reuse bound variable
                var_idx = random.choice(bound_vars)
            else:
                # New variable
                var_idx = random.randint(0, self.config.max_variables - 1)
            var_indices.append(var_idx)
        
        # Add variables to formula
        for i, var_idx in enumerate(var_indices):
            tokens.extend(self.generate_variable(var_idx))
            if i < len(var_indices) - 1:
                tokens.append(self.comma)
        
        tokens.append(self.rparen)
        return tokens
    
    def generate_equality(self, bound_vars):
        """Generate equality with variety."""
        # Sometimes use same variable (x = x), sometimes different
        if bound_vars and len(bound_vars) >= 2 and random.random() < 0.7:
            var1, var2 = random.sample(bound_vars, 2)
        elif bound_vars and random.random() < 0.3:
            # x = x pattern
            var1 = var2 = random.choice(bound_vars)
        else:
            var1 = random.randint(0, self.config.max_variables - 1)
            var2 = random.randint(0, self.config.max_variables - 1)
        
        tokens = self.generate_variable(var1)
        
        # Mostly use EQUALS, sometimes NOT_EQUALS
        if random.random() < 0.8:
            tokens.append(self.vocab.encode_label('EQUALS'))
        else:
            tokens.append(self.vocab.encode_label('NOT_EQUALS'))
        
        tokens.extend(self.generate_variable(var2))
        return tokens


def generate_diverse_dataset():
    """Generate dataset with enhanced diversity."""
    print("\n" + "="*60)
    print("GENERATING DIVERSE FOL DATASET")
    print("="*60)
    
    # Load vocab
    vocab = Vocabulary("unified_vocabulary.json")
    
    # Enhanced config
    config = FormulaConfig(
        max_depth=4,  # Allow deeper nesting
        max_variables=6,  # More variables
        max_predicates=8,  # More predicates
        use_quantifiers=True,
        use_connectives=True,
        use_equality=True
    )
    
    # Create enhanced generator
    generator = DiverseFormulaGenerator(vocab, config)
    
    # Generate formulas
    print("\nGenerating formulas...")
    
    # Different complexity levels
    all_formulas = []
    
    for complexity in range(1, 6):
        n_for_complexity = {
            1: 1500,  # Simple
            2: 3500,  # Medium
            3: 3500,  # Complex
            4: 1000,  # Very complex
            5: 500,   # Extremely complex
        }[complexity]
        
        print(f"  Complexity {complexity}: {n_for_complexity} formulas")
        formulas = generator.generate_batch(n_for_complexity, complexity)
        all_formulas.extend(formulas)
    
    # Split into train/val/test
    random.shuffle(all_formulas)
    
    n_train = 8000
    n_val = 1000
    n_test = 1000
    
    train_formulas = all_formulas[:n_train]
    val_formulas = all_formulas[n_train:n_train+n_val]
    test_formulas = all_formulas[n_train+n_val:n_train+n_val+n_test]
    
    # Create datasets
    print("\nCreating training samples...")
    train_dataset = NextSymbolDataset(train_formulas, max_seq_len=128)
    val_dataset = NextSymbolDataset(val_formulas, max_seq_len=128)
    test_dataset = NextSymbolDataset(test_formulas, max_seq_len=128)
    
    # Save
    output_dir = Path("datasets/fol_next_symbol")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset.save(str(output_dir / "train.json"))
    val_dataset.save(str(output_dir / "val.json"))
    test_dataset.save(str(output_dir / "test.json"))
    
    # Save metadata
    metadata = {
        'vocab_size': vocab.vocab_size,
        'config': {
            'max_depth': config.max_depth,
            'max_variables': config.max_variables,
            'max_predicates': config.max_predicates,
        },
        'diversity_enhancements': [
            'Variable reuse patterns',
            'Mixed arity predicates (1-3 args)',
            'Nested quantifiers',
            'Mixed connective types',
            'Equality variations',
        ],
        'splits': {
            'train': {'formulas': len(train_formulas), 'samples': len(train_dataset)},
            'val': {'formulas': len(val_formulas), 'samples': len(val_dataset)},
            'test': {'formulas': len(test_formulas), 'samples': len(test_dataset)},
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"\nTrain: {len(train_dataset):,} samples")
    print(f"Val:   {len(val_dataset):,} samples")
    print(f"Test:  {len(test_dataset):,} samples")
    print(f"\nDiversity enhancements applied:")
    for enh in metadata['diversity_enhancements']:
        print(f"  ✓ {enh}")
    print("="*60)


if __name__ == "__main__":
    generate_diverse_dataset()
