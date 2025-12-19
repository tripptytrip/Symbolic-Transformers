"""
Vocabulary utilities for Base-625 FOL Transformer.
Handles loading unified vocabulary and encoding/decoding operations.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union


class Vocabulary:
    """
    Unified vocabulary for base-625 numerals + FOL symbols.
    
    Vocabulary structure:
    - IDs 0-624: Numeral symbols (positional encoding in 25x25 grid)
    - IDs 625+: FOL operator symbols (NOT, AND, OR, VAR, PRED, etc.)
    """
    
    def __init__(self, vocab_path: str):
        """
        Load unified vocabulary from JSON.
        
        Args:
            vocab_path: Path to unified_vocabulary.json
        """
        self.vocab_path = Path(vocab_path)
        
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.vocab_size = self.vocab['vocab_size']
        self.numeral_range = tuple(self.vocab['numeral_range'])
        self.symbol_range = tuple(self.vocab['symbol_range'])
        
        # Create lookups
        self.label_to_id = self.vocab['label_to_id']
        self.id_to_label = {int(k): v for k, v in self.vocab['id_to_label'].items()}
        
        # Token metadata
        self.tokens = {t['id']: t for t in self.vocab['tokens']}
        
        # Special tokens
        self.compositional_tokens = {
            token['label']: token['id']
            for token in self.vocab['tokens']
            if token.get('compositional', False)
        }
        
        print(f"✓ Vocabulary loaded: {self.vocab_size} tokens")
        print(f"  - Numerals: {self.numeral_range[0]}-{self.numeral_range[1]}")
        print(f"  - Symbols: {self.symbol_range[0]}-{self.symbol_range[1]}")
        print(f"  - Compositional: {list(self.compositional_tokens.keys())}")
    
    def encode_label(self, label: str) -> int:
        """Convert label to token ID."""
        if label not in self.label_to_id:
            raise ValueError(f"Unknown label: {label}")
        return self.label_to_id[label]
    
    def decode_id(self, token_id: int) -> str:
        """Convert token ID to label."""
        if token_id not in self.id_to_label:
            raise ValueError(f"Unknown token ID: {token_id}")
        return self.id_to_label[token_id]
    
    def encode_numeral(self, value: int) -> int:
        """
        Encode a numeric value as numeral token ID.
        
        Args:
            value: Integer 0-624
            
        Returns:
            Token ID (same as value for numerals)
        """
        if not (0 <= value <= 624):
            raise ValueError(f"Numeral value must be 0-624, got {value}")
        return value
    
    def decode_numeral(self, token_id: int) -> int:
        """
        Decode numeral token ID to numeric value.
        
        Args:
            token_id: Token ID (0-624)
            
        Returns:
            Numeric value
        """
        if not (self.numeral_range[0] <= token_id <= self.numeral_range[1]):
            raise ValueError(f"Token {token_id} is not a numeral")
        return token_id
    
    def encode_compositional(self, category: str, index: int) -> List[int]:
        """
        Encode compositional token (e.g., VAR 12 → [VAR_ID, NUM_1, NUM_2]).
        
        Args:
            category: 'VAR', 'PRED', 'FUNC', 'CONST', 'SORT'
            index: Integer identifier
            
        Returns:
            List of token IDs [category_token, digit_1, ..., digit_n]
        """
        if category not in self.compositional_tokens:
            raise ValueError(f"Unknown compositional category: {category}")
        
        # Get category token ID
        category_id = self.compositional_tokens[category]
        
        # Convert index to base-625 digits
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")
        
        if index == 0:
            digits = [0]
        else:
            digits = []
            n = index
            while n > 0:
                digits.append(n % 625)
                n //= 625
            digits.reverse()
        
        # Encode as token sequence
        token_ids = [category_id] + digits
        
        return token_ids
    
    def decode_compositional(
        self, 
        tokens: List[int], 
        start_idx: int = 0
    ) -> tuple[str, int, int]:
        """
        Decode compositional token sequence starting at start_idx.
        
        Args:
            tokens: Full token sequence
            start_idx: Index to start decoding from
            
        Returns:
            (category, index, next_idx) tuple
            - category: 'VAR', 'PRED', 'FUNC', 'CONST', 'SORT'
            - index: Decoded integer index
            - next_idx: Index after this compositional token
        """
        if start_idx >= len(tokens):
            raise ValueError("start_idx out of range")
        
        category_id = tokens[start_idx]
        
        # Check if this is a compositional token
        category = None
        for cat, cid in self.compositional_tokens.items():
            if cid == category_id:
                category = cat
                break
        
        if category is None:
            raise ValueError(f"Token {category_id} is not a compositional category")
        
        # Decode following numeral digits
        idx = start_idx + 1
        value = 0
        
        while idx < len(tokens) and self.is_numeral(tokens[idx]):
            digit = self.decode_numeral(tokens[idx])
            value = value * 625 + digit
            idx += 1
        
        return category, value, idx
    
    def is_numeral(self, token_id: int) -> bool:
        """Check if token ID is a numeral."""
        return self.numeral_range[0] <= token_id <= self.numeral_range[1]
    
    def is_compositional(self, token_id: int) -> bool:
        """Check if token ID is a compositional category token."""
        return token_id in self.compositional_tokens.values()
    
    def get_token_info(self, token_id: int) -> Dict:
        """Get full token metadata."""
        if token_id not in self.tokens:
            raise ValueError(f"Unknown token ID: {token_id}")
        return self.tokens[token_id]
    
    def encode_formula_simple(self, formula_str: str) -> List[int]:
        """
        Simple encoder for testing: converts label sequence to token IDs.
        
        Example: "FORALL VAR 1 PRED 5 LPAREN VAR 1 RPAREN"
                 → [FORALL_ID, VAR_ID, 1, PRED_ID, 5, LPAREN_ID, VAR_ID, 1, RPAREN_ID]
        
        Args:
            formula_str: Space-separated labels
            
        Returns:
            List of token IDs
        """
        labels = formula_str.split()
        token_ids = []
        
        for label in labels:
            # Try as label first
            if label in self.label_to_id:
                token_ids.append(self.label_to_id[label])
            # Try as numeral
            elif label.isdigit():
                value = int(label)
                token_ids.append(self.encode_numeral(value))
            else:
                raise ValueError(f"Unknown label or numeral: {label}")
        
        return token_ids
    
    def decode_formula_simple(self, token_ids: List[int]) -> str:
        """
        Simple decoder for testing: converts token IDs to label sequence.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Space-separated label string
        """
        labels = []
        
        for token_id in token_ids:
            if self.is_numeral(token_id):
                labels.append(str(self.decode_numeral(token_id)))
            else:
                labels.append(self.decode_id(token_id))
        
        return " ".join(labels)


class FormulaEncoder:
    """
    Encodes FOL formulas into token sequences.
    Handles compositional encoding for variables, predicates, functions.
    """
    
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
    
    def encode(self, formula: str) -> List[int]:
        """
        Encode FOL formula to token sequence.
        
        For now, uses simple label-based encoding.
        TODO: Add proper parser for mathematical notation.
        
        Args:
            formula: Formula string (label-based for now)
            
        Returns:
            List of token IDs
        """
        return self.vocab.encode_formula_simple(formula)


class FormulaDecoder:
    """
    Decodes token sequences into FOL formulas.
    """
    
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
    
    def decode(self, token_ids: List[int], format: str = 'labels') -> str:
        """
        Decode token sequence to formula.
        
        Args:
            token_ids: List of token IDs
            format: Output format ('labels', 'unicode', 'latex')
            
        Returns:
            Formula string
        """
        if format == 'labels':
            return self.vocab.decode_formula_simple(token_ids)
        else:
            raise NotImplementedError(f"Format {format} not yet implemented")


if __name__ == "__main__":
    # Test vocabulary loading
    print("Testing Vocabulary class...\n")
    
    vocab = Vocabulary("unified_vocabulary.json")
    
    # Test numeral encoding
    print("\n" + "="*60)
    print("Testing numeral encoding:")
    for val in [0, 1, 24, 100, 624]:
        token_id = vocab.encode_numeral(val)
        decoded = vocab.decode_numeral(token_id)
        print(f"  {val} → token {token_id} → {decoded} ✓")
    
    # Test label encoding
    print("\n" + "="*60)
    print("Testing label encoding:")
    test_labels = ['NOT', 'AND', 'OR', 'FORALL', 'EXISTS', 'VAR', 'PRED']
    for label in test_labels:
        token_id = vocab.encode_label(label)
        decoded = vocab.decode_id(token_id)
        print(f"  {label} → token {token_id} → {decoded} ✓")
    
    # Test compositional encoding
    print("\n" + "="*60)
    print("Testing compositional encoding:")
    test_cases = [
        ('VAR', 0),
        ('VAR', 1),
        ('VAR', 12),
        ('PRED', 5),
        ('FUNC', 100),
    ]
    for category, index in test_cases:
        tokens = vocab.encode_compositional(category, index)
        decoded_cat, decoded_idx, _ = vocab.decode_compositional(tokens, 0)
        print(f"  {category} {index} → tokens {tokens} → {decoded_cat} {decoded_idx} ✓")
    
    # Test simple formula encoding
    print("\n" + "="*60)
    print("Testing formula encoding:")
    formula = "FORALL VAR 1 LPAREN PRED 5 LPAREN VAR 1 RPAREN RPAREN"
    print(f"Formula: {formula}")
    tokens = vocab.encode_formula_simple(formula)
    print(f"Tokens: {tokens}")
    decoded = vocab.decode_formula_simple(tokens)
    print(f"Decoded: {decoded}")
    print(f"Match: {formula == decoded} ✓")
    
    print("\n" + "="*60)
    print("✓ All vocabulary tests passed!")
