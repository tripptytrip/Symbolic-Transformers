import random
import json
from pathlib import Path
from typing import List
import sys

# Add parent to path to allow imports from utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.vocabulary import Vocabulary

class TransitivityGenerator:
    """
    Generates transitivity chains: A=B, B=C, C=D -> A=D
    """
    def __init__(self, vocab: Vocabulary, seed: int = 42):
        self.vocab = vocab
        random.seed(seed)
        
        # Cache token IDs
        self.sep_id = vocab.encode_label('SEP')
        self.query_id = vocab.encode_label('QUERY')
        self.answer_id = vocab.encode_label('ANSWER')
        self.end_id = vocab.encode_label('END')
        self.equals_id = vocab.encode_label('EQUALS')
        
    def generate_chain(self, length: int) -> List[int]:
        # 1. Select distinct variables (0-99 to use multi-digit embeddings)
        num_vars = length + 1
        var_indices = random.sample(range(100), num_vars)
        
        # 2. Build Premises (a=b, b=c...)
        tokens = []
        for i in range(length):
            tokens.extend(self.vocab.encode_compositional('VAR', var_indices[i]))
            tokens.append(self.equals_id)
            tokens.extend(self.vocab.encode_compositional('VAR', var_indices[i+1]))
            
            if i < length - 1:
                tokens.append(self.sep_id)
                
        # 3. Add Query/Answer (Query a? -> c)
        tokens.append(self.sep_id)
        tokens.append(self.query_id)
        tokens.extend(self.vocab.encode_compositional('VAR', var_indices[0]))
        
        tokens.append(self.answer_id)
        tokens.extend(self.vocab.encode_compositional('VAR', var_indices[-1]))
        tokens.append(self.end_id)
        
        return tokens