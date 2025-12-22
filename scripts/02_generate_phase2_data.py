import sys
import os
import json
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

from data.transitivity_generator import TransitivityGenerator
from utils.vocabulary import Vocabulary

class ReasoningDataset:
    def __init__(self, formulas):
        self.samples = []
        for formula in formulas:
            # Standard autoregressive training format
            for i in range(1, len(formula)):
                self.samples.append({
                    'context': formula[:i],
                    'target': formula[i]
                })
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'samples': self.samples}, f)
        print(f"✓ Saved {len(self.samples)} samples to {path}")

def main():
    print("="*60)
    print("GENERATING PHASE 2 DATA (TRANSITIVITY)")
    print("="*60)
    
    vocab = Vocabulary("unified_vocabulary.json")
    generator = TransitivityGenerator(vocab)
    
    output_dir = Path("datasets/transitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration from Engineering Spec 5.3
    # Chain length distributions
    train_dist = {2: 0.40, 3: 0.35, 4: 0.20, 5: 0.05}
    test_dist = {2: 0.2, 4: 0.2, 6: 0.2, 7: 0.2, 8: 0.2} # Includes OOD lengths
    
    configs = [
        ("train", 50000, train_dist),
        ("val", 5000, train_dist),
        ("test", 5000, test_dist)
    ]
    
    for name, count, dist in configs:
        print(f"\nGenerating {name} set...")
        formulas = []
        for length, weight in dist.items():
            n = int(count * weight)
            for _ in range(n):
                formulas.append(generator.generate_chain(length))
        
        ReasoningDataset(formulas).save(output_dir / f"{name}.json")

    print("\n✅ Phase 2 Data Generation Complete.")

if __name__ == "__main__":
    main()