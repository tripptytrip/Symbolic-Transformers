import json
import os
import sys

# Ensure we can find the utils package
sys.path.append(os.getcwd())

def update_vocabulary():
    vocab_path = "unified_vocabulary.json"
    
    if not os.path.exists(vocab_path):
        print(f"❌ Error: {vocab_path} not found.")
        return

    # 1. Load existing
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    print(f"Current vocab size: {vocab['vocab_size']}")

    # 2. Define Phase 2 tokens
    new_tokens = {
        "SEP": "Special token to separate premises",
        "QUERY": "Start of the query part",
        "ANSWER": "Start of the answer part",
        "END": "End of sequence",
        "UNKNOWN": "Placeholder for unanswerable queries"
    }

    # 3. Add them if missing
    changed = False
    next_id = vocab['vocab_size']

    for token, desc in new_tokens.items():
        if token not in vocab['label_to_id']:
            print(f"  + Adding {token} at ID {next_id}")
            vocab['label_to_id'][token] = next_id
            vocab['id_to_label'][str(next_id)] = token
            vocab['vocab_size'] += 1
            next_id += 1
            changed = True
        else:
            print(f"  . Skipping {token} (already exists)")

    # 4. Save
    if changed:
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"✅ Updated {vocab_path}. New size: {vocab['vocab_size']}")
    else:
        print("✓ Vocabulary already up to date.")

if __name__ == "__main__":
    # Create scripts folder if it doesn't exist (just in case)
    os.makedirs("scripts", exist_ok=True)
    update_vocabulary()