#!/usr/bin/env python3
"""
Test Fixed Tokenizer on Real Data
=================================

Test the fixed NER BPE tokenizer on actual domain_1 (Twitter) data.
"""

import sys
sys.path.append('.')

from ner_bpe_tokenizer import NERBPETokenizer

def test_real_data():
    """Test tokenizer on real Twitter data."""
    
    print("=== TESTING FIXED TOKENIZER ON REAL DATA ===")
    
    # Read actual domain_1 data (subset)
    try:
        with open('data/domain_1_train.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()[:10000]]  # Use first 10k lines
    except FileNotFoundError:
        print("domain_1_train.txt not found, using sample data")
        texts = [
            "I love this song http://bit.ly/abc123",
            "@username check this out",
            "Going to the beach today #vacation",
            "What a beautiful day !",
            "I can not believe this happened",
        ] * 200
    
    print(f"Using {len(texts)} training texts")
    
    # Initialize and train tokenizer
    tokenizer = NERBPETokenizer(vocab_size=1000, domain="twitter")
    print("Training tokenizer...")
    tokenizer.train(texts)
    
    print(f"\nTraining complete!")
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    print(f"BPE merges performed: {len(tokenizer._bpe_merge_ranks)}")
    
    # Save tokenizer
    tokenizer.save("test_fixed_tokenizer/tokenizer.pkl")
    print("Tokenizer saved to test_fixed_tokenizer/tokenizer.pkl")
    
    # Check most recent merges
    print("\nLast 10 BPE merges:")
    if tokenizer._bpe_merge_ranks:
        sorted_merges = sorted(tokenizer._bpe_merge_ranks.items(), key=lambda x: x[1], reverse=True)
        for (char1, char2), rank in sorted_merges[:10]:
            merged = char1 + char2
            print(f"  {repr(char1)} + {repr(char2)} -> {repr(merged)}")
    
    # Test encoding
    print("\nTesting encoding:")
    test_sentences = [
        "I love this song !",
        "@username is awesome",
        "Check out http://example.com",
        "Going to the beach #vacation"
    ]
    
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        print(f"'{sentence}' -> {len(encoded)} tokens -> '{decoded}'")
    
    return tokenizer

if __name__ == "__main__":
    tokenizer = test_real_data()
