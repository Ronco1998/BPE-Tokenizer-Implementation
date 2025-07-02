import argparse
import os
from typing import List
from ner_bpe_tokenizer import NERBPETokenizer as BPETokenizer


def read_text_file(file_path: str) -> List[str]:
    """
    Read lines from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of lines from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def train_tokenizer(domain_file: str, output_dir: str, num_merges: int = 10000, train: bool = True) -> None:
    """
    Train a tokenizer on domain data and save it
    
    Args:
        domain_file: Path to the domain training data file
        output_dir: Directory where to save the trained tokenizer
        num_merges: Number of BPE merge operations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read domain data
    print(f"Reading domain data from {domain_file}")
    texts = read_text_file(domain_file)
    print(f"Read {len(texts)} lines of text")
    
    # Determine domain from filename
    domain = 'unknown'
    base = os.path.basename(domain_file).lower()
    if 'domain_1' in base:
        domain = 'twitter'
    elif 'domain_2' in base:
        domain = 'headlines'
    elif 'unknown' in base:
        domain = 'unknown'

    if train:
        # Initialize and train tokenizer
        print(f"Training BPE tokenizer with {num_merges} merges for domain '{domain}'")
        tokenizer = BPETokenizer(vocab_size=num_merges)
        tokenizer.train(texts)
        print("Tokenizer training complete")

        # Save the tokenizer
        output_path = os.path.join(output_dir, "tokenizer.pkl")
        print(f"Saving tokenizer to {output_path}")
        tokenizer.save(output_path)
        print(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")
        
        # # Report word bigrams
        # top5, bottom5 = tokenizer.report_word_bigrams()
        # if top5:
        #     print("\n=== Most frequent word-bigrams ===")
        #     for bigram, freq in top5:
        #         print(f"'{bigram[0]}' + '{bigram[1]}' → {freq} occurrences")
        # if bottom5:
        #     print("\n=== Least frequent word-bigrams ===")
        #     for bigram, freq in bottom5:
        #         print(f"'{bigram[0]}' + '{bigram[1]}' → {freq} occurrences")
    else:
        output_path = os.path.join(output_dir, "tokenizer.pkl")
        if not os.path.exists(output_path):
            print(f"Tokenizer file {output_path} does not exist. Please train the tokenizer first.")
            return
        print(f"Loading existing tokenizer from {output_path}")
        tokenizer = BPETokenizer.load(output_path)

    # Test the tokenizer on a sample
    if texts:
        sample_text = texts[0].strip()
        print("\nSample encode/decode:")
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {sample_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on domain data")
    parser.add_argument("--domain_file", type=str, required=True, help="Path to the domain data file")
    parser.add_argument("--output_dir", type=str, default="trained_tokenizer", help="Directory to save the tokenizer")
    parser.add_argument("--num_merges", type=int, default=1000, help="Number of BPE merge operations")
    parser.add_argument("--train", action="store_true", help="Whether to train the tokenizer")
    
    args = parser.parse_args()

    train_tokenizer(args.domain_file, args.output_dir, args.num_merges, args.train)