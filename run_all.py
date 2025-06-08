import os
import subprocess

def main():
    print("Select domain:")
    print("1. Domain 1 (Twitter)")
    print("2. Domain 2 (News)")
    print("3. Unknown domain")
    domain_choice = input("Enter 1, 2, or 3: ").strip()
    if domain_choice == '1':
        domain = 'domain_1'
        train_file = 'data/domain_1_train.txt'
        dev_file = 'data/domain_1_dev.txt'
        ner_train = 'data/ner_data/train_1_binary.tagged'
        ner_dev = 'data/ner_data/dev_1_binary.tagged'
    elif domain_choice == '2':
        domain = 'domain_2'
        train_file = 'data/domain_2_train.txt'
        dev_file = 'data/domain_2_dev.txt'
        ner_train = 'data/ner_data/train_2_binary.tagged'
        ner_dev = 'data/ner_data/dev_2_binary.tagged'
    else:
        domain = 'unknown'
        train_file = 'data/domain_1_train.txt'
        dev_file = 'data/domain_1_dev.txt'
        ner_train = 'data/ner_data/train_1_binary.tagged'
        ner_dev = 'data/ner_data/dev_1_binary.tagged'

    tokenizer_dir = f'tokenizers/{domain}'
    tokenizer_path = f'{tokenizer_dir}/tokenizer.pkl'

    print(f"Selected domain: {domain}")
    print("\nDo you want to train a new tokenizer or load an existing one?")
    print("1. Train new tokenizer")
    print("2. Load existing tokenizer")
    action = input("Enter 1 or 2: ").strip()

    # Train or load tokenizer
    if action == '1':
        print(f"Training tokenizer for {domain}...")
        subprocess.run([
            "python3", "train_tokenizer.py",
            "--domain_file", train_file,
            "--output_dir", tokenizer_dir,
            "--vocab_size", "10000",
            "--train"
        ], check=True)
    else:
        print(f"Loading existing tokenizer for {domain}...")
        if not os.path.exists(tokenizer_path):
            print(f"Tokenizer not found at {tokenizer_path}. Please train it first.")
            return

    # Test tokenizer
    print(f"\nTesting tokenizer for {domain}...")
    subprocess.run([
        "python3", "test_tokenizer.py",
        "--tokenizer_path", tokenizer_path,
        "--train_file", train_file,
        "--test_file", dev_file
    ], check=True)

    # Train NER model
    print(f"\nTraining NER model for {domain}...")
    subprocess.run([
        "python3", "train_ner_model.py",
        "--tokenizer_path", tokenizer_path,
        "--train_file", ner_train,
        "--dev_file", ner_dev
    ], check=True)

    print("\nAll steps completed.")

if __name__ == "__main__":
    main()
