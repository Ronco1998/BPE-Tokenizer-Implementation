import os
import subprocess
import sys

def main():
    output_file = "run_all_output.txt"
    def dual_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(output_file, "a") as f:
            print(*args, **kwargs, file=f)

    # Clear output file at start
    open(output_file, "w").close()

    dual_print("Select domain:")
    dual_print("1. Domain 1 (Twitter)")
    dual_print("2. Domain 2 (News)")
    dual_print("3. Unknown domain")
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

    dual_print(f"Selected domain: {domain}")
    dual_print("\nDo you want to train a new tokenizer or load an existing one?")
    dual_print("1. Train new tokenizer")
    dual_print("2. Load existing tokenizer")
    action = input("Enter 1 or 2: ").strip()

    # Train or load tokenizer
    if action == '1':
        # Ask for number of merges
        num_merges = input("Enter number of BPE merges (default 10000): ").strip()
        dual_print(f"Training tokenizer for {domain}...")
        if not num_merges:
            num_merges = "10000"
        subprocess.run([
            "python3", "train_tokenizer.py",
            "--domain_file", train_file,
            "--output_dir", tokenizer_dir,
            "--num_merges", num_merges,
            "--train"
        ], check=True)
    else:
        dual_print(f"Loading existing tokenizer for {domain}...")
        if not os.path.exists(tokenizer_path):
            dual_print(f"Tokenizer not found at {tokenizer_path}. Please train it first.")
            return

    # Test tokenizer
    dual_print(f"\nTesting tokenizer for {domain}...")
    subprocess.run([
        "python3", "test_tokenizer.py",
        "--tokenizer_path", tokenizer_path,
        "--train_file", train_file,
        "--test_file", dev_file
    ], check=True)

    # Train NER model
    dual_print(f"\nTraining NER model for {domain}...")
    subprocess.run([
        "python3", "train_ner_model.py",
        "--tokenizer_path", tokenizer_path,
        "--train_file", ner_train,
        "--dev_file", ner_dev
    ], check=True)

    dual_print("\nAll steps completed.")

if __name__ == "__main__":
    main()
