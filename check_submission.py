import os
import sys
import glob
import pickle
import importlib.util
import zipfile
import re
import shutil
import tempfile
from base_tokenizer import BaseTokenizer
import torch
from train_ner_model import train_ner_model, read_ner_data, evaluate_model, NERModel, NERDataset, collate_fn
from torch.utils.data import DataLoader

# Constants for folder structure
REQUIRED_FOLDERS = ["code", "trained_tokenizer"]
REQUIRED_TOKENIZER_FILES = [
    "trained_tokenizer/tokenizer_1.pkl",
    "trained_tokenizer/tokenizer_2.pkl",
    "trained_tokenizer/tokenizer_3.pkl"
]
REQUIRED_COMP_FILES = []  # Add any required comparison files here

DOMAIN_TEST_FILE = "domain_test.txt"  # Should be in the root or specify path
NER_TRAIN_FILE = "data/ner_data/train.tagged"
NER_DEV_FILE = "data/ner_data/dev.tagged"


def extract_student_ids(zip_filename):
    """Extract student IDs from the zip filename."""
    match = re.match(r'HW2_(\d+)_(\d+)\.zip', os.path.basename(zip_filename))
    if match:
        return match.group(1), match.group(2)
    return None, None


def unzip_submission(zip_path, extract_dir=None):
    """
    Unzip the submission file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to (if None, use a temp dir)
        
    Returns:
        Path to the extracted directory, student IDs
    """
    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip file not found: {zip_path}")
        return None, (None, None)
    
    # Extract student IDs from filename
    id1, id2 = extract_student_ids(zip_path)
    if id1 is None or id2 is None:
        print(f"[ERROR] Could not extract student IDs from filename: {zip_path}")
        print("Filename should be in format: HW2_ID1_ID2.zip")
        return None, (None, None)
    
    # Create a temporary directory if extract_dir is not provided
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp()
    
    # Unzip the file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[OK] Successfully extracted {zip_path} to {extract_dir}")
        return extract_dir, (id1, id2)
    except Exception as e:
        print(f"[ERROR] Failed to extract zip file: {e}")
        return None, (None, None)


def check_structure(root_dir, student_ids):
    """
    Check the submission structure.
    
    Args:
        root_dir: Root directory of the extracted submission
        student_ids: Tuple of student IDs (id1, id2)
        
    Returns:
        Boolean indicating if structure is OK
    """
    print("Checking submission structure...")
    all_ok = True
    id1, id2 = student_ids
    
    # Check required folders
    for folder in REQUIRED_FOLDERS:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"[ERROR] Missing folder: {folder}")
            all_ok = False
        else:
            print(f"[OK] Found folder: {folder}")
    
    # Check required tokenizer files
    for file in REQUIRED_TOKENIZER_FILES:
        file_path = os.path.join(root_dir, file)
        if not os.path.isfile(file_path):
            print(f"[ERROR] Missing tokenizer file: {file}")
            all_ok = False
        else:
            print(f"[OK] Found tokenizer file: {file}")
    
    # Check for the report with the student IDs in the filename
    report_name = f"report_{id1}_{id2}.pdf"
    report_path = os.path.join(root_dir, report_name)
    if not os.path.isfile(report_path):
        print(f"[ERROR] Missing PDF report: {report_name}")
        all_ok = False
    else:
        print(f"[OK] Found PDF report: {report_name}")
    
    # Check comparison files
    for file in REQUIRED_COMP_FILES:
        file_path = os.path.join(root_dir, file)
        if not os.path.isfile(file_path):
            print(f"[ERROR] Missing comparison file: {file}")
            all_ok = False
        else:
            print(f"[OK] Found comparison file: {file}")
    
    return all_ok


def test_tokenizer_efficiency(tokenizer, domain_test_file):
    """
    Test tokenizer efficiency on domain test file.
    
    Args:
        tokenizer: Tokenizer to test
        domain_test_file: Path to the domain test file
        
    Returns:
        Efficiency metric (or None if test failed)
    """
    print(f"\nTesting tokenizer efficiency on {domain_test_file}...")
    if not os.path.isfile(domain_test_file):
        print(f"[ERROR] Domain test file not found: {domain_test_file}")
        return None
    
    with open(domain_test_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    
    if total_chars == 0:
        print("[ERROR] Domain test file is empty.")
        return None
    
    efficiency = total_tokens / total_chars
    print(f"[RESULT] Tokens per character: {efficiency:.4f}")
    return efficiency


def test_tokenizer_methods(tokenizer):
    """
    Test basic tokenizer methods.
    
    Args:
        tokenizer: Tokenizer to test
        
    Returns:
        Boolean indicating if tests passed
    """
    print("Testing tokenizer methods...")
    try:
        test_str = "Hello world!"
        ids = tokenizer.encode(test_str)
        s = tokenizer.decode(ids)
        print(f"[OK] encode/decode methods work. Example: '{test_str}' -> {ids} -> '{s}'")
    except NotImplementedError:
        print("[ERROR] encode or decode not implemented!")
        return False
    except Exception as e:
        print(f"[ERROR] Exception in encode/decode: {e}")
        return False
    return True


def train_and_eval_ner(tokenizer_path, train_file, dev_file):
    """
    Train and evaluate NER model with the tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer file
        train_file: Path to the training data
        dev_file: Path to the development data
        
    Returns:
        Evaluation metrics (or None if evaluation failed)
    """
    print(f"\nTraining and evaluating NER model with tokenizer: {tokenizer_path}")
    
    # Load tokenizer
    try:
        tokenizer = BaseTokenizer.load(tokenizer_path)
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer: {e}")
        return None
    
    # Prepare data
    train_texts, train_labels = read_ner_data(train_file)
    dev_texts, dev_labels = read_ner_data(dev_file)
    train_dataset = NERDataset(train_texts, train_labels, tokenizer)
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate_fn)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModel(tokenizer.get_vocab_size(), num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Train 1 epoch (quick check)
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(-1, C), labels.view(-1))
        loss.backward()
        optimizer.step()
        break  # Only one batch for speed
    
    # Evaluate
    metrics = evaluate_model(model, dev_loader, device)
    print(f"[RESULT] Dev set metrics: {metrics}")
    return metrics


def check_submission_zip(zip_path):
    """
    Check a submission zip file.
    
    Args:
        zip_path: Path to the submission zip file
    """
    print(f"=== Checking Submission: {zip_path} ===")
    
    # Extract the zip file
    extract_dir, student_ids = unzip_submission(zip_path)
    if extract_dir is None:
        print("[FAIL] Failed to extract submission zip. Exiting.")
        return
    
    # Store original working directory
    orig_dir = os.getcwd()
    
    try:
        # Move to the extracted directory for all operations
        os.chdir(extract_dir)
        
        # Check structure
        structure_ok = check_structure(extract_dir, student_ids)
        if not structure_ok:
            print("[FAIL] Submission structure is incorrect. Please fix the above errors.")
            return
        
        # Test each tokenizer
        for i in range(1, 4):
            tokenizer_path = os.path.join(extract_dir, f"trained_tokenizer/tokenizer_{i}.pkl")
            print(f"\n--- Checking Tokenizer {i} ---")
            
            try:
                tokenizer = BaseTokenizer.load(tokenizer_path)
            except Exception as e:
                print(f"[ERROR] Could not load tokenizer {i}: {e}")
                continue
            
            # Test methods
            if not test_tokenizer_methods(tokenizer):
                continue
            
            # Test efficiency
            domain_test_path = os.path.join(extract_dir, DOMAIN_TEST_FILE)
            test_tokenizer_efficiency(tokenizer, domain_test_path)
            
            # Train and eval NER
            train_file = os.path.join(extract_dir, NER_TRAIN_FILE)
            dev_file = os.path.join(extract_dir, NER_DEV_FILE)
            train_and_eval_ner(tokenizer_path, train_file, dev_file)
        
        print("\n[INFO] Submission check complete.")
        print("If you see only [OK] and [RESULT] messages above, your submission is valid!")
        
    finally:
        # Change back to the original directory
        os.chdir(orig_dir)
        
        # Clean up the temporary directory (uncomment in production)
        # shutil.rmtree(extract_dir)


def main():
    """
    Main function. Handle arguments and run checks.
    """
    if len(sys.argv) < 2:
        print("Usage: python check_submission.py <path_to_zip_file>")
        print("Example: python check_submission.py HW2_123456789_987654321.zip")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    check_submission_zip(zip_path)


if __name__ == "__main__":
    main() 