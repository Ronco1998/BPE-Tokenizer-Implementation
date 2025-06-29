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
REQUIRED_FOLDERS = ["code", "trained_tokenizers"]
REQUIRED_TOKENIZER_FILES = [
    "trained_tokenizers/tokenizer_1.pkl",
    "trained_tokenizers/tokenizer_2.pkl",
    "trained_tokenizers/tokenizer_3.pkl"
]
 # Add any required comparison files here

DOMAIN_TEST_FILE = "domain_test.txt"  # Should be in the root or specify path
NER_TRAIN_FILE = "../data_clean/ner_data/train_1_binary.tagged"
NER_DEV_FILE = "../data_clean/ner_data/dev_1_binary.tagged"


def extract_student_ids(zip_filename):
    """Extract student IDs from the zip filename. Handles both one and two ID formats."""
    base = os.path.basename(zip_filename)
    match_two = re.match(r'HW2_(\d+)_(\d+)\.zip', base)
    if match_two:
        return match_two.group(1), match_two.group(2)
    match_one = re.match(r'HW2_(\d+)\.zip', base)
    if match_one:
        return match_one.group(1), None
    return None, None


def unzip_submission(zip_path, extract_dir=None):
    """
    Unzip the submission file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to (if None, use a folder named after the zip file in the current directory)
        
    Returns:
        Path to the extracted directory, student IDs
    """
    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip file not found: {zip_path}")
        return None, (None, None)
    
    # Extract student IDs from filename
    id1, id2 = extract_student_ids(zip_path)
    print(f"[INFO] Extracted student IDs: id1={id1}, id2={id2}")
    if id1 is None:
        print(f"[ERROR] Could not extract student ID(s) from filename: {zip_path}")
        print("Filename should be in format: HW2_ID1_ID2.zip or HW2_ID.zip")
        return None, (None, None)
    
    # Create a directory in the current working directory if extract_dir is not provided
    if extract_dir is None:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_dir = os.path.abspath(base_name)
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
    
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
    if id2 is not None:
        report_name = f"report_{id1}_{id2}.pdf"
    else:
        report_name = f"report_{id1}.pdf"
    report_path = os.path.join(root_dir, report_name)
    if not os.path.isfile(report_path):
        print(f"[ERROR] Missing PDF report: {report_name}")
        all_ok = False
    else:
        print(f"[OK] Found PDF report: {report_name}")
    
   
    
    return all_ok




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
    
    # Check for train_file and dev_file, try parent dir and cwd if not found
    attempted_train_files = [train_file]
    if not os.path.isfile(train_file):
        alt_train_file = os.path.abspath(os.path.join(os.path.dirname(train_file), '..', os.path.basename(train_file)))
        attempted_train_files.append(alt_train_file)
        if os.path.isfile(alt_train_file):
            print(f"[INFO] Training file not found in extracted dir, using parent dir: {alt_train_file}")
            train_file = alt_train_file
        else:
            cwd_train_file = os.path.join(os.getcwd(), os.path.basename(train_file))
            attempted_train_files.append(cwd_train_file)
            if os.path.isfile(cwd_train_file):
                print(f"[INFO] Training file not found in extracted or parent dir, using cwd: {cwd_train_file}")
                train_file = cwd_train_file
            else:
                print(f"[ERROR] Training file not found. Attempted paths:")
                for p in attempted_train_files:
                    print(f"  - {p}")
                return None
    
    attempted_dev_files = [dev_file]
    if not os.path.isfile(dev_file):
        alt_dev_file = os.path.abspath(os.path.join(os.path.dirname(dev_file), '..', os.path.basename(dev_file)))
        attempted_dev_files.append(alt_dev_file)
        if os.path.isfile(alt_dev_file):
            print(f"[INFO] Dev file not found in extracted dir, using parent dir: {alt_dev_file}")
            dev_file = alt_dev_file
        else:
            cwd_dev_file = os.path.join(os.getcwd(), os.path.basename(dev_file))
            attempted_dev_files.append(cwd_dev_file)
            if os.path.isfile(cwd_dev_file):
                print(f"[INFO] Dev file not found in extracted or parent dir, using cwd: {cwd_dev_file}")
                dev_file = cwd_dev_file
            else:
                print(f"[ERROR] Dev file not found. Attempted paths:")
                for p in attempted_dev_files:
                    print(f"  - {p}")
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
            tokenizer_path = os.path.join(extract_dir, f"trained_tokenizers/tokenizer_{i}.pkl")
            print(f"\n--- Checking Tokenizer {i} ---")
            
            try:
                tokenizer = BaseTokenizer.load(tokenizer_path)
            except Exception as e:
                print(f"[ERROR] Could not load tokenizer {i}: {e}")
                continue
            
            # Test methods
            if not test_tokenizer_methods(tokenizer):
                continue
            
    
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