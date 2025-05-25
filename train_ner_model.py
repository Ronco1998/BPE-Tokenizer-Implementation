import argparse
import os
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from base_tokenizer import BaseTokenizer


# Dataset class for NER data
class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.encoded_texts = []
        self.encoded_labels = []
        self.word_to_subtoken = []  # For each sentence: maps word_idx -> first subtoken_idx for that word
        self.subtoken_to_word = []  # For each sentence: maps subtoken_idx -> primary word_idx it represents
        
        # Check if tokenizer has a space token
        self.space_token_id = None
        if hasattr(tokenizer, 'space_token') and tokenizer.space_token in tokenizer.token_to_id:
            self.space_token_id = tokenizer.token_to_id[tokenizer.space_token]
        
        for text, text_labels in zip(texts, labels):
            # Get the words and their positions in the original text
            words = text.split()
            if not words:
                continue
                
            word_spans = []  # (start_char, end_char) for each word
            char_pos = 0
            for word in words:
                # Find the word in the text, starting from char_pos
                word_start = text.find(word, char_pos)
                if word_start == -1:  # Word not found, use approximation
                    word_start = char_pos
                word_end = word_start + len(word)
                word_spans.append((word_start, word_end))
                char_pos = word_end + 1  # +1 to skip space
            
            # Encode the full text (allowing tokens to cross word boundaries)
            token_ids = tokenizer.encode(text)
            
            # Skip space tokens when tracking word boundaries (they're not part of any word)
            if self.space_token_id is not None:
                # Create a mapping from token position including spaces to position excluding spaces
                non_space_mapping = []
                cleaned_token_ids = []
                for i, token_id in enumerate(token_ids):
                    if token_id != self.space_token_id:
                        non_space_mapping.append(len(cleaned_token_ids))
                        cleaned_token_ids.append(token_id)
                    else:
                        non_space_mapping.append(-1)  # Skip space tokens
            else:
                # If no space token, use all tokens
                cleaned_token_ids = token_ids
                non_space_mapping = list(range(len(token_ids)))
            
            # Get the decoded representation of each non-space subtoken
            subtokens = [tokenizer.decode([token_id]) for token_id in cleaned_token_ids]
            
            # Map each subtoken to the word(s) it covers
            word_to_sub = []  # word_idx -> first subtoken that primarily covers it
            sub_to_word = [-1] * len(token_ids)  # subtoken_idx -> primary word it represents (-1 if none)
            
            # Simulate character positions based on concatenating subtokens
            # This is approximate but helps demonstrate the concept
            subtoken_start_chars = []
            char_pos = 0
            for subtoken in subtokens:
                subtoken_start_chars.append(char_pos)
                char_pos += len(subtoken)
            
            # For each word, find the first subtoken that overlaps with it
            for word_idx, (word_start, word_end) in enumerate(word_spans):
                # Find first non-space subtoken that overlaps this word
                first_subtoken = -1
                for cleaned_sub_idx, sub_start in enumerate(subtoken_start_chars):
                    sub_end = sub_start + len(subtokens[cleaned_sub_idx])
                    
                    # Check if this subtoken overlaps with the word
                    if max(sub_start, word_start) < min(sub_end, word_end):
                        # Subtoken overlaps with this word
                        if first_subtoken == -1:
                            # Map back to original token_ids index (including spaces)
                            for orig_idx, mapped_idx in enumerate(non_space_mapping):
                                if mapped_idx == cleaned_sub_idx:
                                    first_subtoken = orig_idx
                                    break
                            # Mark this subtoken as representing this word primarily
                            sub_to_word[first_subtoken] = word_idx
                
                if first_subtoken != -1:
                    word_to_sub.append(first_subtoken)
                else:
                    # Fallback - use a subtoken close to the word's position
                    closest_cleaned_idx = min(range(len(subtoken_start_chars)), 
                                      key=lambda i: abs(subtoken_start_chars[i] - word_start))
                    # Map back to original token_ids index
                    for orig_idx, mapped_idx in enumerate(non_space_mapping):
                        if mapped_idx == closest_cleaned_idx:
                            closest_idx = orig_idx
                            break
                    else:
                        closest_idx = 0  # Default if mapping fails
                    
                    word_to_sub.append(closest_idx)
                    sub_to_word[closest_idx] = word_idx
            
            # Create labels: each subtoken either gets its primary word's label or -100
            token_labels = [-100] * len(token_ids)  # Start with all -100
            for sub_idx, word_idx in enumerate(sub_to_word):
                if word_idx != -1:  # If this subtoken primarily represents a word
                    # Check if this is the first subtoken for this word
                    if word_to_sub[word_idx] == sub_idx:
                        token_labels[sub_idx] = text_labels[word_idx]
            
            # Store results
            self.encoded_texts.append(token_ids)
            self.encoded_labels.append(token_labels)
            self.word_to_subtoken.append(word_to_sub)
            self.subtoken_to_word.append(sub_to_word)
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long),
            'word_to_subtoken': self.word_to_subtoken[idx],
            'subtoken_to_word': self.subtoken_to_word[idx],
        }


# Simple NER model
class NERModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        super(NERModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, input_ids):
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(embeddings)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Classify each token
        logits = self.classifier(lstm_output)
        
        return logits


def pad_sequences(batch, pad_id: int = 0):
    """
    Pad sequences in a batch to the same length
    
    Args:
        batch: Batch of sequences
        pad_id: ID to use for padding
        
    Returns:
        Padded batch and mask
    """
    max_len = max(len(seq) for seq in batch)
    padded_batch = []
    mask = []
    
    for seq in batch:
        padded_seq = seq + [pad_id] * (max_len - len(seq))
        padded_batch.append(padded_seq)
        mask.append([1] * len(seq) + [0] * (max_len - len(seq)))
    
    return padded_batch, mask


def collate_fn(batch):
    """
    Collate function for DataLoader
    
    Args:
        batch: Batch of samples
        
    Returns:
        Collated batch with padded sequences
    """
    input_ids = [item['input_ids'].tolist() for item in batch]
    labels = [item['labels'].tolist() for item in batch]
    word_to_subtoken = [item['word_to_subtoken'] for item in batch]
    subtoken_to_word = [item['subtoken_to_word'] for item in batch]
    input_ids, attention_mask = pad_sequences(input_ids, pad_id=0)
    labels, _ = pad_sequences(labels, pad_id=-100)
    # No padding for word_to_subtoken/subtoken_to_word (used only for evaluation, not for loss)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
        'labels': torch.tensor(labels, dtype=torch.long),
        'word_to_subtoken': word_to_subtoken,
        'subtoken_to_word': subtoken_to_word,
    }


def read_ner_data(file_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Read NER data from a file
    
    Args:
        file_path: Path to the NER data file
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    current_text = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line indicates end of sentence
            if not line:
                if current_text:
                    texts.append(' '.join(current_text))
                    labels.append(current_labels)
                    current_text = []
                    current_labels = []
                continue
            
            # Parse token and tag
            parts = line.split('\t')
            if len(parts) == 2:
                token, tag = parts
                # Convert tag to binary (1 if entity, 0 if not)
                label = 1 if tag != 'O' else 0
                current_text.append(token)
                current_labels.append(label)
    
    # Add the last sentence if it exists
    if current_text:
        texts.append(' '.join(current_text))
        labels.append(current_labels)
    
    return texts, labels


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate the model on a dataset
    
    Args:
        model: The NER model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            word_to_subtoken = batch['word_to_subtoken']
            # Forward pass
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.numpy()
            
            # For each sentence in batch
            for i in range(len(preds)):
                # For each word, get prediction from its first subtoken
                for word_idx, subtoken_idx in enumerate(word_to_subtoken[i]):
                    # Skip if subtoken_idx is out of bounds (due to padding)
                    if subtoken_idx >= len(preds[i]):
                        continue
                    
                    pred_label = preds[i][subtoken_idx]
                    
                    # For evaluation, we compare each word's prediction against its gold label
                    # We get the gold label for this word (from its first subtoken)
                    gold_label = labels_np[i][subtoken_idx]
                    
                    # Only consider valid labels (skip -100)
                    if gold_label == -100:
                        continue
                    
                    all_preds.append(pred_label)
                    all_labels.append(gold_label)
    
    # Calculate metrics
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_preds)
    accuracy = correct / total if total > 0 else 0
    
    true_positives = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    pred_positives = sum(p == 1 for p in all_preds)
    actual_positives = sum(l == 1 for l in all_labels)
    
    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_ner_model(
    tokenizer_path: str,
    train_file: str,
    dev_file: str,
    output_dir: str,
    batch_size: int = 32,
    lr: float = 0.001,
    num_epochs: int = 5
):
    """
    Train a NER model
    
    Args:
        tokenizer_path: Path to the trained tokenizer
        train_file: Path to the training data
        dev_file: Path to the development data
        output_dir: Directory to save the trained model
        batch_size: Batch size for training
        lr: Learning rate
        num_epochs: Number of training epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BaseTokenizer.load(tokenizer_path)
    
    # Load data
    print(f"Loading training data from {train_file}")
    train_texts, train_labels = read_ner_data(train_file)
    
    print(f"Loading dev data from {dev_file}")
    dev_texts, dev_labels = read_ner_data(dev_file)
    
    # Create datasets
    print("Creating datasets")
    train_dataset = NERDataset(train_texts, train_labels, tokenizer)
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize model
    print("Initializing model")
    model = NERModel(tokenizer.get_vocab_size(), num_classes=2)  # Binary classification
    model.to(device)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padded tokens
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss calculation
            B, T, C = logits.shape  # Batch size, sequence length, num classes
            loss = loss_fn(logits.view(-1, C), labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on dev set
        metrics = evaluate_model(model, dev_dataloader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train loss: {total_loss / len(train_dataloader):.4f}")
        print(f"  Dev accuracy: {metrics['accuracy']:.4f}")
        print(f"  Dev precision: {metrics['precision']:.4f}")
        print(f"  Dev recall: {metrics['recall']:.4f}")
        print(f"  Dev F1: {metrics['f1']:.4f}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1
            }, model_path)
            print(f"  Saved new best model with F1: {best_f1:.4f}")
    
    print(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained tokenizer")
    parser.add_argument("--train_file", type=str, default="data/ner_data/train_binary.tagged", help="Path to the training data")
    parser.add_argument("--dev_file", type=str, default="data/ner_data/dev_binary.tagged", help="Path to the development data")
    # Fixed parameters (not modifiable by students)
    output_dir = "models"
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 20
    args = parser.parse_args()
    train_ner_model(
        args.tokenizer_path,
        args.train_file,
        args.dev_file,
        output_dir,
        batch_size,
        learning_rate,
        num_epochs
    ) 