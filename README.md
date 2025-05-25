# NLP Homework: BPE Tokenizer Implementation

This homework assignment focuses on implementing different versions of the Byte Pair Encoding (BPE) tokenizer algorithm and evaluating their performance on a downstream NER (Named Entity Recognition) task.

## Overview

In this assignment, you'll:
1. Implement your own BPE tokenizer by extending the BaseTokenizer class
2. Train your tokenizer on domain-specific data
3. Evaluate your tokenizer on a binary NER task (entity vs. non-entity)
4. Compare different tokenization approaches based on:
   - Encoding speed
   - Efficiency (tokens needed for new data)
   - Performance on the downstream NER task

## File Structure

- `base_tokenizer.py`: Base abstract tokenizer class that you'll extend
- `train_tokenizer.py`: Script to train and save a tokenizer
- `train_ner_model.py`: Script to train a NER model using a trained tokenizer
- `data/`: Directory containing training data
  - `domain_1.txt`, `domain_2.txt`: Domain-specific text data for tokenizer training
  - `ner_data/`: NER task data
    - `train_1_binary.tagged`, `train_2_binary.tagged`: Training data for NER
    - `dev_1_binary.tagged`, `dev_2_binary.tagged`: Development data for NER

## Instructions

### Step 1: Implement Your Tokenizer

Create a new file (e.g., `my_tokenizer.py`) that implements a BPE tokenizer by extending the BaseTokenizer class. Your implementation should include the following methods:

```python
def train(self, texts: List[str]) -> None:
    # Train your tokenizer on the given texts
    pass

def encode(self, text: str) -> List[int]:
    # Convert text to token IDs
    pass

def decode(self, token_ids: List[int]) -> str:
    # Convert token IDs back to text
    pass
```

You're free to experiment with different BPE variations and optimizations.

### Step 2: Train Your Tokenizer

Train your tokenizer using the provided script:

```bash
python train_tokenizer.py --domain_file data/domain_1.txt --output_dir tokenizers --vocab_size 5000
```

This will save your trained tokenizer to the specified output directory.

### Step 3: Train the NER Model

Train the NER model using your tokenizer:

```bash
python train_ner_model.py --tokenizer_path tokenizers/tokenizer.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged
```

### Step 4: Evaluate and Experiment

Try different tokenization approaches and compare:
1. Measure encoding speed (tokens per second)
2. Compare vocabulary efficiency on unseen text
3. Evaluate NER performance using different tokenizer configurations

### Evaluation Criteria

Your implementation will be evaluated on:
1. **Correctness**: Does your tokenizer correctly implement the BPE algorithm?
2. **Speed**: How fast is your tokenizer at encoding new text?
3. **Efficiency**: How well does your tokenizer generalize to new data?
4. **NER Performance**: How well does a model using your tokenizer perform on the NER task?

## Notes on Subword Tokenization for NER

For the NER task, we'll use "first-subtoken labeling" - only the first subtoken of a word receives the entity label, while other subtokens are marked as non-entities (0). 

### Example of Subword Tokenization

Consider the sentence: "Steve Jobs ate an apple."

A traditional word tokenizer might split it into: ["Steve", "Jobs", "ate", "an", "apple", "."]

However, a subword tokenizer might split it across word boundaries:
["Steve J", "ob", "s ate", " an", " apple", "."]

In this case, for the NER task:
- If "Steve Jobs" is an entity (label 1):
  - Only the first subtoken "Steve J" receives label 1
  - The other subtokens "ob" and "s ate" receive label -100 (ignored during training)
- For non-entity words like "an" and "apple":
  - Their first subtokens receive label 0
  - Any subsequent subtokens receive label -100

This approach handles cases where tokenization doesn't align with word boundaries, which is common in subword tokenization methods like BPE.

### Step 5: Test Your Tokenizer Manually

You can test your tokenizer manually using the test script:

```bash
python test_tokenizer.py --tokenizer_path tokenizers/tokenizer.pkl --train_file data/domain_1.txt --test_file data/domain_2.txt
```

This will print efficiency, reconstruction, and encoding/decoding examples for your tokenizer.

Good luck! 