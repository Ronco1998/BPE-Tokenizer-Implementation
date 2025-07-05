from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any
import re
import random
# import numpy as np
from base_tokenizer import BaseTokenizer


class NERSpecializedTokenizer(BaseTokenizer):
    """
    Custom BPE Tokenizer specifically designed for Named Entity Recognition tasks.
    Optimized for handling Twitter data, news headlines, and mixed domains.
    
    Key NER-focused features:
    - Preserves capitalization patterns crucial for entity recognition
    - Handles hashtags, mentions, and social media conventions
    - Optimized merge selection based on entity boundary patterns
    - Fast training suitable for large datasets
    """
    
    def __init__(self, vocab_size: int = 4000, max_iterations: int = 2000, min_freq_threshold: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_iterations = max_iterations
        self.min_freq_threshold = min_freq_threshold
        self.merge_operations = []  # Store BPE merge rules
        self.word_tokenization_cache = {}  # Cache for faster processing
        
    def _normalize_text(self, text: str) -> str:
        """
        Light text normalization that preserves NER-important features.
        """
        # Only normalize excessive whitespace, keep everything else
        return ' '.join(text.split())
    
    def _get_word_level_tokens(self, text: str) -> List[str]:
        """
        Initial word-level tokenization that preserves entity boundaries.
        """
        # Split on whitespace but keep social media patterns intact
        words = text.split()
        word_tokens = []
        
        for word in words:
            # Convert word to character sequence with end marker
            chars = list(word) + ['</w>']
            word_tokens.append(chars)
            
        return word_tokens
    
    def _collect_character_pairs(self, word_tokens: List[List[str]]) -> Counter:
        """
        Collect all adjacent character pairs from the tokenized words.
        """
        pair_counts = Counter()
        
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1
                
        return pair_counts
    
    def _calculate_ner_relevance_score(self, pair: Tuple[str, str], frequency: int) -> float:
        """
        Calculate relevance score for a character pair based on NER usefulness.
        """
        char1, char2 = pair
        score = frequency  # Base frequency score
        
        # Bonus for capitalization patterns (important for named entities)
        if char1.isupper() and char2.islower():
            score += frequency * 0.3  # Title case pattern
        
        # Bonus for maintaining word boundaries with capitals
        if char1.isupper() and char2.isupper():
            score += frequency * 0.2  # Acronym pattern
            
        # Bonus for social media patterns
        if char1 == '#' or char1 == '@':
            score += frequency * 0.4  # Hashtag/mention patterns
            
        # Bonus for alphanumeric combinations (dates, codes, etc.)
        if char1.isalpha() and char2.isdigit():
            score += frequency * 0.25
        if char1.isdigit() and char2.isalpha():
            score += frequency * 0.25
            
        # Penalty for splitting common letter combinations
        common_pairs = {'th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ha', 'to'}
        if char1 + char2 in common_pairs:
            score += frequency * 0.1  # Small bonus for common patterns
            
        return score
    
    def _find_best_merge_pair(self, pair_counts: Counter) -> Optional[Tuple[str, str]]:
        """
        Find the best pair to merge based on NER relevance scoring.
        """
        if not pair_counts:
            return None
            
        best_pair = None
        best_score = -1
        
        for pair, count in pair_counts.items():
            if count >= self.min_freq_threshold:
                score = self._calculate_ner_relevance_score(pair, count)
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    
        return best_pair
    
    def _apply_merge(self, word_tokens: List[List[str]], merge_pair: Tuple[str, str]) -> List[List[str]]:
        """
        Apply a merge operation to all word tokens.
        """
        first, second = merge_pair
        new_word_tokens = []
        
        for word in word_tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    # Merge the pair
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tokens.append(new_word)
            
        return new_word_tokens
    
    def _build_vocabulary_from_tokens(self, word_tokens: List[List[str]]) -> None:
        """
        Build the final vocabulary from processed word tokens.
        """
        # Collect all unique tokens
        all_tokens = set()
        for word in word_tokens:
            all_tokens.update(word)
            
        # Sort tokens for consistent ordering
        sorted_tokens = sorted(all_tokens)
        
        # Add to vocabulary (special tokens already added in base class)
        current_id = len(self.token_to_id)
        for token in sorted_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
    
    def train(self, texts: List[str]) -> None:
        """
        Train the NER-specialized BPE tokenizer on the given texts.
        """
        print(f"Training NER-specialized tokenizer on {len(texts)} texts...")
        print(f"Target vocabulary size: {self.vocab_size}")
        
        # Step 1: Normalize and tokenize all texts
        print("Step 1: Initial tokenization...")
        all_word_tokens = []
        for text in texts:
            normalized_text = self._normalize_text(text)
            word_tokens = self._get_word_level_tokens(normalized_text)
            all_word_tokens.extend(word_tokens)
        
        print(f"Created {len(all_word_tokens)} word tokens")
        
        # Step 2: Iterative BPE training
        print("Step 2: Learning BPE merges...")
        iteration = 0
        
        while iteration < self.max_iterations and len(self.token_to_id) < self.vocab_size:
            # Collect current character pairs
            pair_counts = self._collect_character_pairs(all_word_tokens)
            
            # Find best merge
            best_pair = self._find_best_merge_pair(pair_counts)
            if best_pair is None:
                print("No more valid merges found")
                break
                
            # Apply merge
            all_word_tokens = self._apply_merge(all_word_tokens, best_pair)
            self.merge_operations.append(best_pair)
            
            iteration += 1
            if iteration % 100 == 0:
                current_vocab_size = len(set(token for word in all_word_tokens for token in word))
                print(f"Iteration {iteration}: Merged {best_pair}, vocab size ≈ {current_vocab_size}")
        
        # Step 3: Build final vocabulary
        print("Step 3: Building final vocabulary...")
        self._build_vocabulary_from_tokens(all_word_tokens)
        
        print(f"Training complete! Final vocabulary size: {len(self.token_to_id)}")
        print(f"Applied {len(self.merge_operations)} merge operations")
    
    def _apply_bpe_to_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single word.
        """
        if word in self.word_tokenization_cache:
            return self.word_tokenization_cache[word]
            
        # Start with character-level representation
        tokens = list(word) + ['</w>']
        
        # Apply each merge operation in order
        for merge_pair in self.merge_operations:
            first, second = merge_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        # Cache result for faster future lookups
        self.word_tokenization_cache[word] = tokens
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs using the trained BPE model.
        """
        if not text.strip():
            return []
            
        normalized_text = self._normalize_text(text)
        words = normalized_text.split()
        
        token_ids = []
        for word in words:
            word_tokens = self._apply_bpe_to_word(word)
            for token in word_tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Use UNK token for unknown tokens
                    token_ids.append(self.special_tokens["[UNK]"])
                    
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("[UNK]")
        
        # Reconstruct text by joining tokens and handling end-of-word markers
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        
        return text.strip()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the trained tokenizer.
        """
        return {
            "vocabulary_size": len(self.token_to_id),
            "merge_operations": len(self.merge_operations),
            "special_tokens": len(self.special_tokens),
            "cache_size": len(self.word_tokenization_cache)
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze how the tokenizer processes a given text.
        """
        words = self._normalize_text(text).split()
        analysis = {
            "original_text": text,
            "normalized_text": self._normalize_text(text),
            "word_count": len(words),
            "word_tokenizations": {},
            "total_tokens": 0,
            "token_ids": self.encode(text)
        }
        
        for word in words:
            tokens = self._apply_bpe_to_word(word)
            analysis["word_tokenizations"][word] = tokens
            analysis["total_tokens"] += len(tokens)
            
        return analysis
