from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from base_tokenizer import BaseTokenizer
import re
import heapq
# filepath: c:\Users\ronic\VS_Coding\NLP\HW\HW2\train_tokenizer.py
import sys
print(sys.executable)
print(sys.path)
import numpy as np
# ...existing code...
import random


class MyTokenizer(BaseTokenizer):
    """
    Enhanced BPE Tokenizer with space handling for cross-word tokens
    Optimized for speed and NER performance
    LIMITED TO WORD BIGRAMS (max 2 words per token)
    """
    
    def __init__(self, vocab_size: int = 4000, max_merges: int = 1000, min_token_freq: int = 2, random_seed: int = 42):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_merges = max_merges
        self.min_token_freq = min_token_freq
        self.bpe_merges = []
        self.word_cache = {}
        self.text_cache = {}
        self.char_offsets = {}  # Cache character offsets for faster alignment
        
        # Always set the same random seed for reproducibility
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"Random seed set to: {random_seed} (deterministic tokenizer)")
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text while preserving spaces, emojis, and special characters
        Light preprocessing that maintains important information
        """
        # Basic whitespace normalization only
        text = ' '.join(text.split())
        return text
    
    def _extract_substrings_optimized(self, texts: List[str], max_words: int = 2) -> Dict[str, int]:
        """
        Extract substrings up to max_words (word bigrams)
        """
        substring_counts = defaultdict(int)
        
        # Process texts in batches for better memory usage
        batch_size = 1000
        processed_count = 0
        
        # Limit processing for very large datasets
        max_texts_to_process = min(len(texts), 50000)  # Process at most 50k texts
        
        for batch_start in range(0, max_texts_to_process, batch_size):
            batch_end = min(batch_start + batch_size, max_texts_to_process)
            batch_texts = texts[batch_start:batch_end]
            
            for text in batch_texts:
                preprocessed = self._preprocess_text(text)
                
                # Skip very long texts to avoid memory issues
                if len(preprocessed) > 1000:
                    continue
                
                # Extract word-level substrings
                words = preprocessed.split()
                
                # Single words
                for word in words:
                    if self._is_meaningful_substring(word):
                        substring_counts[word] += 1
                
                # Word bigrams (pairs of consecutive words)
                for i in range(len(words) - 1):
                    bigram = words[i] + ' ' + words[i + 1]
                    if self._is_meaningful_substring(bigram):
                        substring_counts[bigram] += 1
                
                # Also extract character-level substrings for single characters and cross-word patterns
                text_len = len(preprocessed)
                for i in range(text_len):
                    # Single characters
                    char = preprocessed[i]
                    substring_counts[char] += 1
                    
                    # Character pairs that might span word boundaries
                    if i < text_len - 1:
                        char_pair = preprocessed[i:i+2]
                        # Only include if it contains a space (cross-word boundary)
                        if ' ' in char_pair:
                            substring_counts[char_pair] += 1
            
            processed_count += len(batch_texts)
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count}/{max_texts_to_process} texts...")
        
        print(f"Extracted {len(substring_counts)} unique substrings (up to word bigrams)")
        return dict(substring_counts)
    
    def _is_meaningful_substring(self, substring: str) -> bool:
        """
        Check if a substring is meaningful for NER tasks
        Allows up to 2 words (word bigrams)
        """
        # Basic length check
        if len(substring.strip()) == 0:
            return False
        
        # Count words in the substring
        words = substring.strip().split()
        num_words = len(words)
        
        # Allow single characters (including space)
        if len(substring) == 1:
            return True
        
        # Allow character pairs that span word boundaries (contain space)
        if len(substring) == 2 and ' ' in substring:
            return True
        
        # For word-level tokens, limit to 2 words maximum
        if num_words > 2:
            return False
        
        # Allow single words
        if num_words == 1:
            word = words[0]
            # Reject very short words unless they're common
            if len(word) == 1:
                return word.isalnum() or word in ".,!?;:"
            return True
        
        # Allow word bigrams (2 words)
        if num_words == 2:
            # Both words should be meaningful
            word1, word2 = words
            if len(word1) >= 1 and len(word2) >= 1:
                return True
        
        return False

    def _get_linguistic_features(self, substring: str) -> Dict[str, float]:
        """
        Calculate linguistic features that are helpful for NER
        Enhanced for word-level patterns
        """
        features = {}
        
        # Basic features
        features['length'] = len(substring)
        features['word_count'] = len(substring.strip().split())
        features['has_space'] = 1.0 if ' ' in substring else 0.0
        features['starts_with_space'] = 1.0 if substring.startswith(' ') else 0.0
        features['ends_with_space'] = 1.0 if substring.endswith(' ') else 0.0
        
        # Character type features
        features['has_upper'] = 1.0 if any(c.isupper() for c in substring) else 0.0
        features['has_lower'] = 1.0 if any(c.islower() for c in substring) else 0.0
        features['has_digit'] = 1.0 if any(c.isdigit() for c in substring) else 0.0
        features['has_punct'] = 1.0 if any(not c.isalnum() and c != ' ' for c in substring) else 0.0
        
        # Word-level features for NER
        words = substring.strip().split()
        if len(words) == 1:
            word = words[0]
            features['is_title_case'] = 1.0 if word.istitle() else 0.0
            features['is_all_caps'] = 1.0 if word.isupper() and word.isalpha() else 0.0
        elif len(words) == 2:
            word1, word2 = words
            features['is_title_case'] = 1.0 if (word1.istitle() or word2.istitle()) else 0.0
            features['is_all_caps'] = 1.0 if (word1.isupper() or word2.isupper()) else 0.0
            features['both_title'] = 1.0 if (word1.istitle() and word2.istitle()) else 0.0
        else:
            features['is_title_case'] = 0.0
            features['is_all_caps'] = 0.0
            features['both_title'] = 0.0
        
        # Boundary features (important for NER)
        features['word_boundary'] = 1.0 if features['word_count'] == 2 else 0.0
        features['cross_word_char'] = 1.0 if (len(substring) == 2 and ' ' in substring) else 0.0
        
        return features

    def _calculate_ner_score(self, substring: str, freq: int) -> float:
        """
        Calculate a score for how useful a substring would be for NER tasks
        Enhanced for word-level patterns
        """
        features = self._get_linguistic_features(substring)
        
        # Base frequency score (with log scaling to prevent dominance)
        base_score = np.log(freq + 1)
        
        # NER-specific bonuses
        ner_bonus = 0.0
        
        # Strong bonuses for entity-related patterns
        if features.get('both_title', 0) > 0:
            ner_bonus += 100  # Both words title case - very likely entity
        
        if features['is_title_case'] > 0:
            ner_bonus += 50  # Title case is very important for NER
        
        if features['word_boundary'] > 0:
            ner_bonus += 40  # Word bigrams are very valuable for NER
        
        if features['cross_word_char'] > 0:
            ner_bonus += 30  # Cross-word character pairs help with alignment
        
        # Moderate bonuses
        if features['has_upper'] > 0 and features['has_lower'] > 0:
            ner_bonus += 15  # Mixed case often indicates entities
        
        if features['has_digit'] > 0:
            ner_bonus += 10  # Numbers often part of entities
        
        # Bonus for emoji patterns (can be contextually important)
        if any(ord(c) > 127 for c in substring):
            ner_bonus += 8  # Emojis and special chars
        
        # Length/word count bonuses
        if features['word_count'] == 2:
            ner_bonus += 25  # Strong preference for word bigrams
        elif features['word_count'] == 1 and features['length'] > 1:
            ner_bonus += 10  # Moderate bonus for single words
        elif features['length'] == 1:
            ner_bonus += 2   # Small bonus for characters
        
        return base_score + ner_bonus

    def train(self, texts: List[str]) -> None:
        """
        Enhanced BPE training optimized for speed and NER performance
        LIMITED TO WORD BIGRAMS
        """
        print("Starting optimized BPE training (word bigrams limit)...")
        
        # Step 1: Quick word frequency analysis (DETERMINISTIC)
        print("Step 1: Analyzing word frequencies...")
        word_freq = Counter()
        total_chars = 0
        
        # Use deterministic sampling instead of random
        sample_texts = texts[:1000]  # Always use first 1000 texts
        for text in sample_texts:
            words = text.split()
            word_freq.update(words)
            total_chars += len(text)
        
        print(f"Analyzed {len(word_freq)} unique words from sample")
        
        # Step 2: Extract substrings up to word bigrams (DETERMINISTIC)
        print(f"Step 2: Extracting substrings (up to word bigrams)...")
        
        substring_freq = self._extract_substrings_optimized(texts, max_words=2)
        
        # Step 3: Smart filtering with dynamic thresholds (DETERMINISTIC)
        print("Step 3: Filtering and scoring substrings...")
        total_texts = len(texts)
        min_freq = max(2, total_texts // 50000)  # More aggressive filtering
        
        # Filter and score substrings
        scored_substrings = []
        for substr, freq in substring_freq.items():
            if freq >= min_freq and self._is_meaningful_substring(substr):
                score = self._calculate_ner_score(substr, freq)
                scored_substrings.append((score, substr, freq))
        
        print(f"Scored {len(scored_substrings)} candidate substrings")
        
        # Step 4: Initialize vocabulary efficiently with guaranteed space token (DETERMINISTIC)
        print("Step 4: Initializing vocabulary...")
        all_chars = set()
        
        # Use deterministic sampling for character extraction
        sample_texts_for_chars = texts[:100]  # Always use first 100 texts
        for text in sample_texts_for_chars:
            preprocessed = self._preprocess_text(text)
            all_chars.update(preprocessed)
        
        # FORCE space character inclusion
        all_chars.add(' ')
        
        next_id = len(self.special_tokens)
        
        # Add space token FIRST to ensure it gets a low ID
        if ' ' not in self.token_to_id:
            self.token_to_id[' '] = next_id
            self.id_to_token[next_id] = ' '
            next_id += 1
            print(f"Added space token with ID: {self.token_to_id[' ']}")
        
        # Add all other characters in SORTED order for deterministic behavior
        for char in sorted(all_chars):
            if char not in self.token_to_id:  # Skip space since we already added it
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1
        
        print(f"Initial vocab size: {len(self.token_to_id)}")
        print(f"Space token ID: {self.token_to_id.get(' ', 'NOT FOUND')}")
        
        # Step 5: Add high-scoring substrings (up to word bigrams) - DETERMINISTIC SORTING
        print("Step 5: Adding high-scoring tokens...")
        
        # CRITICAL: Sort deterministically by score, then by substring for tie-breaking
        scored_substrings.sort(key=lambda x: (-x[0], x[1]))  # Descending score, ascending substring
        
        # Reserve space for BPE merges
        max_direct_additions = min(
            len(scored_substrings), 
            self.vocab_size - len(self.token_to_id) - 200  # Reserve 200 for BPE
        )
        
        added_tokens = 0
        word_bigram_tokens = 0
        cross_word_char_tokens = 0
        
        for score, substring, freq in scored_substrings[:max_direct_additions]:
            # ENFORCE WORD BIGRAM LIMIT
            if not self._is_meaningful_substring(substring):
                continue
                
            if len(self.token_to_id) >= self.vocab_size - 200:
                break
            if substring not in self.token_to_id:
                self.token_to_id[substring] = next_id
                self.id_to_token[next_id] = substring
                next_id += 1
                added_tokens += 1
                
                # Count different types
                words = substring.strip().split()
                if len(words) == 2:
                    word_bigram_tokens += 1
                elif len(substring) == 2 and ' ' in substring:
                    cross_word_char_tokens += 1
        
        print(f"Added {added_tokens} direct tokens:")
        print(f"  - Word bigrams: {word_bigram_tokens}")
        print(f"  - Cross-word chars: {cross_word_char_tokens}")
        
        # Step 6: Optimized BPE for remaining capacity (limited to word bigrams)
        if len(self.token_to_id) < self.vocab_size:
            remaining_capacity = self.vocab_size - len(self.token_to_id)
            print(f"Running BPE for {remaining_capacity} remaining slots (word bigram limit)...")
            self._run_optimized_bpe_word_bigrams(texts[:5000], next_id, remaining_capacity)
        
        print(f"Training complete! Final vocab size: {len(self.token_to_id)}")
        print(f"Word bigram tokens: {word_bigram_tokens}")
        
        # Verify space token exists and test it
        if ' ' in self.token_to_id:
            print(f"Space token confirmed: ID {self.token_to_id[' ']}")
            # Test space encoding/decoding
            test_space_encoding = self.encode(" ")
            test_space_decoding = self.decode(test_space_encoding)
            print(f"Space encoding test: ' ' -> {test_space_encoding} -> '{test_space_decoding}'")
        else:
            print("ERROR: No space token found!")
        
        # Print the 5 longest tokens (should be word bigrams now)
        self._print_longest_tokens()
        
        # Step 7: Optimized caching
        print("Building optimized caches...")
        self._build_optimized_caches(texts[:2000])  # Limit for memory efficiency

        # NEW
        analysis = self.get_token_analysis()
        print("\n=== TOKEN ANALYSIS ===")
        print(f"Total tokens: {analysis['total_tokens']}")
        print(f"Special tokens: {analysis['special_tokens']}")
        print(f"Single character tokens: {analysis['single_char_tokens']}")
        print(f"Single word tokens: {analysis['single_word_tokens']}")
        print(f"Word bigram tokens: {analysis['word_bigram_tokens']}")
        print(f"Cross-word char tokens: {analysis['cross_word_char_tokens']}")
        print("\nToken analysis complete.")
        
    def _print_longest_tokens(self):
        """
        Print the 5 longest tokens in the vocabulary
        Now should show word bigrams as the longest
        """
        # Get all non-special tokens with their lengths and word counts
        token_info = []
        for token, token_id in self.token_to_id.items():
            if token not in self.special_tokens:
                word_count = len(token.strip().split())
                char_length = len(token)
                token_info.append((word_count, char_length, token, token_id))
        
        # Sort by word count first, then by character length
        token_info.sort(reverse=True)
        longest_tokens = token_info[:5]
        
        print("\n=== 5 LONGEST TOKENS ===")
        for i, (word_count, char_length, token, token_id) in enumerate(longest_tokens, 1):
            # Clean representation for display
            display_token = repr(token)
            print(f"{i}. {word_count} words, {char_length} chars: {display_token} (ID: {token_id})")
        print("========================\n")
        
    def _run_optimized_bpe_word_bigrams(self, texts: List[str], start_id: int, max_merges: int):
        """
        Optimized BPE implementation LIMITED TO WORD BIGRAMS
        """
        # For BPE with word bigram limits, we'll focus on character-level merges
        # that don't violate the word bigram constraint
        
        # Convert to character sequences (sample for speed)
        sample_size = min(1000, len(texts))
        char_sequences = []
        for text in texts[:sample_size]:
            char_sequences.append(list(self._preprocess_text(text)))
        
        next_id = start_id
        base_threshold = max(2, len(texts) // 100000)
        
        # Limit BPE merges to maintain word bigram constraint
        for merge_round in range(min(max_merges, 200)):  # Reduced for word-level focus
            if len(self.token_to_id) >= self.vocab_size:
                break
            
            # Count pairs
            pair_counts = defaultdict(int)
            for char_seq in char_sequences:
                for j in range(len(char_seq) - 1):
                    pair = (char_seq[j], char_seq[j + 1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            # Find best pair that doesn't violate word bigram constraint
            best_pair = None
            best_score = 0
            
            for pair, count in pair_counts.items():
                if count < base_threshold:
                    continue
                
                merged = ''.join(pair)
                
                # Check if merged token violates word bigram constraint
                if not self._is_meaningful_substring(merged):
                    continue
                
                ner_score = self._calculate_ner_score(merged, count)
                
                if ner_score > best_score:
                    best_score = ner_score
                    best_pair = pair
            
            if not best_pair or best_score < base_threshold:
                break
            
            # Apply merge
            merged_token = ''.join(best_pair)
            new_sequences = []
            
            for char_seq in char_sequences:
                new_seq = []
                i = 0
                while i < len(char_seq):
                    if (i < len(char_seq) - 1 and 
                        char_seq[i] == best_pair[0] and 
                        char_seq[i + 1] == best_pair[1]):
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(char_seq[i])
                        i += 1
                new_sequences.append(new_seq)
            
            char_sequences = new_sequences
            self.bpe_merges.append(best_pair)
            
            # Add to vocabulary
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.id_to_token[next_id] = merged_token
                next_id += 1
            
            if len(self.token_to_id) >= self.vocab_size:
                break
            
            if merge_round % 50 == 0:
                print(f"BPE merge {merge_round}: {repr(merged_token)} (score: {best_score:.1f})")
        
        print(f"BPE word-bigram-limited merges completed: {len(self.bpe_merges)} merges")
    
    def _build_optimized_caches(self, texts: List[str]):
        """
        Build optimized caches for faster encoding
        """
        cache_count = 0
        for text in texts:
            if len(text) < 200 and cache_count < 1000:  # Limit cache size
                processed = self._preprocess_text(text)
                if processed not in self.text_cache:
                    tokens = self._segment_text_optimized(processed)
                    self.text_cache[processed] = tokens
                    cache_count += 1
        
        print(f"Cached {len(self.text_cache)} text segmentations")
    
    def _segment_text_optimized(self, text: str) -> List[str]:
        """
        Optimized text segmentation with NER-aware token selection
        LIMITED TO WORD BIGRAMS with proper space preservation
        """
        if not text:
            return []
        
        tokens = []
        i = 0
        
        while i < len(text):
            best_token = None
            best_length = 0
            best_score = 0
            
            # Special handling for spaces - always tokenize spaces properly
            if text[i] == ' ':
                # Count consecutive spaces
                space_count = 0
                temp_i = i
                while temp_i < len(text) and text[temp_i] == ' ':
                    space_count += 1
                    temp_i += 1
                
                # Add each space as a separate token to ensure proper decoding
                for _ in range(space_count):
                    tokens.append(' ')
                i = temp_i
                continue
            
            # For non-space characters, try to find the best token
            # First, try to find word bigrams by looking ahead
            words_from_here = []
            temp_i = i
            current_word = ""
            
            # Extract up to 2 words starting from position i
            while temp_i < len(text) and len(words_from_here) < 2:
                char = text[temp_i]
                if char == ' ':
                    if current_word:
                        words_from_here.append(current_word)
                        current_word = ""
                    break  # Stop at first space when collecting words
                else:
                    current_word += char
                    temp_i += 1
            
            if current_word:
                words_from_here.append(current_word)
            
            # Try word bigram if we can find the pattern "word space word"
            if len(words_from_here) >= 1:
                # Look for "word space word" pattern
                word1 = words_from_here[0]
                word1_end = i + len(word1)
                
                if (word1_end < len(text) and 
                    text[word1_end] == ' ' and 
                    word1_end + 1 < len(text)):
                    
                    # Find the second word after the space
                    word2_start = word1_end + 1
                    word2_end = word2_start
                    while word2_end < len(text) and text[word2_end] != ' ':
                        word2_end += 1
                    
                    if word2_end > word2_start:
                        word2 = text[word2_start:word2_end]
                        word_bigram = word1 + ' ' + word2
                        
                        if word_bigram in self.token_to_id:
                            best_token = word_bigram
                            best_length = len(word_bigram)
                            best_score = 100  # High score for word bigrams
            
            # Try single word if no bigram found
            if best_score < 50 and len(words_from_here) >= 1:
                single_word = words_from_here[0]
                if single_word in self.token_to_id:
                    score = 50 + len(single_word)  # Prefer longer words
                    if score > best_score:
                        best_token = single_word
                        best_length = len(single_word)
                        best_score = score
            
            # Fallback: try character-level tokens
            if best_score < 30:
                max_check = min(5, len(text) - i)  # Limit to prevent space issues
                
                for length in range(max_check, 0, -1):
                    candidate = text[i:i + length]
                    
                    # Skip candidates that contain spaces (we handle spaces separately)
                    if ' ' in candidate:
                        continue
                        
                    if candidate in self.token_to_id:
                        score = length
                        
                        # Bonus for title case
                        if candidate.strip() and candidate.strip().istitle():
                            score += 5
                        
                        if score > best_score:
                            best_token = candidate
                            best_length = length
                            best_score = score
            
            if best_token:
                tokens.append(best_token)
                i += best_length
            else:
                # Fallback to single character
                tokens.append(text[i])
                i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Optimized encoding with caching
        """
        if not text:
            return []
        
        preprocessed = self._preprocess_text(text)
        
        # Check cache first
        if preprocessed in self.text_cache:
            tokens = self.text_cache[preprocessed]
        else:
            tokens = self._segment_text_optimized(preprocessed)
            # Cache if reasonably sized
            if len(preprocessed) < 100 and len(self.text_cache) < 5000:
                self.text_cache[preprocessed] = tokens
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Fallback
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        token_ids.append(self.special_tokens["[UNK]"])
        
        return token_ids
    
    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Encode text and return character offsets for each token (for NER alignment)
        Enhanced to handle spaces properly
        """
        if not text:
            return [], []
        
        preprocessed = self._preprocess_text(text)
        tokens = self._segment_text_optimized(preprocessed)
        
        token_ids = []
        offsets = []
        current_pos = 0
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
                start_pos = current_pos
                end_pos = current_pos + len(token)
                offsets.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # Handle unknown tokens character by character
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        token_ids.append(self.special_tokens["[UNK]"])
                    
                    start_pos = current_pos
                    end_pos = current_pos + 1
                    offsets.append((start_pos, end_pos))
                    current_pos = end_pos
        
        return token_ids, offsets
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Simple decode - join tokens directly
        """
        if not token_ids:
            return ""
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("[UNK]")
        
        return ''.join(tokens)
    
    def get_token_analysis(self) -> Dict[str, any]:
        """
        Analyze the types of tokens created
        Enhanced for word-level analysis
        """
        analysis = {
            'total_tokens': len(self.token_to_id),
            'special_tokens': len(self.special_tokens),
            'single_char_tokens': 0,
            'single_word_tokens': 0,
            'word_bigram_tokens': 0,
            'cross_word_char_tokens': 0,
            'word_bigram_examples': [],
            'cross_word_examples': []
        }
        
        for token in self.token_to_id.keys():
            if token in self.special_tokens:
                continue
            
            words = token.strip().split()
            
            if len(token) == 1:
                analysis['single_char_tokens'] += 1
            elif len(words) == 1:
                analysis['single_word_tokens'] += 1
            elif len(words) == 2:
                analysis['word_bigram_tokens'] += 1
                if len(analysis['word_bigram_examples']) < 10:
                    analysis['word_bigram_examples'].append(repr(token))
            elif len(token) == 2 and ' ' in token:
                analysis['cross_word_char_tokens'] += 1
                if len(analysis['cross_word_examples']) < 10:
                    analysis['cross_word_examples'].append(repr(token))
        
        return analysis
    
    def test_cross_word_tokenization(self, text: str) -> Dict[str, any]:
        """
        Test method to see how text gets tokenized, especially word bigram patterns
        """
        encoded = self.encode(text)
        tokens = [self.id_to_token[tid] for tid in encoded if tid in self.id_to_token]
        decoded = self.decode(encoded)
        
        word_bigram_tokens = []
        cross_word_tokens = []
        
        for token in tokens:
            words = token.strip().split()
            if len(words) == 2:
                word_bigram_tokens.append(token)
            elif ' ' in token:
                cross_word_tokens.append(token)
        
        return {
            'original': text,
            'tokens': tokens,
            'word_bigram_tokens': word_bigram_tokens,
            'cross_word_tokens': cross_word_tokens,
            'encoded': encoded,
            'decoded': decoded,
            'matches_original': text == decoded
        }