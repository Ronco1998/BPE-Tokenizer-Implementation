# ner_bpe_tokenizer.py
"""
NER-aware BPE Tokenizer
=======================
A specialized Byte Pair Encoding tokenizer optimized for Named Entity Recognition tasks.

Key Features:
- Prioritizes word bigrams for better entity boundary preservation
- NER-aware scoring system that favors capitalized tokens
- Domain-specific preprocessing (Twitter, headlines, generic)
- Configurable bigram quota to ensure entity-friendly vocabulary
- Progress tracking with tqdm for training transparency

Author: [Your Name]
Date: July 2025
"""
from __future__ import annotations

import sys
import math
import heapq
import unicodedata
import os
import pickle
from collections import Counter
from html import unescape
from typing import Dict, List, Tuple
from tqdm import tqdm

from base_tokenizer import BaseTokenizer

# Configure UTF-8 output for Windows compatibility
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 or redirected stream

# Domain-specific preprocessing patterns
import re
_TW_USER = re.compile(r"@[A-Za-z0-9_]{1,15}") # Twitter usernames
_TW_URL = re.compile(r"https?://\S+") # Twitter URLs
_HASHTAG_RE = re.compile(r"#\w[\w\d_]*") # Twitter hashtags
_NEWS_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b") # News date
_PUNCT_PAD = re.compile(r"([,.;:!?()\"'])") # Punctuation padding
_EMOJI = re.compile(
    r"[\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|"
    r"[\u2600-\u26FF]|[\u2700-\u27BF]"
) # Emoji patterns

# Unicode normalization table
UNICODE_PUNCT_TABLE = str.maketrans({
    "“": '"',    # left double quotation mark
    "”": '"',    # right double quotation mark
    "‘": "'",    # left single quotation mark
    "’": "'",    # right single quotation mark
    "—": "-",    # em dash
    "–": "-",    # en dash
    "…": "...",  # ellipsis
})


class NERBPETokenizer(BaseTokenizer):
    """
    BPE tokenizer specialized for Named Entity Recognition tasks.
    
    This tokenizer implements a NER-aware BPE algorithm that:
    1. Prioritizes word bigrams to preserve entity boundaries
    2. Uses NER-specific scoring to favor capitalized tokens
    3. Applies domain-specific preprocessing
    4. Maintains bigram constraints (max 2 words per token)
    """

    # Special tokens
    UNK_TOKEN = "[UNK]"
    SPACE_TOKEN = " "
    END_MARK = "</w>"
    
    # Compiled regex for problematic characters (better performance)
    PROBLEMATIC_CHARS_RE = re.compile(r'[ÐÑ\x80⁄μ°½¼¾©³´¿¡£ÂÃï]')
    
    # Domain-specific preprocessing tokens
    PREPROCESSING_TOKENS = {
        'twitter': ["<URL>", "<USER>", "<HASHTAG>", "<EMOJI>"],
        'headline': ["<DATE>", "<EMOJI>"],
        'generic': ["<EMOJI>"]
    }

    def __init__(self, vocab_size: int = 4000, *, domain: str = "unknown") -> None:
        """
        Initialize the NER-aware BPE tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            domain: Text domain for preprocessing ('twitter', 'headline', or 'unknown')
        """
        # Initialize without calling super() to avoid unwanted special tokens
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.domain = domain.lower()
        print(f"Initializing NERBPETokenizer for domain: {self.domain}")
        self.vocab_size = vocab_size
        self._bpe_merge_ranks: Dict[Tuple[str, str], int] = {}
        self._token_frequencies: Dict[str, int] = {}  # Track token frequencies during training

        # Add only the special tokens we need
        for tok in (self.UNK_TOKEN, self.SPACE_TOKEN):
            self._add_token(tok)
        
        # Add domain-specific preprocessing tokens
        self._add_preprocessing_tokens()

    def _add_preprocessing_tokens(self) -> None:
        """Add domain-specific preprocessing tokens to vocabulary."""
        # Add tokens for current domain
        if self.domain in self.PREPROCESSING_TOKENS:
            for token in self.PREPROCESSING_TOKENS[self.domain]:
                self._add_token(token)
                self._token_frequencies[token] = 0  # Initialize frequency tracking
        
        else:
            # Always add generic tokens as fallback
            for token in self.PREPROCESSING_TOKENS['generic']:
                if token not in self.token_to_id:
                    self._add_token(token)
                    self._token_frequencies[token] = 0  # Initialize frequency tracking
        
        print(f"Added preprocessing tokens for domain '{self.domain}': "
              f"{self.PREPROCESSING_TOKENS.get(self.domain, self.PREPROCESSING_TOKENS['generic'])}")

    def train(self, texts: List[str], *, char_limit: int | None = None, bigram_quota: float = 0.3, 
              min_word_score: float = 15.0, min_bigram_score: float = 20.0, min_frequency: int = 2) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            char_limit: Maximum number of most frequent characters to add first
            bigram_quota: Fraction of remaining slots reserved for bigrams (0.0-1.0)
            min_word_score: Minimum NER score for word tokens
            min_bigram_score: Minimum NER score for bigram tokens
            min_frequency: Minimum frequency threshold for words and bigrams (default: 2)

        Training Process:
        1. Compute character, word, and bigram frequencies
        2. Add all unique characters to vocabulary
        3. Score and filter words/bigrams by NER relevance
        4. Add top-scoring tokens respecting bigram quota
        5. Run BPE merges to fill remaining vocabulary slots
        """
        # Reduce aggressive parameter adjustments that hurt performance
        if char_limit is None:
            if self.vocab_size <= 1000:
                char_limit = min(60, self.vocab_size // 8)  # Less aggressive reduction
            elif self.vocab_size <= 3000:
                char_limit = min(120, self.vocab_size // 10)  # Less aggressive reduction
            else:
                char_limit = 200
        
        # Less aggressive parameter changes for small vocabularies
        if self.vocab_size <= 3000:
            bigram_quota = max(0.4, bigram_quota)  # Reduced from 0.6
            min_word_score = max(25.0, min_word_score + 5.0)  # Less aggressive increase
            min_bigram_score = max(15.0, min_bigram_score - 5.0)  # Actually lower threshold
            min_frequency = max(2, min_frequency)  # Keep at 2, don't increase

        total_steps = 7  # Updated to include analysis
        pbar = tqdm(total=total_steps, desc="Training tokenizer", unit="step")
        
        # Step 1: Compute frequencies
        pbar.set_description("Computing frequencies")
        char_freq, word_freq, bigram_freq = self._get_frequencies(texts)
        pbar.update(1)
        
        # Step 2: Add all characters to vocabulary
        pbar.set_description("Adding characters")
        self._add_characters(texts, char_freq, char_limit)
        pbar.update(1)
        
        # Step 3: Score entities with NER-aware scoring
        pbar.set_description("Scoring entities")
        entity_scores_heap = self._score_entities(word_freq, bigram_freq, min_word_score, min_bigram_score, min_frequency)
        pbar.update(1)
        
        # Step 4: Add top entities respecting bigram quota
        pbar.set_description("Adding top entities")
        added_bigrams, added_singles = self._add_top_entities(entity_scores_heap, bigram_quota)
        pbar.update(1)
        
        # Step 5: Fill remaining slots with BPE merges
        if self._remaining_slots > 0:
            pbar.set_description("Running BPE merges")
            self._run_bpe_merges(texts)
        pbar.update(1)
        
        # Step 6: Final diagnostics
        pbar.set_description("Final diagnostics")
        unk_share = self._estimate_unk_share(texts)
        pbar.update(1)
        
        # Step 7: Token analysis
        pbar.set_description("Analyzing tokens")
        self._analyze_tokens(texts)
        pbar.update(1)
        pbar.close()
        
        # Print training summary
        self._print_training_summary(added_bigrams, added_singles, unk_share)

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs using NER-aware longest match.

        The encoding prioritizes:
        1. Word bigrams (e.g., "New York")
        2. Single words
        3. Character-level tokens
        4. UNK token as fallback

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        txt = self._preprocess(text)
        tokens = []
        i = 0
        
        while i < len(txt):
            # Handle spaces directly
            if txt[i] == " ":
                tokens.append(self._token_to_id[self.SPACE_TOKEN])
                i += 1
                continue
            
            # Find best match using NER-aware strategy
            best_match, best_length = self._find_best_match(txt, i)
            
            if best_match:
                tokens.append(self._token_to_id[best_match])
                i += best_length
            else:
                # Fallback to character or UNK
                char = txt[i]
                token_id = self._token_to_id.get(char, self._token_to_id[self.UNK_TOKEN])
                tokens.append(token_id)
                i += 1
        
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return "".join(self._id_to_token.get(i, self.UNK_TOKEN) for i in token_ids)


    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.token_to_id)

    # -------------------------------------------------------------------------
    # Private Training Methods
    # -------------------------------------------------------------------------

    def _get_frequencies(self, texts: List[str]) -> Tuple[Counter[str], Counter[str], Counter[str]]:
        """Compute character, word, and bigram frequencies from texts."""
        char_freq = Counter()
        word_freq = Counter()
        bigram_freq = Counter()
        
        for line in tqdm(texts, desc="Computing frequencies", leave=False):
            processed = self._preprocess(line)
            
            # Character frequencies (from first 100 lines for efficiency)
            if len(char_freq) == 0 or line in texts[:100]:
                char_freq.update(processed)
            
            # Word and bigram frequencies
            words = processed.split()
            word_freq.update(words)
            
            # Bigram frequencies
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_freq[bigram] += 1
        
        return char_freq, word_freq, bigram_freq

    def _add_characters(self, texts: List[str], char_freq: Counter, char_limit: int) -> None:
        """Add all unique characters to vocabulary, prioritizing frequent ones."""
        # Discover all unique characters
        all_chars = set()
        for line in tqdm(texts[:1000], desc="Discovering characters", leave=False):
            all_chars.update(self._preprocess(line))
        
        all_chars.discard(" ")  # Space already in vocab
        char_freq.pop(" ", None)
        
        # Add frequent characters first
        added_chars = set()
        for ch, freq in char_freq.most_common(min(char_limit, len(all_chars))):
            if self._remaining_slots > 0:
                self._add_token(ch)
                self._token_frequencies[ch] = freq  # Store character frequency
                added_chars.add(ch)
        
        # Add remaining characters in sorted order for deterministic behavior
        for ch in sorted(all_chars - added_chars):
            if self._remaining_slots > 0:
                self._add_token(ch)
                self._token_frequencies[ch] = 0  # No frequency data for remaining chars
            else:
                break

    def _score_entities(self, word_freq: Counter, bigram_freq: Counter, 
                       min_word_score: float, min_bigram_score: float, min_frequency: int = 2) -> List:
        """Score words and bigrams using NER-aware scoring, returning a heap."""
        entity_scores_heap = []
        
        # Score words with minimum frequency requirement
        for word, freq in tqdm(word_freq.items(), desc="Scoring words", leave=False):
            # Only consider words with frequency >= min_frequency and proper characters
            if freq >= min_frequency and len(word) > 1 and not self._has_problematic_chars(word):
                score = self._calc_ner_score((word, ""), freq)
                if score >= min_word_score:
                    heapq.heappush(entity_scores_heap, (-score, word, freq))
        
        # Score bigrams with minimum frequency requirement - be more lenient for bigrams
        bigram_min_freq = max(1, min_frequency - 1)  # Allow bigrams with freq 1 if min_frequency is 2
        for bigram, freq in tqdm(bigram_freq.items(), desc="Scoring bigrams", leave=False):
            # Only consider bigrams with frequency >= bigram_min_freq and proper characters
            if freq >= bigram_min_freq and not self._has_problematic_chars(bigram):
                w1, w2 = bigram.split(" ", 1)
                score = self._calc_ner_score((w1, " " + w2), freq)
                if score >= min_bigram_score:
                    heapq.heappush(entity_scores_heap, (-score, bigram, freq))
        
        return entity_scores_heap

    def _has_problematic_chars(self, text: str) -> bool:
        """Check if text contains problematic Unicode or corrupted characters."""
        # Use compiled regex for better performance
        if self.PROBLEMATIC_CHARS_RE.search(text):
            return True
        
        # Check for too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .-') / len(text)
        if special_char_ratio > 0.4:  # More than 40% special chars
            return True
            
        return False

    def _add_top_entities(self, entity_scores_heap: List, bigram_quota: float) -> Tuple[int, int]:
        """Add top-scoring entities to vocabulary respecting bigram quota."""
        rest_slots = self._remaining_slots
        bg_slots = int(rest_slots * bigram_quota)
        
        # Reserve significant portion for BPE merges
        bpe_reserve = min(rest_slots // 4, 300)  # Reduced from 1/3 and 500
        available_slots = rest_slots - bpe_reserve
        bg_slots = int(available_slots * bigram_quota)
        word_slots = max(0, available_slots - bg_slots)
        
        print(f"Available slots: {rest_slots}, BPE reserved: {bpe_reserve}, Bigram slots: {bg_slots}, Word slots: {word_slots}")
        
        # Extract candidates - heap structure is (-score, token, freq)
        all_candidates = []
        while entity_scores_heap and len(all_candidates) < available_slots * 2:  # Extract more candidates
            neg_score, token, freq = heapq.heappop(entity_scores_heap)
            # Stricter frequency requirement for candidates
            if freq >= 2:  # Require at least frequency 2
                all_candidates.append((token, -neg_score, freq))
        
        # Separate bigrams and singles
        bigrams = [(s, sc, f) for s, sc, f in all_candidates if self._is_bigram(s)]
        singles = [(s, sc, f) for s, sc, f in all_candidates if not self._is_bigram(s)]
        
        # Sort by score (higher score = better)
        bigrams.sort(key=lambda x: x[1], reverse=True)
        singles.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(bigrams)} bigram candidates and {len(singles)} word candidates")
        
        # Add top bigrams (already sorted by score)
        added_bigrams = 0
        for s, score, freq in bigrams[:bg_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            self._add_token(s)
            self._token_frequencies[s] = freq
            added_bigrams += 1
        
        # If we didn't use all bigram slots, convert them to word slots
        unused_bigram_slots = bg_slots - added_bigrams
        word_slots += unused_bigram_slots
        
        # Add top singles (already sorted by score) - be more selective
        added_singles = 0
        for s, score, freq in singles[:word_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            # Reduce the score threshold
            if score >= 20.0:  # Reduced from 30.0
                self._add_token(s)
                self._token_frequencies[s] = freq
                added_singles += 1
        
        print(f"Added {added_bigrams} bigrams and {added_singles} words")
        return added_bigrams, added_singles

    def _run_bpe_merges(self, texts: List[str]) -> None:
        """Run BPE merges to fill remaining vocabulary slots."""
        # Build word frequency for BPE with better sampling
        word_freq = Counter()
        
        # Use larger sample for BPE to get better statistics
        if self.vocab_size <= 1000:
            sample_size = min(50000, len(texts))  # Increased sample
        elif self.vocab_size <= 3000:
            sample_size = min(200000, len(texts))  # Increased sample
        else:
            sample_size = min(500000, len(texts))
        
        print(f"Building BPE word frequencies from {sample_size} texts...")
        
        for text in tqdm(texts[:sample_size], desc="Building word frequencies", leave=False):
            tokens = self._preprocess(text).split()
            for w in tokens:
                # Process all words, not just length > 1
                if w:  # Just check it's not empty
                    word_freq[tuple(w) + (self.END_MARK,)] += 1
        
        print(f"Created {len(word_freq)} unique words for BPE processing")
        
        # More aggressive BPE parameters
        iteration = 0
        max_iterations = self._remaining_slots * 3  # Allow more iterations
        min_pair_frequency = 1  # Keep threshold low
        
        print(f"Starting BPE with {self._remaining_slots} remaining slots, max iterations: {max_iterations}")
        
        with tqdm(total=min(max_iterations, self._remaining_slots), desc="BPE merges", leave=False) as pbar:
            while self._remaining_slots > 0 and iteration < max_iterations:
                # Find best pair to merge
                best_pair = self._find_best_bpe_pair(word_freq)
                if not best_pair:
                    print(f"No valid pairs found at iteration {iteration}")
                    break
                
                # Get frequency for the best pair
                pair_freq = Counter()
                for w, f in word_freq.items():
                    for p in self._bpe_pairs(w):
                        pair_freq[p] += f
                best_freq = pair_freq.get(best_pair, 0)
                
                if best_freq < min_pair_frequency:
                    print(f"Stopping BPE: best frequency {best_freq} < threshold {min_pair_frequency}")
                    break
                
                # Apply merge and track frequency
                merged_token = "".join(best_pair)
                self._perform_merge(best_pair, word_freq)
                self._token_frequencies[merged_token] = best_freq
                self._bpe_merge_ranks[best_pair] = iteration
                
                iteration += 1
                pbar.update(1)
                
                if iteration % 50 == 0:
                    pbar.set_postfix({"merge": f"'{best_pair[0]}'+''{best_pair[1]}'", "freq": best_freq, "remaining": self._remaining_slots})
        
        print(f"BPE completed after {iteration} iterations, {self._remaining_slots} slots remaining")

    def _find_best_bpe_pair(self, word_freq: Counter) -> Tuple[str, str] | None:
        """Find the best character pair to merge in BPE, prioritizing frequency."""
        # Count all adjacent pairs
        pair_freq = Counter()
        for w, f in word_freq.items():
            for p in self._bpe_pairs(w):
                pair_freq[p] += f
        
        if not pair_freq:
            return None
        
        # Find best pair prioritizing frequency first, then useful merges
        best_pair = None
        best_score = -1
        
        # Sort by frequency first to prioritize high-frequency pairs
        for pair, freq in pair_freq.most_common():
            merged_token = "".join(pair)
            
            # Check bigram constraint
            if self._violates_bigram_constraint(merged_token):
                continue
            
            # Skip if merged token already exists
            if merged_token in self._token_to_id:
                continue
            
            # Heavily prioritize frequency for BPE
            frequency_score = freq * 20  # Increased weight for frequency
            
            # Bonus for useful character combinations
            char_bonus = 0
            if len(merged_token) == 2:
                # Bonus for common character combinations
                if merged_token.lower() in ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ou', 'ea', 'ti', 'to', 'it', 'st', 'io', 'le', 'is', 'ar', 'as', 'de', 'rt', 'se']:
                    char_bonus += 50
                elif merged_token.isalpha():
                    char_bonus += 10
            
            total_score = frequency_score + char_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_pair = pair
        
        return best_pair

    # -------------------------------------------------------------------------
    # Private Encoding Methods
    # -------------------------------------------------------------------------

    def _find_best_match(self, text: str, start: int) -> Tuple[str | None, int]:
        """Find the best token match starting at position start."""
        best_match = None
        best_length = 0
        
        # First priority: Check for preprocessing tokens (longest match first)
        preprocessing_tokens = []
        if self.domain in self.PREPROCESSING_TOKENS:
            preprocessing_tokens.extend(self.PREPROCESSING_TOKENS[self.domain])
        preprocessing_tokens.extend(self.PREPROCESSING_TOKENS['generic'])
        
        # Sort by length (longest first) for proper matching
        preprocessing_tokens.sort(key=len, reverse=True)
        
        for token in preprocessing_tokens:
            if (start + len(token) <= len(text) and 
                text[start:start + len(token)] == token):
                best_match = token
                best_length = len(token)
                break
        
        # Second priority: Try word-level matches (prefer bigrams)
        if not best_match and text[start] != " ":
            words = self._extract_words_from_position(text, start)
            
            # Try bigram first
            if len(words) >= 2:
                bigram = f"{words[0]} {words[1]}"
                if bigram in self._token_to_id:
                    best_match = bigram
                    best_length = len(words[0]) + 1 + len(words[1])
            
            # Try single word if no bigram
            if not best_match and len(words) >= 1:
                if words[0] in self._token_to_id:
                    best_match = words[0]
                    best_length = len(words[0])
        
        # Third priority: Fallback to character-level longest match
        if not best_match:
            max_check = min(30, len(text) - start)
            for length in range(max_check, 0, -1):
                candidate = text[start:start + length]
                if candidate in self._token_to_id:
                    best_match = candidate
                    best_length = length
                    break
        
        return best_match, best_length

    def _extract_words_from_position(self, text: str, start: int) -> List[str]:
        """Extract up to 2 words starting from position start."""
        words = []
        i = start
        
        # Extract first word
        if i < len(text) and text[i] != " ":
            word_start = i
            while i < len(text) and text[i] != " ":
                i += 1
            words.append(text[word_start:i])
        
        # Skip spaces
        while i < len(text) and text[i] == " ":
            i += 1
        
        # Extract second word if available
        if i < len(text) and text[i] != " ":
            word_start = i
            while i < len(text) and text[i] != " ":
                i += 1
            words.append(text[word_start:i])
        
        return words

    # -------------------------------------------------------------------------
    # Scoring and Feature Extraction
    # -------------------------------------------------------------------------

    def _calc_ner_score(self, pair: Tuple[str, str], frequency: int = 1) -> float:
        """Calculate NER relevance score for a token pair."""
        joined = "".join(pair)
        base_score = math.log(frequency + 1) * 1.5  # Reduce from 2 to balance with NER features
        features = self._get_ner_indicators(joined)
        
        # NER-specific bonuses - rebalance to be less aggressive
        ner_score = 0.0
        
        # Character tokens
        if features['length'] == 1:
            ner_score += 2
            if joined.isupper():
                ner_score += 3
            elif joined.isdigit():
                ner_score += 1
            elif joined == ' ':
                ner_score += 10
            # Less harsh penalty for punctuation
            elif not joined.isalnum():
                ner_score -= 2  # Reduced from -5
        
        # Word tokens
        elif features['num_words'] == 1:
            ner_score += 10
            if features['has_uppercase']:
                ner_score += 15  # Reduced from 20
            if features['is_all_caps']:
                ner_score += 8   # Reduced from 10
            if features['title_case_ratio']:
                ner_score += 35  # Reduced from 50
            if features['has_mixed_case']:
                ner_score += 12  # Reduced from 15
            if features['has_digits']:
                ner_score += 3
            if features['has_special_chars']:
                ner_score += 6   # Reduced from 8
        
        # Bigram tokens - maintain high priority but be less extreme
        elif features['num_words'] == 2:
            ner_score += 30  # Reduced from 40
            if features['title_case_ratio'] > 0:
                ner_score += features['title_case_ratio'] * 35  # Reduced from 50
            if features['has_digits']:
                ner_score += 3
        
        # Cross-word character pairs
        elif features['has_space'] and features['length'] == 2:
            ner_score += 25  # Reduced from 30
            if pair[0] == ' ' and len(pair[1]) > 0 and pair[1][0].isupper():
                ner_score += 30  # Reduced from 40
        
        # Additional bonuses - reduce to be less aggressive
        if features['has_space']:
            ner_score += 12  # Reduced from 15
        if features['has_mixed_case']:
            ner_score += 15  # Reduced from 20
        if features['has_digits']:
            ner_score += 3
        if features['has_special_chars']:
            ner_score += 8   # Reduced from 10
        
        # Less harsh punctuation penalty
        if features['length'] > 1:
            punct_ratio = sum(1 for c in joined if not c.isalnum() and c != ' ') / features['length']
            if punct_ratio > 0.5:
                ner_score -= 8   # Reduced from -15
            elif punct_ratio > 0.3:
                ner_score -= 4   # Reduced from -8
        
        return base_score + ner_score

    def _get_ner_indicators(self, token: str) -> Dict[str, float]:
        """Extract NER-relevant features from a token."""
        features = {
            'length': len(token),
            'num_words': len(token.strip().split()),
            'has_space': 1.0 if ' ' in token else 0.0,
            'has_digits': 1.0 if any(c.isdigit() for c in token) else 0.0,
            'has_uppercase': 1.0 if any(c.isupper() for c in token) else 0.0,
            'has_lowercase': 1.0 if any(c.islower() for c in token) else 0.0,
            'has_mixed_case': 0.0,
            'has_special_chars': 1.0 if any(not c.isalnum() and c != ' ' for c in token) else 0.0,
            'title_case_ratio': 0.0,
            'is_word_pair': False,
            'is_all_caps': False,
        }
        
        # Mixed case detection
        if features['has_uppercase'] > 0 and features['has_lowercase'] > 0:
            features['has_mixed_case'] = 1.0
        
        # All caps detection (alphabetic characters only)
        alpha_chars = [c for c in token if c.isalpha()]
        if alpha_chars:
            features['is_all_caps'] = all(c.isupper() for c in alpha_chars)
        
        # Word-level features
        words = token.strip().split()
        
        if len(words) == 1 and words[0]:
            features['title_case_ratio'] = 1.0 if words[0].istitle() else 0.0
        elif len(words) == 2:
            features['is_word_pair'] = True
            title_case_count = sum(1 for w in words if w and w.istitle())
            features['title_case_ratio'] = title_case_count / 2.0
        
        return features

    # -------------------------------------------------------------------------
    # Preprocessing Methods
    # -------------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """Apply domain-specific preprocessing and track token usage."""
        if self.domain == "twitter":
            return self._pre_twitter(text)
        elif self.domain in {"headline", "headlines"}:
            return self._pre_headline(text)
        else:
            return self._pre_generic(text)

    def _pre_twitter(self, text: str) -> str:
        """Preprocess Twitter text and track preprocessing token usage."""
        # text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        
        # Track URL replacements
        url_count = len(_TW_URL.findall(text))
        if url_count > 0:
            self._token_frequencies["<URL>"] = self._token_frequencies.get("<URL>", 0) + url_count
        text = _TW_URL.sub("<URL>", text)
        
        # Track USER replacements
        user_count = len(_TW_USER.findall(text))
        if user_count > 0:
            self._token_frequencies["<USER>"] = self._token_frequencies.get("<USER>", 0) + user_count
        text = _TW_USER.sub("<USER>", text)
        
        # Track HASHTAG replacements
        hashtag_count = len(_HASHTAG_RE.findall(text))
        if hashtag_count > 0:
            self._token_frequencies["<HASHTAG>"] = self._token_frequencies.get("<HASHTAG>", 0) + hashtag_count
        text = _HASHTAG_RE.sub("<HASHTAG>", text)
        
        # Track EMOJI replacements
        emoji_count = len(_EMOJI.findall(text))
        if emoji_count > 0:
            self._token_frequencies["<EMOJI>"] = self._token_frequencies.get("<EMOJI>", 0) + emoji_count
        text = _EMOJI.sub("<EMOJI>", text)
        
        # text = text.translate(UNICODE_PUNCT_TABLE)
        text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_headline(self, text: str) -> str:
        """Preprocess headline text and track preprocessing token usage."""
        # text = unicodedata.normalize("NFKC", text)
        
        # Track DATE replacements
        date_count = len(_NEWS_DATE.findall(text))
        if date_count > 0:
            self._token_frequencies["<DATE>"] = self._token_frequencies.get("<DATE>", 0) + date_count
        text = _NEWS_DATE.sub("<DATE>", text)
        
        # Track EMOJI replacements
        emoji_count = len(_EMOJI.findall(text))
        if emoji_count > 0:
            self._token_frequencies["<EMOJI>"] = self._token_frequencies.get("<EMOJI>", 0) + emoji_count
        text = _EMOJI.sub("<EMOJI>", text)
        
        # text = text.translate(UNICODE_PUNCT_TABLE)
        text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_generic(self, text: str) -> str:
        """Preprocess generic text and track preprocessing token usage."""
        # Restore proper preprocessing that was disabled
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        
        # Track EMOJI replacements
        emoji_count = len(_EMOJI.findall(text))
        if emoji_count > 0:
            self._token_frequencies["<EMOJI>"] = self._token_frequencies.get("<EMOJI>", 0) + emoji_count
        text = _EMOJI.sub("<EMOJI>", text)
        
        # Restore proper punctuation handling
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _violates_bigram_constraint(self, token: str) -> bool:
        """Check if token violates bigram constraint (max 2 words)."""
        return len(token.strip().split()) > 2

    def _print_training_summary(self, added_bigrams: int, added_singles: int, unk_share: float) -> None:
        """Print training summary statistics."""
        print(f"Training complete!")
        print(f"Vocabulary size: {len(self._token_to_id)}")
        print(f"Bigrams added: {added_bigrams}")
        print(f"Words added: {added_singles}")
        print(f"Total bigrams: {sum(self._is_bigram(t) for t in self._token_to_id)}")
        if unk_share > 0.02:
            print(f"WARNING: UNK ratio is {unk_share:.2%} (> 2%)")

    def _estimate_unk_share(self, corpus: List[str]) -> float:
        """Estimate UNK token share in corpus."""
        total, unk = 0, 0
        for line in corpus[1000:2000]:  # Sample for efficiency
            ids = self.encode(line)
            total += len(ids)
            unk += sum(1 for i in ids if i == self.token_to_id[self.UNK_TOKEN])
        return unk / total if total else 0.0

    # -------------------------------------------------------------------------
    # Properties and Helper Methods
    # -------------------------------------------------------------------------

    @property
    def _token_to_id(self):
        """Access inherited token_to_id dictionary."""
        return self.token_to_id

    @property
    def _id_to_token(self):
        """Access inherited id_to_token dictionary."""
        return self.id_to_token

    @property
    def _remaining_slots(self) -> int:
        """Calculate remaining vocabulary slots."""
        return max(0, self.vocab_size - len(self._token_to_id))

    @staticmethod
    def _bpe_pairs(word: Tuple[str, ...]):
        """Get all adjacent character pairs in a word."""
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    @staticmethod
    def _is_bigram(token: str) -> bool:
        """Check if token is a valid bigram (exactly 2 words)."""
        return token.count(" ") == 1 and not token.startswith(" ") and not token.endswith(" ")

    def _add_token(self, token: str) -> None:
        """Add token to vocabulary if not already present and space available."""
        if token in self.token_to_id or len(self.token_to_id) >= self.vocab_size:
            return
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    def _perform_merge(self, pair: Tuple[str, str], word_freq: Counter) -> None:
        """Perform BPE merge operation on word frequency counter."""
        merged = "".join(pair)
        self._add_token(merged)
        new_freq = Counter()
        
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            word_list = list(word)
            
            while i < len(word_list):
                if (i < len(word_list) - 1 and 
                    word_list[i] == pair[0] and 
                    word_list[i + 1] == pair[1]):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
            
            new_freq[tuple(new_word)] += freq
        
        word_freq.clear()
        word_freq.update(new_freq)

    def _analyze_tokens(self, texts: List[str]) -> None:
        """
        Analyze trained tokens from vocabulary and output detailed statistics to a text file.
        
        Creates a comprehensive report including:
        - Token type distribution from vocabulary
        - Top/bottom ranked bigrams created during training with their frequencies
        - Detailed token list with rankings
        """
        from datetime import datetime
        
        # Categorize tokens from vocabulary (not from encoding)
        token_categories = {
            'special': [],
            'preprocessing': [],
            'characters': [],
            'words': [],
            'bigrams': [],
            'subwords': [],
            'merged_chars': []
        }
        
        # Analyze all tokens in vocabulary
        for token in self.token_to_id.keys():
            if token in [self.UNK_TOKEN, self.SPACE_TOKEN, "[PAD]", "[BOS]", "[EOS]"]:
                token_categories['special'].append(token)
            elif any(token in tokens for tokens in self.PREPROCESSING_TOKENS.values()):
                token_categories['preprocessing'].append(token)
            elif len(token) == 1:
                token_categories['characters'].append(token)
            elif self._is_bigram(token):
                token_categories['bigrams'].append(token)
            elif len(token.strip().split()) == 1 and len(token) > 1:
                if ' ' in token or any(not c.isalnum() and c != ' ' for c in token):
                    token_categories['subwords'].append(token)
                else:
                    token_categories['words'].append(token)
            else:
                token_categories['merged_chars'].append(token)
        
        # Prepare bigram data with frequencies and scores - SORT BY FREQUENCY
        bigram_data = []
        for bigram in token_categories['bigrams']:
            # Get frequency from training (default to 0 if not found)
            freq = self._token_frequencies.get(bigram, 0)
            # Calculate score for reference but don't use for sorting
            score = self._calc_ner_score((bigram.split()[0], " " + bigram.split()[1]), max(1, freq))
            bigram_data.append((bigram, freq, score))
        
        # Sort bigrams by FREQUENCY (highest first), not by score
        bigram_data.sort(key=lambda x: x[1], reverse=True)
        
        # Generate analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"token_analysis_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_token_analysis_report(f, token_categories, bigram_data, texts)
        
        print(f"Token analysis saved to: {filename}")
        
        # Also print top/bottom bigrams to console
        if bigram_data:
            print("\nTop 5 Ranked Bigrams (by frequency):")
            for i, (bigram, freq, score) in enumerate(bigram_data[:5], 1):
                print(f"  {i}. \"{bigram}\" (freq: {freq}, score: {score:.2f})")
            
            print("\nBottom 5 Ranked Bigrams (by frequency):")
            for i, (bigram, freq, score) in enumerate(bigram_data[-5:], len(bigram_data)-4):
                print(f"  {i}. \"{bigram}\" (freq: {freq}, score: {score:.2f})")

    def _write_token_analysis_report(self, f, token_categories: Dict, bigram_data: List, texts: List[str]) -> None:
        """Write comprehensive token analysis report to file."""
        from datetime import datetime
        
        f.write("=" * 80 + "\n")
        f.write("NER-AWARE BPE TOKENIZER - TOKEN ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Domain: {self.domain}\n")
        f.write(f"Vocabulary Size: {len(self.token_to_id)}\n")
        f.write(f"Training Corpus Size: {len(texts)} texts\n")
        f.write("\n")
        
        # Domain-specific preprocessing tokens info
        f.write("PREPROCESSING TOKENS\n")
        f.write("-" * 30 + "\n")
        if self.domain in self.PREPROCESSING_TOKENS:
            f.write(f"Domain '{self.domain}' tokens: {self.PREPROCESSING_TOKENS[self.domain]}\n")
        f.write(f"Generic tokens: {self.PREPROCESSING_TOKENS['generic']}\n")
        f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        total_tokens = len(self.token_to_id)
        for category, tokens in token_categories.items():
            count = len(tokens)
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            f.write(f"{category.capitalize():15}: {count:5} tokens ({percentage:5.1f}%)\n")
        
        f.write(f"{'Total':15}: {total_tokens:5} tokens\n")
        f.write("\n")
        
        # Bigram ranking analysis - NOW SORTED BY FREQUENCY
        f.write("BIGRAM RANKING ANALYSIS (Ranked by Frequency)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total bigrams created: {len(bigram_data)}\n")
        
        if bigram_data:
            f.write("\n")
            f.write("TOP 10 RANKED BIGRAMS (by frequency):\n")
            f.write(f"{'Rank':<5} {'Bigram':<25} {'Frequency':<10} {'NER Score':<12}\n")
            f.write("-" * 55 + "\n")
            for rank, (bigram, freq, score) in enumerate(bigram_data[:10], 1):
                f.write(f"{rank:<5} {repr(bigram):<25} {freq:<10} {score:<12.2f}\n")
            f.write("\n")
            
            f.write("BOTTOM 10 RANKED BIGRAMS (by frequency):\n")
            f.write(f"{'Rank':<5} {'Bigram':<25} {'Frequency':<10} {'NER Score':<12}\n")
            f.write("-" * 55 + "\n")
            for rank, (bigram, freq, score) in enumerate(bigram_data[-10:], len(bigram_data)-9):
                f.write(f"{rank:<5} {repr(bigram):<25} {freq:<10} {score:<12.2f}\n")
            f.write("\n")
        else:
            f.write("No bigrams were created during training.\n")
            f.write("This may be due to:\n")
            f.write("- Small vocabulary size limiting bigram allocation\n")
            f.write("- High minimum frequency thresholds\n")
            f.write("- Insufficient bigram quota allocation\n")
            f.write("\n")
        
        # Detailed category analysis
        for category, tokens in token_categories.items():
            if not tokens:
                continue
                
            f.write(f"{category.upper()} TOKENS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Count: {len(tokens)}\n")
            f.write(f"Tokens (up to 20):\n")
            
            for i, token in enumerate(tokens[:20], 1):
                freq = self._token_frequencies.get(token, 0)
                token_repr = repr(token) if len(token) <= 25 else repr(token[:22] + "...")
                f.write(f"  {i:2}. {token_repr:<30} (freq: {freq})\n")
            
            f.write("\n")
        
        # Complete token list (fixed duplicate output)
        f.write("COMPLETE TOKEN VOCABULARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'ID':<6} {'Token':<35} {'Frequency':<10} {'Type':<12}\n")
        f.write("-" * 70 + "\n")
        
        # Sort by token ID for deterministic output
        all_tokens_by_id = sorted(self.token_to_id.items(), key=lambda x: x[1])
        
        for token, token_id in all_tokens_by_id:
            token_type = self._get_token_type(token)
            freq = self._token_frequencies.get(token, 0)
            token_repr = repr(token) if len(token) <= 30 else repr(token[:27] + "...")
            f.write(f"{token_id:<6} {token_repr:<35} {freq:<10} {token_type:<12}\n")

    def _get_token_type(self, token: str) -> str:
        """Determine the type of a token for analysis."""
        if token in [self.UNK_TOKEN, self.SPACE_TOKEN, "[PAD]", "[BOS]", "[EOS]"]:
            return "special"
        # Check if it's a preprocessing token
        elif any(token in tokens for tokens in self.PREPROCESSING_TOKENS.values()):
            return "preprocessing"
        elif len(token) == 1:
            return "character"
        elif self._is_bigram(token):
            return "bigram"
        elif len(token.strip().split()) == 1:
            if any(not c.isalnum() and c != ' ' for c in token):
                return "subword"
            else:
                return "word"
        else:
            return "merged"