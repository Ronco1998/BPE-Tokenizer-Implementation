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
from collections import Counter
from html import unescape
from typing import Dict, List, Tuple
from tqdm import tqdm

from base_tokenizer import BaseTokenizer

# Configure UTF-8 output for Windows compatibility
# try:
#     sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# except AttributeError:
#     pass  # Python < 3.7 or redirected stream

# Domain-specific preprocessing patterns
import re
_TW_USER = re.compile(r"@[A-Za-z0-9_]{1,15}")
_TW_URL = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w[\w\d_]*")
_NEWS_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PUNCT_PAD = re.compile(r"([,.;:!?()\"'])")
_EMOJI = re.compile(
    r"[\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|"
    r"[\u2600-\u26FF]|[\u2700-\u27BF]"
)

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

    def __init__(self, vocab_size: int = 4000, *, domain: str = "unknown") -> None:
        """
        Initialize the NER-aware BPE tokenizer.

        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            domain: Text domain for preprocessing ('twitter', 'headline', or 'unknown')
        """
        super().__init__()
        self.domain = domain.lower()
        print(f"Initializing NERBPETokenizer for domain: {self.domain}")
        self.vocab_size = vocab_size
        self._bpe_merge_ranks: Dict[Tuple[str, str], int] = {}

        # Add special tokens to vocabulary
        for tok in (self.UNK_TOKEN, self.SPACE_TOKEN):
            self._add_token(tok)

    def train(self, texts: List[str], *, char_limit: int = 256, bigram_quota: float = 0.3, 
              min_word_score: float = 15.0, min_bigram_score: float = 30.0) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            char_limit: Maximum number of most frequent characters to add first
            bigram_quota: Fraction of remaining slots reserved for bigrams (0.0-1.0)
            min_word_score: Minimum NER score for word tokens
            min_bigram_score: Minimum NER score for bigram tokens

        Training Process:
        1. Compute character, word, and bigram frequencies
        2. Add all unique characters to vocabulary
        3. Score and filter words/bigrams by NER relevance
        4. Add top-scoring tokens respecting bigram quota
        5. Run BPE merges to fill remaining vocabulary slots
        """
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
        entity_scores_heap = self._score_entities(word_freq, bigram_freq, min_word_score, min_bigram_score)
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
        for ch, _ in char_freq.most_common(min(char_limit, len(all_chars))):
            if self._remaining_slots > 0:
                self._add_token(ch)
                added_chars.add(ch)
        
        # Add remaining characters in sorted order for deterministic behavior
        for ch in sorted(all_chars - added_chars):
            if self._remaining_slots > 0:
                self._add_token(ch)
            else:
                break

    def _score_entities(self, word_freq: Counter, bigram_freq: Counter, 
                       min_word_score: float, min_bigram_score: float) -> List:
        """Score words and bigrams using NER-aware scoring, returning a heap."""
        entity_scores_heap = []
        
        # Score words
        for word, freq in tqdm(word_freq.items(), desc="Scoring words", leave=False):
            score = self._calc_ner_score((word, ""), freq)
            if score >= min_word_score:
                heapq.heappush(entity_scores_heap, (-score, word, freq))
        
        # Score bigrams
        for bigram, freq in tqdm(bigram_freq.items(), desc="Scoring bigrams", leave=False):
            w1, w2 = bigram.split(" ", 1)
            score = self._calc_ner_score((w1, " " + w2), freq)
            if score >= min_bigram_score:
                heapq.heappush(entity_scores_heap, (-score, bigram, freq))
        
        return entity_scores_heap

    def _add_top_entities(self, entity_scores_heap: List, bigram_quota: float) -> Tuple[int, int]:
        """Add top-scoring entities to vocabulary respecting bigram quota."""
        rest_slots = self._remaining_slots
        bg_slots = int(rest_slots * bigram_quota)
        word_slots = max(0, rest_slots - bg_slots)
        
        # Extract candidates
        all_candidates = []
        while entity_scores_heap and len(all_candidates) < rest_slots * 2:
            neg_score, token, freq = heapq.heappop(entity_scores_heap)
            all_candidates.append((token, -neg_score))
        
        # Separate bigrams and singles
        bigrams = [(s, sc) for s, sc in all_candidates if self._is_bigram(s)]
        singles = [(s, sc) for s, sc in all_candidates if not self._is_bigram(s)]
        
        # Add top bigrams
        added_bigrams = 0
        for s, score in bigrams[:bg_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            self._add_token(s)
            added_bigrams += 1
        
        # Add top singles
        added_singles = 0
        for s, score in singles[:word_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            self._add_token(s)
            added_singles += 1
        
        return added_bigrams, added_singles

    def _run_bpe_merges(self, texts: List[str]) -> None:
        """Run BPE merges to fill remaining vocabulary slots."""
        # Build word frequency for BPE
        word_freq = Counter()
        sample_size = min(100000, len(texts))
        
        for text in tqdm(texts[:sample_size], desc="Building word frequencies", leave=False):
            tokens = self._preprocess(text).split()
            for pos, w in enumerate(tokens):
                word_freq[tuple(w) + (self.END_MARK,)] += 1
                if pos > 0:
                    word_freq[tuple(" " + w) + (self.END_MARK,)] += 1
        
        # BPE iteration loop
        iteration = 0
        max_iterations = min(self._remaining_slots, 1000)
        
        with tqdm(total=max_iterations, desc="BPE merges", leave=False) as pbar:
            while self._remaining_slots > 0 and iteration < max_iterations:
                # Find best pair to merge
                best_pair = self._find_best_bpe_pair(word_freq)
                if not best_pair:
                    break
                
                # Apply merge
                self._perform_merge(best_pair, word_freq)
                self._bpe_merge_ranks[best_pair] = iteration
                
                iteration += 1
                pbar.update(1)
                
                if iteration % 100 == 0:
                    pbar.set_postfix({"merge": f"'{best_pair[0]}'+''{best_pair[1]}'"})

    def _find_best_bpe_pair(self, word_freq: Counter) -> Tuple[str, str] | None:
        """Find the best character pair to merge in BPE."""
        # Count all adjacent pairs
        pair_freq = Counter()
        for w, f in word_freq.items():
            for p in self._bpe_pairs(w):
                pair_freq[p] += f
        
        if not pair_freq:
            return None
        
        # Find best pair with NER scoring and bigram constraint
        best_pair = None
        best_score = -1
        
        for pair, freq in pair_freq.items():
            merged_token = "".join(pair)
            
            # Check bigram constraint
            if self._violates_bigram_constraint(merged_token):
                continue
            
            # Calculate NER score
            ner_score = self._calc_ner_score(pair, freq)
            if ner_score > best_score:
                best_score = ner_score
                best_pair = pair
        
        return best_pair

    # -------------------------------------------------------------------------
    # Private Encoding Methods
    # -------------------------------------------------------------------------

    def _find_best_match(self, text: str, start: int) -> Tuple[str | None, int]:
        """Find the best token match starting at position start."""
        best_match = None
        best_length = 0
        
        # Try word-level matches first (prefer bigrams)
        if text[start] != " ":
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
        
        # Fallback to character-level longest match
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
        base_score = math.log(frequency + 1)
        features = self._get_ner_indicators(joined)
        
        # NER-specific bonuses
        ner_bonus = 0.0
        
        # Character tokens
        if features['length'] == 1:
            ner_bonus += 2
            if joined.isupper():
                ner_bonus += 3
            elif joined.isdigit():
                ner_bonus += 5
            elif joined == ' ':
                ner_bonus += 10
        
        # Word tokens
        elif features['word_count'] == 1:
            ner_bonus += 10
            if features['has_uppercase']:
                ner_bonus += 50  # Title case very important for NER
            if features['has_mixed_case']:
                ner_bonus += 15
            if features['has_digits']:
                ner_bonus += 10
            if features['has_special_chars']:
                ner_bonus += 8
        
        # Bigram tokens (highest priority)
        elif features['word_count'] == 2:
            ner_bonus += 25
            if features['is_capitalized_pair']:
                ner_bonus += 100  # Both words capitalized - very likely entity
            elif features['title_case_ratio'] > 0:
                ner_bonus += 50
            if features['has_digits']:
                ner_bonus += 10
        
        # Cross-word character pairs
        elif features['has_space'] and features['length'] == 2:
            ner_bonus += 30
            if pair[0] == ' ' and len(pair[1]) > 0 and pair[1][0].isupper():
                ner_bonus += 40
            elif len(pair[0]) > 0 and pair[0][0].isupper() and len(pair[1]) > 0 and pair[1][0].isupper():
                ner_bonus += 25
        
        # Additional bonuses
        if features['has_space']:
            ner_bonus += 40
        if features['has_mixed_case']:
            ner_bonus += 15
        if features['has_digits']:
            ner_bonus += 10
        if features['has_special_chars']:
            ner_bonus += 8
        
        return base_score + ner_bonus

    def _get_ner_indicators(self, token: str) -> Dict[str, float]:
        """Extract NER-relevant features from a token."""
        features = {
            'length': len(token),
            'word_count': len(token.strip().split()),
            'has_space': 1.0 if ' ' in token else 0.0,
            'has_digits': 1.0 if any(c.isdigit() for c in token) else 0.0,
            'has_uppercase': 1.0 if any(c.isupper() for c in token) else 0.0,
            'has_lowercase': 1.0 if any(c.islower() for c in token) else 0.0,
            'has_mixed_case': 0.0,
            'has_special_chars': 1.0 if any(not c.isalnum() and c != ' ' for c in token) else 0.0,
            'title_case_ratio': 0.0,
            'is_word_pair': False,
            'is_capitalized_pair': False,
        }
        
        # Mixed case detection
        if features['has_uppercase'] > 0 and features['has_lowercase'] > 0:
            features['has_mixed_case'] = 1.0
        
        # Word-level features
        words = token.strip().split()
        
        if len(words) == 1 and words[0]:
            features['title_case_ratio'] = 1.0 if words[0].istitle() else 0.0
        elif len(words) == 2:
            features['is_word_pair'] = True
            title_case_count = sum(1 for w in words if w and w.istitle())
            features['title_case_ratio'] = title_case_count / 2.0
            features['is_capitalized_pair'] = title_case_count == 2
        
        return features

    # -------------------------------------------------------------------------
    # Preprocessing Methods
    # -------------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """Apply domain-specific preprocessing."""
        if self.domain == "twitter":
            return self._pre_twitter(text)
        elif self.domain in {"headline", "headlines"}:
            return self._pre_headline(text)
        else:
            return self._pre_generic(text)

    def _pre_twitter(self, text: str) -> str:
        """Preprocess Twitter text."""
        # text = unicodedata.normalize("NFKC", text)
        # text = unescape(text)
        # text = _TW_URL.sub("<URL>", text)
        # text = _TW_USER.sub("<USER>", text)
        # text = _HASHTAG_RE.sub("<HASHTAG>", text)
        # text = _EMOJI.sub("<EMOJI>", text)
        # text = text.translate(UNICODE_PUNCT_TABLE)
        # text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_headline(self, text: str) -> str:
        """Preprocess headline text."""
        # text = unicodedata.normalize("NFKC", text)
        # text = _NEWS_DATE.sub("[DATE]", text)
        # text = _EMOJI.sub("<EMOJI>", text)
        # text = text.translate(UNICODE_PUNCT_TABLE)
        # text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_generic(self, text: str) -> str:
        """Preprocess generic text."""
        # text = unicodedata.normalize("NFKC", text)
        # text = unescape(text)
        # text = _EMOJI.sub("<EMOJI>", text)
        # text = text.translate(UNICODE_PUNCT_TABLE)
        # text = _PUNCT_PAD.sub(r" \1 ", text)
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
        Analyze trained tokens and output detailed statistics to a text file.
        
        Creates a comprehensive report including:
        - Token frequency analysis
        - Token type distribution
        - Top tokens by category
        - Detailed token list with rankings
        """
        from datetime import datetime
        
        # Calculate token frequencies in the corpus
        token_usage = Counter()
        sample_texts = texts[:5000]  # Sample for efficiency
        
        for text in tqdm(sample_texts, desc="Analyzing token usage", leave=False):
            token_ids = self.encode(text)
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    token_usage[self.id_to_token[token_id]] += 1
        
        # Categorize tokens
        token_categories = {
            'special': [],
            'characters': [],
            'words': [],
            'bigrams': [],
            'subwords': [],
            'merged_chars': []
        }
        
        for token, freq in token_usage.items():
            if token in [self.UNK_TOKEN, self.SPACE_TOKEN, "[PAD]", "[BOS]", "[EOS]"]:
                token_categories['special'].append((token, freq))
            elif len(token) == 1:
                token_categories['characters'].append((token, freq))
            elif self._is_bigram(token):
                token_categories['bigrams'].append((token, freq))
            elif len(token.strip().split()) == 1 and len(token) > 1:
                if ' ' in token or any(not c.isalnum() and c != ' ' for c in token):
                    token_categories['subwords'].append((token, freq))
                else:
                    token_categories['words'].append((token, freq))
            else:
                token_categories['merged_chars'].append((token, freq))
        
        # Sort categories by frequency
        for category in token_categories:
            token_categories[category].sort(key=lambda x: x[1], reverse=True)
        
        # Generate analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"c:\\Users\\ronic\\VS_Coding\\NLP\\HW\\HW2\\token_analysis_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_token_analysis_report(f, token_categories, token_usage, texts)
        
        print(f"Token analysis saved to: {filename}")

    def _write_token_analysis_report(self, f, token_categories: Dict, token_usage: Counter, texts: List[str]) -> None:
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
        
        # Most frequent tokens overall
        f.write("TOP 50 MOST FREQUENT TOKENS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Rank':<5} {'Token':<30} {'Freq':<10} {'Type':<15}\n")
        f.write("-" * 65 + "\n")
        
        all_tokens_by_freq = sorted(token_usage.items(), key=lambda x: x[1], reverse=True)
        for rank, (token, freq) in enumerate(all_tokens_by_freq[:50], 1):
            token_type = self._get_token_type(token)
            token_repr = repr(token) if len(token) <= 25 else repr(token[:22] + "...")
            f.write(f"{rank:<5} {token_repr:<30} {freq:<10} {token_type:<15}\n")
        
        f.write("\n")
        
        # Detailed category analysis
        for category, tokens in token_categories.items():
            if not tokens:
                continue
                
            f.write(f"{category.upper()} TOKENS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Count: {len(tokens)}\n")
            f.write(f"Top 20 by frequency:\n")
            f.write(f"{'Rank':<5} {'Token':<35} {'Frequency':<10}\n")
            f.write("-" * 55 + "\n")
            
            for rank, (token, freq) in enumerate(tokens[:20], 1):
                token_repr = repr(token) if len(token) <= 30 else repr(token[:27] + "...")
                f.write(f"{rank:<5} {token_repr:<35} {freq:<10}\n")
            
            f.write("\n")
        
        # NER-specific analysis
        f.write("NER-SPECIFIC ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Capitalized tokens
        capitalized_tokens = [(t, f) for t, f in token_usage.items() 
                            if len(t.strip().split()) >= 1 and t.strip().split()[0].istitle()]
        capitalized_tokens.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"Capitalized tokens: {len(capitalized_tokens)}\n")
        f.write("Top 15 capitalized tokens:\n")
        for rank, (token, freq) in enumerate(capitalized_tokens[:15], 1):
            f.write(f"  {rank:2}. {repr(token)} (freq: {freq})\n")
        f.write("\n")
        
        # Bigram analysis
        bigram_tokens = token_categories['bigrams']
        if bigram_tokens:
            capitalized_bigrams = [(t, f) for t, f in bigram_tokens 
                                 if all(w.istitle() for w in t.split())]
            f.write(f"Total bigrams: {len(bigram_tokens)}\n")
            f.write(f"Capitalized bigrams: {len(capitalized_bigrams)}\n")
            f.write("Top 10 capitalized bigrams:\n")
            for rank, (token, freq) in enumerate(capitalized_bigrams[:10], 1):
                f.write(f"  {rank:2}. \"{token}\" (freq: {freq})\n")
            f.write("\n")
        
        # BPE merge analysis
        f.write("BPE MERGE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total BPE merges performed: {len(self._bpe_merge_ranks)}\n")
        f.write("Latest 10 BPE merges:\n")
        sorted_merges = sorted(self._bpe_merge_ranks.items(), key=lambda x: x[1], reverse=True)
        for (char1, char2), rank in sorted_merges[:10]:
            merged = char1 + char2
            freq = token_usage.get(merged, 0)
            f.write(f"  {repr(char1)} + {repr(char2)} -> {repr(merged)} (freq: {freq})\n")
        f.write("\n")
        
        # Complete token list
        f.write("COMPLETE TOKEN VOCABULARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'ID':<6} {'Token':<40} {'Freq':<8} {'Type':<12}\n")
        f.write("-" * 70 + "\n")
        
        # Sort by token ID for deterministic output
        all_tokens_by_id = sorted(self.token_to_id.items(), key=lambda x: x[1])
        for token, token_id in all_tokens_by_id:
            freq = token_usage.get(token, 0)
            token_type = self._get_token_type(token)
            token_repr = repr(token) if len(token) <= 35 else repr(token[:32] + "...")
            f.write(f"{token_id:<6} {token_repr:<40} {freq:<8} {token_type:<12}\n")

    def _get_token_type(self, token: str) -> str:
        """Determine the type of a token for analysis."""
        if token in [self.UNK_TOKEN, self.SPACE_TOKEN, "[PAD]", "[BOS]", "[EOS]"]:
            return "special"
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