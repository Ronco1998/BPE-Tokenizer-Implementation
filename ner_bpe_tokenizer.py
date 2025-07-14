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
import re
import unicodedata
from collections import Counter
from html import unescape
from typing import Dict, List, Tuple
from tqdm import tqdm

from base_tokenizer import BaseTokenizer

# # Configure UTF-8 output for Windows compatibility
# try:
#     sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# except AttributeError:
#     pass  # Python < 3.7 or redirected stream

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
    
    # Stop-word whitelist for NER scoring boost
    STOP_WORD_WHITELIST = {"and", "or", "but", "if", "because", "i", "you", "me", "my", "your"}
    
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
        self.strict_clean = False

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

    def train(self, texts: List[str], *, char_limit: int | None = None, bigram_quota: float = 0.12, 
              min_word_score: float = 18.0, min_bigram_score: float = 60.0, min_frequency: int = 1) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            char_limit: Maximum number of most frequent characters to add first
            bigram_quota: Fraction of remaining slots reserved for bigrams (0.0-1.0)
            min_word_score: Minimum NER score for word tokens
            min_bigram_score: Minimum NER score for bigram tokens
            min_frequency: Minimum frequency threshold for words and bigrams (default: 1)

        Training Process:
        1. Compute character, word, and bigram frequencies
        2. Add all unique characters to vocabulary
        3. Score and filter words/bigrams by NER relevance
        4. Add top-scoring tokens respecting bigram quota
        5. Run BPE merges to fill remaining vocabulary slots
        """
        # Reduce aggressive parameter adjustments that hurt performance
        if char_limit is None:
            char_limit = 256            # keep most glyphs but still adaptive
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
                       min_word_score: float, min_bigram_score: float, min_frequency: int = 1) -> List:
        """Score words and bigrams using NER-aware scoring, returning a heap."""
        entity_scores_heap = []
        word_scores = []
        bigram_scores = []
        
        # Score words with minimum frequency requirement
        for word, freq in tqdm(word_freq.items(), desc="Scoring words", leave=False):
            # Filter out invalid tokens before scoring
            if not self._is_valid_token(word):
                continue
            if freq >= min_frequency:
                score = self._calc_ner_score((word, ""), freq)
                word_scores.append(score)
                if score >= min_word_score:
                    heapq.heappush(entity_scores_heap, (-score, word, freq))

        # Score bigrams
        for bigram, freq in tqdm(bigram_freq.items(), desc="Scoring bigrams", leave=False):
            # Filter out invalid tokens before scoring
            if not self._is_valid_token(bigram):
                continue
            if freq >= 2:
                w1, w2 = bigram.split(" ", 1)
                score = self._calc_ner_score((w1, " " + w2), freq)
                bigram_scores.append(score)
                if score >= min_bigram_score:
                    heapq.heappush(entity_scores_heap, (-score, bigram, freq))
        
        # Print average scores
        if word_scores:
            avg_word_score = sum(word_scores) / len(word_scores)
            print(f"Average word score: {avg_word_score:.2f}")
        else:
            print("No valid word scores to average.")
        if bigram_scores:
            avg_bigram_score = sum(bigram_scores) / len(bigram_scores)
            print(f"Average bigram score: {avg_bigram_score:.2f}")
        else:
            print("No valid bigram scores to average.")
        
        return entity_scores_heap

    def _add_top_entities(self, entity_scores_heap: List, bigram_quota: float) -> Tuple[int, int]:
        """Add top-scoring entities to vocabulary respecting bigram quota."""
        bpe_slots = min (200, self.vocab_size // 4)
        rest_slots = self._remaining_slots - bpe_slots  # Reserve slots for BPE merges
        bg_slots = int(rest_slots * bigram_quota)
        word_slots = max(0, rest_slots - bg_slots - bpe_slots)
        print(f"Adding top entities: {bg_slots} bigrams, {word_slots} words (total {rest_slots} slots remaining)")
        
        # Extract candidates - heap structure is (-score, token, freq)
        all_candidates = []
        while entity_scores_heap:  # Extract more candidates
            neg_score, token, freq = heapq.heappop(entity_scores_heap)
            all_candidates.append((token, -neg_score, freq))
        
        # Separate bigrams and singles
        bigrams = [(t, sc, f) for t, sc, f in all_candidates if self._is_bigram(t)]
        singles = [(t, sc, f) for t, sc, f in all_candidates if not self._is_bigram(t)]
        
        # Sort by combined score and frequency (balanced approach)
        # Higher frequency tokens get preference among similar scores
        bigrams.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by score first, then frequency
        singles.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by score first, then frequency
        
        print(f"Found {len(bigrams)} bigram candidates and {len(singles)} word candidates")
        
        # Add top bigrams with better selection criteria
        added_bigrams = 0
        for s, score, freq in bigrams[:bg_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            # Additional filtering: prioritize higher frequency and meaningful patterns
            if freq >= 1:  # Minimum frequency requirement (reduced from 2)
                self._add_token(s)
                self._token_frequencies[s] = freq
                added_bigrams += 1
        
        # Add top singles with better selection criteria
        added_singles = 0
        for s, score, freq in singles[:word_slots]:
            if len(self._token_to_id) >= self.vocab_size:
                break
            # Additional filtering: prioritize higher frequency words
            if freq >= 2:  # Minimum frequency requirement (reduced from 2)
                self._add_token(s)
                self._token_frequencies[s] = freq
                added_singles += 1
        
        print(f"Added {added_bigrams} bigrams and {added_singles} words")
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
        
        print(f"Created {len(word_freq)} unique words for BPE processing")
        
        # More aggressive BPE parameters
        iteration = 0
        max_iterations = min(self._remaining_slots, 2000)
        
        with tqdm(total=max_iterations, desc="BPE merges", leave=False) as pbar:
            while self._remaining_slots > 0 and iteration < max_iterations:
                # Find best pair to merge
                result = self._find_best_bpe_pair(word_freq)
                if result is None:
                    print(f"No valid pairs found at iteration {iteration}")
                    break
                best_pair, best_freq = result
                
                # Apply merge
                merged_token = "".join(best_pair)
                self._perform_merge(best_pair, word_freq)
                self._token_frequencies[merged_token] = best_freq
                self._bpe_merge_ranks[best_pair] = iteration
                
                iteration += 1
                pbar.update(1)
                
                if iteration % 50 == 0:
                    pbar.set_postfix({"merge": f"'{best_pair[0]}'+''{best_pair[1]}'", "freq": best_freq, "remaining": self._remaining_slots})
        
        print(f"BPE completed after {iteration} iterations, {self._remaining_slots} slots remaining")

    def _find_best_bpe_pair(self, word_freq: Counter) -> Tuple[Tuple[str, str], int] | None:
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
        best_pair_freq = None

        # Sort by frequency first to prioritize high-frequency pairs
        for pair, freq in pair_freq.most_common():
            merged_token = "".join(pair)

            # Check bigram constraint
            if self._violates_bigram_constraint(merged_token):
                continue

            if merged_token in self._token_to_id:
                continue

            # Calculate NER score
            ner_score = self._calc_ner_score(pair, freq) * freq
            if ner_score > best_score:
                best_score = ner_score
                best_pair = pair
                best_pair_freq = freq

        if best_pair is not None and best_pair_freq is not None:
            return (best_pair, best_pair_freq)
        else:
            return None

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

    def _get_ner_indicators(self, token: str) -> Dict[str, float]:
        """
        Extract orthographic and shape features that proved most predictive
        for NE tokens in train_1/dev_1.
        """
        SPECIAL = set(".,!?;:'`_-")
        length = len(token)
        words = token.split()

        # Basic flags
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        is_all_caps = token.isupper() and token.isalpha()
        is_all_lower = token.islower() and token.isalpha()
        has_mixed_case = has_upper and has_lower and not token.istitle()
        has_digits = any(c.isdigit() for c in token)
        has_special = any(c in SPECIAL for c in token)
        num_words = len(words)

        # Title-case ratio (0, 0.5 or 1.0 for up to two-word merges)
        if num_words == 1:
            title_ratio = 1.0 if words[0].istitle() else 0.0
        elif num_words == 2:
            title_ratio = sum(w.istitle() for w in words) / 2.0
        else:
            title_ratio = 0.0  # we never merge >2 words

        # Check if token is on stop-word whitelist
        is_stopword = token.lower() in self.STOP_WORD_WHITELIST

        return {
            "length": length,
            "num_words": num_words,
            "title_ratio": title_ratio,
            "is_all_caps": is_all_caps,
            "is_all_lower": is_all_lower,
            "has_mixed_case": has_mixed_case,
            "has_digits": has_digits,
            "has_special": has_special,
            "has_space": " " in token,
            "is_stopword": is_stopword,
        }

    def _calc_ner_score(self, pair: Tuple[str, str], frequency: int = 1) -> float:
        """
        Score a candidate pair for how useful it is as a named-entity token.
        The score is `log(freq) + ner_bonus`.  Weights come from corpus stats
        (title-case strongest; all-caps & mixed-case useful; pure lowercase rare).
        """
        joined = "".join(pair)
        f = self._get_ner_indicators(joined)

        # Base term from frequency (diminishing returns via log)
        score = math.log(frequency + 1)

        # Check for stop-word following title-case pattern
        # Boost for bigrams where the stopword is the first word and the second is title-case (e.g., "the New")
        if f["is_stopword"] and len(pair) == 2 and pair[1].strip():
            second_part = pair[1].strip()
            if second_part and second_part.istitle():
                score += 8  # Boost for stop words before title-case (e.g., "the New")

        # Check for <UPPER> <digit> pattern
        if f["num_words"] == 2:
            words = joined.split()
            if (len(words) == 2 and 
                words[0].isupper() and words[0].isalpha() and
                any(c.isdigit() for c in words[1])):
                score += 10  # Boost for upper-digit patterns (e.g., "F 22", "GPS 3")

        # ------- single character (kept tiny) -------
        if f["length"] == 1:
            if joined.isupper():
                score += 6          # initials, acronyms ("U")  
            elif joined.isdigit():
                score += 3

        # ------- one-word tokens -------
        elif f["num_words"] == 1:
            if f["title_ratio"] == 1.0:            # 68 % of corpus NEs
                score += 60
            elif f["is_all_caps"]:                 # "USA", "UN"
                score += 25
            elif f["has_mixed_case"]:              # "iPhone", "eBay"
                score += 18
            elif f["is_all_lower"]:                # seldom true en-mass
                score -= 5

            # Minor cues - enhanced bonuses when title-case is present
            if 5 <= f["length"] <= 7:              # modal length bucket
                score += 5
            if f["has_special"]:
                # Give +4 (was +2) for hyphen or apostrophe, but only if at least one letter is title-case
                if any(c in "-'" for c in joined) and any(c.isupper() for c in joined):
                    score += 4
                else:
                    score += 2  # Keep original bonus otherwise

        # ------- two-word merges / bigrams -------
        elif f["num_words"] == 2:
            score += 50                            # bigrams are gold for NER
            if f["title_ratio"] == 1.0:            # "New York"
                score += 40
            elif f["title_ratio"] >= 0.5:          # one title word
                score += 20

            # Secondary boosts
            if f["has_digits"]:
                score += 6                         # "Formula 1"
            if f["has_special"]:
                score += 3                         # "AT&T Park"

        # ------- cross-word char-pairs (" N", "Y ") -------
        elif f["has_space"] and f["length"] == 2:
            score += 20                            # helps seed bigrams

        return score


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
        # Apply gentle Unicode normalization (NFC instead of NFKC)
        # text = unicodedata.normalize("NFC", text)
        text = unescape(text)
        
        # Remove problematic characters that cause encoding issues
        # text = self.PROBLEMATIC_CHARS_RE.sub('', text)
        
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
        
        # Apply proper Unicode character translation
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_headline(self, text: str) -> str:
        """Preprocess headline text and track preprocessing token usage."""
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
        
        # Apply proper Unicode character translation
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = _PUNCT_PAD.sub(r" \1 ", text)
        return " ".join(text.split())

    def _pre_generic(self, text: str) -> str:
        """Preprocess generic text and track preprocessing token usage."""
        # Track EMOJI replacements
        emoji_count = len(_EMOJI.findall(text))
        if emoji_count > 0:
            self._token_frequencies["<EMOJI>"] = self._token_frequencies.get("<EMOJI>", 0) + emoji_count
        text = _EMOJI.sub("<EMOJI>", text)
        
        # Apply proper Unicode character translation
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

    def _is_valid_token(self, token: str) -> bool:
        """Check if token should be added to vocabulary (filters problematic characters and noise)."""
        # Always allow special tokens and preprocessing tokens
        special_tokens = {self.UNK_TOKEN, self.SPACE_TOKEN}
        all_preprocessing_tokens = set()
        for domain_tokens in self.PREPROCESSING_TOKENS.values():
            all_preprocessing_tokens.update(domain_tokens)
        
        if token in special_tokens or token in all_preprocessing_tokens:
            return True
        
        # For regular tokens, check if they contain problematic characters
        if self.PROBLEMATIC_CHARS_RE.search(token):
            return False
        
        # Define known punctuation marks that are allowed
        known_punctuation = {' ', '.', ',', '!', '?', ':', ';', "'", '"', '-', '(', ')', '[', ']', '{', '}', '&', '#', '@', '$', '%', '*', '+', '=', '/', '\\', '|', '<', '>', '~', '`', '^', '_'}
        
        # Check if token contains only alphanumeric characters, spaces, and known punctuation
        for char in token:
            if not (char.isalnum() or char in known_punctuation):
                return False
        
        # Additional filtering for meaningful entity patterns
        if self._is_bigram(token):
            return self._is_meaningful_bigram(token)
        elif len(token.strip().split()) == 1:
            return self._is_meaningful_word(token)
        
        return True
    
    def _is_meaningful_bigram(self, token: str) -> bool:
        """Check if a bigram token represents a meaningful entity pattern."""
        words = token.split()
        if len(words) != 2:
            return False
        
        word1, word2 = words
        
        # Filter out noise patterns
        # More flexible approach for single character words
        # Allow single characters that are uppercase, digits, or common linguistic patterns
        def is_meaningful_single_char(char):
            if len(char) != 1:
                return True  # Not a single character
            # Allow uppercase letters (often initials or abbreviations)
            if char.isupper():
                return True
            # Allow digits (often part of model names, years, etc.)
            if char.isdigit():
                return True
            # Allow common single letter words in English
            if char.lower() in {'a', 'i'}:
                return True
            # Reject other single lowercase letters
            return False
        
        if not is_meaningful_single_char(word1) or not is_meaningful_single_char(word2):
            return False
        
        # Reject if both words are lowercase (less likely to be entities)
        if word1.islower() and word2.islower():
            return False
        
        # Reject if it's mostly punctuation noise
        punct_count = sum(1 for c in token if not c.isalnum() and c != ' ')
        if punct_count > len(token) // 3:  # More than 1/3 punctuation
            return False
        
        # Reject Twitter-specific noise patterns
        if token.startswith('*') and token.endswith('*'):
            return False
        if '-' in word1 and '-' in word2:  # Double hyphenated words are likely noise
            return False
        
        return True
    
    def _is_meaningful_word(self, token: str) -> bool:
        """Check if a word token represents a meaningful entity pattern."""
        token = token.strip()
        
        # Allow single characters with more nuanced rules
        if len(token) == 1:
            # Allow uppercase letters (initials, abbreviations)
            if token.isupper():
                return True
            # Allow digits (years, model numbers, etc.)
            if token.isdigit():
                return True
            # Allow common single letter words
            if token.lower() in {'a', 'i'}:
                return True
            # Allow common punctuation that might be meaningful
            if token in {'.', '!', '?', '-', '&'}:
                return True
            # Reject other single characters
            return False
        
        # For longer tokens, reject if it's mostly punctuation
        punct_count = sum(1 for c in token if not c.isalnum())
        if punct_count > len(token) // 2:  # More than half punctuation
            return False
        
        # Reject common noise patterns but be more specific
        # Only reject if it's clearly decorative/formatting
        if (token.startswith('*') and token.endswith('*') and len(token) > 2) or \
           (token.startswith('[') and token.endswith(']') and len(token) > 2):
            return False
        
        # Reject tokens that are only repeated characters (like "---" or "...")
        if len(set(token)) == 1 and len(token) > 2:
            return False
        
        return True

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