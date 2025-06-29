from __future__ import annotations

import heapq
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Iterable, Optional
import unicodedata
from html import unescape
from tqdm.auto import tqdm

from base_tokenizer import BaseTokenizer

# ────────────────────────────────────────────────────────────────────────────
# Helper regexes & constants (shared for every domain)
# ────────────────────────────────────────────────────────────────────────────
_TW_USER  = re.compile(r"@[A-Za-z0-9_]{1,15}")  # Twitter @handle
_TW_URL   = re.compile(r"https?://\S+")        # http(s) URLs
_HASHTAG_RE = re.compile(r"#\w[\w\d_]*")      # #hashtag
_NEWS_PUN = re.compile(r"([,.;:!?()\"'])")    # punctuation splitter #TODO
_NEWS_DATE = re.compile(r'\b\d{4}-\d{2}-\d{2}\b') # dates like 2023-10-05
REP_CHARS = re.compile(r"(.)\1{2,}")
PUNCT_PAD = re.compile(r"([,.;:!?()\"])")
# One coarse emoji detector (covers all BMP + supplementary planes)
_EMOJI = re.compile(
    r"[\U0001F1E6-\U0001F1FF]|"      # flags
    r"[\U0001F300-\U0001F5FF]|"      # symbols & pictographs
    r"[\U0001F600-\U0001F64F]|"      # emoticons
    r"[\U0001F680-\U0001F6FF]|"      # transport & map
    r"[\u2600-\u26FF]|"              # misc symbols
    r"[\u2700-\u27BF]"               # dingbats
)

EMOTICON_RE = re.compile(
    r"""
    (?:
        [:=;8]               # eyes
        (?:-|'|~)?           # optional nose
        [)(DOPp/*\\]         # mouth
    )
    |
    (?:\^{2})               # '^^'
    """,
    re.VERBOSE,
)

# Printable ASCII punctuation + ‘…’ (unicode ellipsis)
_PUNCT_CHARS = list(r"""!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + ["…"]

UNICODE_PUNCT_TABLE = str.maketrans({
    "“": '"',    "”": '"',
    "‘": "'",    "’": "'",
    "«": '"',    "»": '"',
    "—": "-",    "–": "-",
    "…": "...",
})
END_WORD_MARK = "</w>"

def _split_hashtag(match: re.Match) -> str:
    """
    Replace '#Word123' → '<HASHTAG> Word123'
    so the word itself is still visible to NER.
    """
    tag_body = match.group()[1:]           # drop the leading '#'
    return f"<HASHTAG> {tag_body}"

# ────────────────────────────────────────────────────────────────────────────
# BPETokenizer class
# ────────────────────────────────────────────────────────────────────────────
class BPETokenizer(BaseTokenizer):
    """Byte‑Pair‑Encoding tokenizer with optional per‑domain special tokens."""

    def __init__(
        self,
        num_merges: int = 1_000,
        log_every: int = 20,
        domain: str = "generic",
        special_tokens: Optional[List[str]] = None,
        vocab_out_path: Optional[str] = None,
        debug: bool = False,  # Add debug flag
    ) -> None:
        super().__init__()

        self.num_merges   = num_merges
        self.log_every    = log_every
        self.domain       = domain.lower()
        self.vocab_out    = vocab_out_path or f"{self.domain}_vocab.txt"
        self.debug        = debug  # Store debug flag

        # Debug tracking attributes
        if self.debug:
            self.debug_stats = {
                'total_tokens_created': 0,
                'unk_tokens_created': 0,
                'unk_token_examples': [],
                'encoding_calls': 0,
                'merge_applications': {},
                'preprocessing_changes': []
            }

        # ── 1. Domain‑specific special tokens ─────────────────────────────
        domain_extras: List[str] = []
        if self.domain == "twitter":
            domain_extras = ["<URL>", "<USER>", "<HASHTAG>", "<EMOTICON>"]
        elif self.domain == "headlines":  # News/headlines domain
            domain_extras = ["[DATE]"]
        elif self.domain == "unknown":  # Mixed domain (Twitter + Headlines)
            domain_extras = ["<URL>", "<USER>", "<HASHTAG>", "<EMOTICON>", "[DATE]"]
        # Generic emoji placeholder is always useful regardless of domain
        domain_extras.append("<EMOJI>")

        # Merge built‑in + user‑supplied special tokens
        if special_tokens is None:
            special_tokens = []
        self.special_tokens.update({tok: len(self.special_tokens)+i for i, tok in enumerate(domain_extras + special_tokens)})


        # ── 2. Core vocabulary initialisation ─────────────────────────────
        # Map special tokens first (so they get the lowest ids after PAD/UNK/...)
        self.token_to_id.update(self.special_tokens)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # Pre‑load punctuation so they never become OOV
        for ch in _PUNCT_CHARS:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        # Add common Unicode characters that appear in entity names
        common_unicode_chars = [
            # Latin extended characters common in names
            'á', 'à', 'â', 'ä', 'ã', 'å', 'æ', 'ç', 'é', 'è', 'ê', 'ë', 'í', 'ì', 'î', 'ï',
            'ñ', 'ó', 'ò', 'ô', 'ö', 'õ', 'ø', 'ú', 'ù', 'û', 'ü', 'ý', 'ÿ', 'ß',
            'Á', 'À', 'Â', 'Ä', 'Ã', 'Å', 'Æ', 'Ç', 'É', 'È', 'Ê', 'Ë', 'Í', 'Ì', 'Î', 'Ï',
            'Ñ', 'Ó', 'Ò', 'Ô', 'Ö', 'Õ', 'Ø', 'Ú', 'Ù', 'Û', 'Ü', 'Ý', 'Ÿ',
            # Additional characters for international names
            'š', 'ž', 'č', 'ř', 'ů', 'ě', 'ť', 'ď', 'ň', 'ľ', 'ĺ', 'ŕ',
            'Š', 'Ž', 'Č', 'Ř', 'Ů', 'Ě', 'Ť', 'Ď', 'Ň', 'Ľ', 'Ĺ', 'Ŕ',
            # Cyrillic characters commonly romanized
            'ć', 'đ', 'ł', 'ń', 'ś', 'ź', 'ż', 'Ć', 'Đ', 'Ł', 'Ń', 'Ś', 'Ź', 'Ż',
            # Common symbols in names  
            "'", "-", ".", "&"
        ]
        for ch in common_unicode_chars:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch


        # Reserve END_WORD_MARK as a real vocabulary symbol (not emitted)
        if END_WORD_MARK not in self.token_to_id:
            wid = len(self.token_to_id)
            self.token_to_id[END_WORD_MARK] = wid
            self.id_to_token[wid] = END_WORD_MARK
        
        self.space_token = END_WORD_MARK
        self.special_tokens[self.space_token] = len(self.special_tokens)

        # ── 3. Training artefacts – one per *instance* (i.e. per domain) ──
        self.merges: List[Tuple[str, str]] = []          # ordered list of merges
        self.ranks:  Dict[Tuple[str, str], int] = {}     # quick look‑up → rank        # Working state (used only during `train()`)
        self.word_freqs: Dict[Tuple[str, ...], int] = {}
        self.pair_stats: Dict[Tuple[str, str], int] = {}
        self._heap: List[Tuple[int, Tuple[str, str]]] = []
        
        # Track word bigram frequencies for reporting
        self.word_bigram_freqs: Dict[Tuple[str, str], int] = {}

    # ────────────────────────────────────────────────────────────────────
    # Domain-specific preprocessing methods
    # ────────────────────────────────────────────────────────────────────
    def _preprocess_twitter(self, text: str) -> str:
        """Twitter-specific preprocessing."""
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        text = _TW_URL.sub("<URL>",  text)
        text = _TW_USER.sub("<USER>", text)
        text = _HASHTAG_RE.sub(_split_hashtag, text)
        text = EMOTICON_RE.sub("<EMOTICON>", text)
        text = _EMOJI.sub('<EMOJI>', text)
        # Clean up problematic characters but preserve Unicode text
        text = text.replace("\uFE0F", "")   # VS-16 (emoji modifier)
        text = text.replace("\u200B", " ")  # zero-width space -> regular space
        text = text.replace("\uFEFF", "")   # byte-order mark
        
        # Remove only truly problematic non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        text = text.translate(UNICODE_PUNCT_TABLE)
        
        # Fix common concatenated word patterns BEFORE any other processing
        # This prevents common concatenations from becoming UNK tokens
        common_patterns = [
            (r'\btothe\b', 'to the'),
            (r'\binthe\b', 'in the'), 
            (r'\bonthe\b', 'on the'),
            (r'\bforthe\b', 'for the'),
            (r'\bofthe\b', 'of the'),
            (r'\bfromthe\b', 'from the'),
            (r'\bwiththe\b', 'with the'),
            (r'\batthe\b', 'at the'),
            (r'\bbythe\b', 'by the'),
            (r'\bunderthe\b', 'under the'),
            (r'\boverthe\b', 'over the'),
            (r'\bandthe\b', 'and the'),
            (r'\bofthis\b', 'of this'),
            (r'\bforthis\b', 'for this'),
            (r'\bwiththis\b', 'with this'),
            (r'\binthat\b', 'in that'),
            (r'\bforthat\b', 'for that'),
            (r'\bonthat\b', 'on that'),
            # Keep these as single words
            (r'\btoday\b', 'today'),
            (r'\btomorrow\b', 'tomorrow'),
            (r'\byesterday\b', 'yesterday'),
        ]
        
        for pattern, replacement in common_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        # Preserve capitalization patterns that might indicate entities
        words = text.split()
        processed_words = []
        for word in words:
            # Keep all-caps words (often acronyms/entities)
            if word.isupper() and len(word) > 1:
                processed_words.append(word)
            # Keep capitalized words (potential proper nouns)
            elif word and word[0].isupper():
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        text = " ".join(processed_words)
        return text

    def _preprocess_headlines(self, text: str) -> str:
        """Headlines/news-specific preprocessing."""
        text = _NEWS_PUN.sub(r" \1 ", text)
        text = _EMOJI.sub('<EMOJI>', text)
        # Replace dates using the new regex constant
        text = _NEWS_DATE.sub('[DATE]', text)
        text = unicodedata.normalize("NFKC", text)
        return text

    def _preprocess_generic(self, text: str) -> str:
        """Generic domain preprocessing."""
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        text = _EMOJI.sub('<EMOJI>', text)
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        # Generic cleanup
        text = text.replace("\uFE0F", "")   # VS-16 (emoji "colour" selector)
        text = text.replace("\u200B", "")   # zero-width space
        text = text.replace("\uFEFF", "")   # byte-order mark
        
        return text.strip()

    def _preprocess_unknown(self, text: str) -> str:
        """Unknown domain preprocessing - combines Twitter and Headlines approaches."""
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        
        # Apply Twitter-specific replacements
        text = _TW_URL.sub("<URL>",  text)
        text = _TW_USER.sub("<USER>", text)
        text = _HASHTAG_RE.sub(_split_hashtag, text)
        text = EMOTICON_RE.sub("<EMOTICON>", text)
        
        # Apply Headlines-specific replacements
        text = _NEWS_DATE.sub('[DATE]', text)
        
        # Common processing
        text = _EMOJI.sub('<EMOJI>', text)
        text = text.replace("\uFE0F", "")   # VS-16 (emoji modifier)
        text = text.replace("\u200B", " ")  # zero-width space -> regular space
        text = text.replace("\uFEFF", "")   # byte-order mark
        
        # Remove only truly problematic non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        text = text.translate(UNICODE_PUNCT_TABLE)
        
        # Fix common concatenated word patterns BEFORE punctuation padding
        common_patterns = [
            (r'\btothe\b', 'to the'),
            (r'\binthe\b', 'in the'), 
            (r'\bonthe\b', 'on the'),
            (r'\bforthe\b', 'for the'),
            (r'\bofthe\b', 'of the'),
            (r'\bfromthe\b', 'from the'),
            (r'\bwiththe\b', 'with the'),
            (r'\batthe\b', 'at the'),
            (r'\bbythe\b', 'by the'),
            (r'\bunderthe\b', 'under the'),
            (r'\boverthe\b', 'over the'),
            (r'\bandthe\b', 'and the'),
            (r'\bofthis\b', 'of this'),
            (r'\bforthis\b', 'for this'),
            (r'\bwiththis\b', 'with this'),
            (r'\binthat\b', 'in that'),
            (r'\bforthat\b', 'for that'),
            (r'\bonthat\b', 'on that'),
            # Keep these as single words
            (r'\btoday\b', 'today'),
            (r'\btomorrow\b', 'tomorrow'),
            (r'\byesterday\b', 'yesterday'),
        ]
        
        for pattern, replacement in common_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        # Preserve capitalization patterns (important for mixed domain)
        words = text.split()
        processed_words = []
        for word in words:
            # Keep all-caps words (often acronyms/entities)
            if word.isupper() and len(word) > 1:
                processed_words.append(word)
            # Keep capitalized words (potential proper nouns)
            elif word and word[0].isupper():
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        text = " ".join(processed_words)
        return text

    # ────────────────────────────────────────────────────────────────────
    # Preprocessing (domain aware)
    # ────────────────────────────────────────────────────────────────────
    def _preprocess(self, text: str) -> str:
        """Return a clean, space‑separated string suitable for tokenisation."""
        if self.domain == "twitter":
            return self._preprocess_twitter(text)
        elif self.domain == "headlines":
            return self._preprocess_headlines(text)
        elif self.domain == "unknown":
            return self._preprocess_unknown(text)
        else:
            return self._preprocess_generic(text)
    # Shared methods for both symbol and word merges
    # ────────────────────────────────────────────────────────────────────
    
    def _get_merge_stats(self, items: List, freq_dict: Dict) -> Dict[Tuple[str, str], int]:
        """Get pair statistics for any type of items (tokens or words)."""
        stats: Dict[Tuple[str, str], int] = defaultdict(int)
        for item, f in freq_dict.items():
            for i in range(len(item) - 1):
                stats[(item[i], item[i + 1])] += f
        return stats
    
    def _build_merge_heap(self, stats: Dict[Tuple[str, str], int]) -> List[Tuple[int, Tuple[str, str]]]:
        """Build a heap from pair statistics."""
        heap = [(-f, p) for p, f in stats.items() if f > 0]
        heapq.heapify(heap)
        return heap
    
    def _update_merge_stats(self, items: List, freq_dict: Dict, merged_pair: Tuple[str, str], 
                           stats: Dict[Tuple[str, str], int]) -> Set[Tuple[str, str]]:
        """Update statistics after a merge operation."""
        affected: Set[Tuple[str, str]] = set()
        
        # Find all pairs that could be affected by this merge
        for item in items:
            for i in range(len(item) - 1):
                pair = (item[i], item[i + 1])
                if merged_pair[0] in pair or merged_pair[1] in pair:
                    affected.add(pair)
        
        affected.discard(merged_pair)
        
        # Recalculate frequencies for affected pairs
        for p in affected:
            stats[p] = 0
        
        for item, f in freq_dict.items():
            for i in range(len(item) - 1):
                p = (item[i], item[i + 1])
                if p in affected:
                    stats[p] += f
        
        return affected

    def extract_complete_words(self, tokenized_words: List[List[str]]) -> List[str]:
        """Extract complete words from tokenized representation."""
        complete_words = []
        for word_tokens in tokenized_words:
            if word_tokens and word_tokens[-1] == END_WORD_MARK:
                # Reconstruct the word by joining all tokens except END_WORD_MARK
                word = "".join(word_tokens[:-1])
                if word and word not in self.special_tokens:
                    complete_words.append(word)
        return complete_words    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────
    def train(self, texts: List[str], word_merge_ratio: float = 0.2) -> None:
        """Learn merges from *texts* belonging to **this** tokenizer's domain.
        
        Args:
            texts: Training texts
            word_merge_ratio: Fraction of merges to dedicate to word-level merging (default 0.2 = 20%)
        """
        if self.merges:
            raise RuntimeError(f"Tokenizer for domain '{self.domain}' is already trained; create a new instance to retrain.")

        # Calculate merge distribution based on word_merge_ratio
        word_merges = int(word_merge_ratio * self.num_merges)
        symbol_merges = self.num_merges - word_merges
        
        print(f"[BPE:{self.domain}] Training with {symbol_merges} symbol merges + {word_merges} word merges (ratio: {word_merge_ratio:.1%})")        # Tokenise + build initial stats
        tokenised_words = self._tokenize_texts(texts)
        self.word_freqs = self._rebuild_freqs(tokenised_words)

        # Protect user‑supplied special tokens from merging away
        for tok in self.special_tokens:
            self.word_freqs[(tok, END_WORD_MARK)] = 10**9

        self.pair_stats = self._get_merge_stats(tokenised_words, self.word_freqs)
        self._heap = self._build_merge_heap(self.pair_stats)

        # Seed character‑level tokens into the vocab
        for ch in {c for w in tokenised_words for c in w if c != END_WORD_MARK}:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        # ────────────────────────────────────────────────────────────────
        # Phase 1: Symbol-level merges (80%) with entity-aware scoring  
        # ────────────────────────────────────────────────────────────────
        merges_done, print_buf = 0, []
        pbar = tqdm(total=self.num_merges, desc=f"BPE:{self.domain} [Phase 1: Symbol merges]", unit="merge")
        
        print(f"[BPE:{self.domain}] Phase 1: Symbol merges ({symbol_merges}/{self.num_merges})")

        while merges_done < symbol_merges and self._heap and False:
            # Get multiple candidate pairs and score them
            candidates = []
            temp_heap = []
            
            # Get up to 10 top candidates, being less strict about stale entries
            for _ in range(min(10, len(self._heap))):
                if not self._heap:
                    break
                freq_neg, pair = heapq.heappop(self._heap)
                freq = -freq_neg
                current_freq = self.pair_stats.get(pair, 0)
                if current_freq > 0:  # Just check if pair still exists with positive frequency
                    candidates.append((pair, current_freq))  # Use current frequency, not stale heap frequency
                temp_heap.append((freq_neg, pair))
            
            # Put non-selected candidates back
            for item in temp_heap:
                if item not in [((-f, p)) for p, f in candidates]:
                    heapq.heappush(self._heap, item)
            
            # If no valid candidates found, rebuild heap and try again
            if not candidates:
                print(f"[BPE:{self.domain}] Rebuilding heap at merge {merges_done} due to stale entries...")
                self.word_freqs = self._rebuild_freqs(tokenised_words)
                self.pair_stats = self._get_merge_stats(tokenised_words, self.word_freqs)
                self._heap = self._build_merge_heap(self.pair_stats)
                if not self._heap:
                    print(f"[BPE:{self.domain}] No more valid pairs after rebuild - stopping at {merges_done} merges")
                    break
                continue
                
            # Score candidates with entity-awareness
            best_pair, best_score, best_freq = None, -1, 0
            for pair, freq in candidates:
                score = self._entity_aware_score(pair, freq)
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    best_freq = freq

            if best_pair is None:
                break

            # 1️⃣ merge everywhere
            tokenised_words = self._merge_pair(best_pair, tokenised_words)
            self.pair_stats.pop(best_pair, None)

            # 2️⃣ update local stats & heap
            changed = self._update_stats(tokenised_words, best_pair)
            for p in changed:
                heapq.heappush(self._heap, (-self.pair_stats[p], p))

            # 3️⃣ bookkeeping
            self.merges.append(best_pair)
            self.ranks[best_pair] = merges_done
            merged_sym = "".join(best_pair)
            if merged_sym not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_sym] = idx
                self.id_to_token[idx] = merged_sym

            print_buf.append((best_pair, best_freq))
            merges_done += 1
            pbar.update(1)            # 5️⃣ safety rebuild every 200 merges (keeps stats fresh & heap small)
            if merges_done % 200 == 0:
                self.word_freqs = self._rebuild_freqs(tokenised_words)
                self.pair_stats = self._get_merge_stats(tokenised_words, self.word_freqs)
                self._heap = self._build_merge_heap(self.pair_stats)

        # ────────────────────────────────────────────────────────────────
        # Phase 2: Whole-word merges (20%) - bigrams
        # ────────────────────────────────────────────────────────────────
        if word_merges > 0:
            print(f"[BPE:{self.domain}] Phase 2: Word bigram merges ({word_merges}/{self.num_merges})")
            # Update progress bar description for Phase 2
            pbar.set_description(f"BPE:{self.domain} [Phase 2: Word merges]")
            
            # Store original texts for whole-word analysis
            original_texts = list(texts) if not isinstance(texts, list) else texts
            print(f"[BPE:{self.domain}] Starting word merge phase with {len(original_texts)} texts...")
            
            # Perform whole-word merges
            merges_done = self._perform_word_merges(
                original_texts, tokenised_words, word_merges, merges_done, pbar
            )

        pbar.close()

        # Final inventory cleanup (symbols that never merged yet appeared)
        for sym in self._extract_vocab_symbols():
            if sym not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[sym] = idx
                self.id_to_token[idx] = sym

        # 🚀 Export learned vocabulary to textfile
        self._export_vocab()    # ────────────────────────────────────────────────────────────────────
    # Public API – encode / decode
    # ────────────────────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string to a list of token ids using proper BPE algorithm.
        """
        if not text.strip():
            return []
            
        # Preprocess the text using domain-specific preprocessing
        text = self._preprocess(text)
        if not text.strip():
            return []
        
        token_ids = []
        
        # Process each word separately  
        for word in text.split():
            if not word:
                continue
                
            # Check if it's a special token
            if word in self.special_tokens:
                token_ids.append(self.token_to_id[word])
                continue
            
            # Convert word to character sequence with end-of-word marker
            word_chars = list(word) + [END_WORD_MARK]
            
            # Apply BPE merges iteratively
            while len(word_chars) > 1:
                # Find the highest priority merge that can be applied
                best_merge = None
                best_rank = float('inf')
                best_pos = -1
                
                for i in range(len(word_chars) - 1):
                    pair = (word_chars[i], word_chars[i + 1])
                    if pair in self.ranks and self.ranks[pair] < best_rank:
                        best_merge = pair
                        best_rank = self.ranks[pair]
                        best_pos = i
                
                # If no merge found, break
                if best_merge is None:
                    break
                
                # Apply the merge
                merged_token = best_merge[0] + best_merge[1]
                word_chars = word_chars[:best_pos] + [merged_token] + word_chars[best_pos + 2:]
            
            # Convert tokens to IDs
            for token in word_chars:
                if token == END_WORD_MARK:
                    continue  # Skip the end marker in final output
                elif token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # For tokens not in vocab, try to break them down
                    for char in token:
                        if char in self.token_to_id:
                            token_ids.append(self.token_to_id[char])
                        else:
                            token_ids.append(self.token_to_id["[UNK]"])
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Reverse of `encode()`. Handles both symbol-level and word-level merged tokens."""
        if not token_ids:
            return ""

        words, current = [], []
        for tid in token_ids:
            tok = self.id_to_token.get(tid)
            if tok == END_WORD_MARK or tok is None:
                continue
            if tok in self.special_tokens:           # skip BOS/EOS/UNK…
                continue
            if tok.endswith(END_WORD_MARK):
                tok_content = tok[:-len(END_WORD_MARK)]
                current.append(tok_content)
                
                # Handle word-level merged tokens (containing underscores)
                reconstructed_word = "".join(current)
                if "_" in reconstructed_word:
                    # This is a word-level merge, replace underscores with spaces
                    reconstructed_word = reconstructed_word.replace("_", " ")
                
                words.append(reconstructed_word)
                current = []
            else:
                current.append(tok)
        if current:                                   # unterminated last word
            reconstructed_word = "".join(current)
            if "_" in reconstructed_word:
                reconstructed_word = reconstructed_word.replace("_", " ")
            words.append(reconstructed_word)
        
        # Join words with proper spacing
        if not words:
            return ""
        
        result = words[0]
        for i in range(1, len(words)):
            word = words[i]
            prev_word = words[i-1]
            
            # Check if current word is punctuation or starts with punctuation
            if word and word[0] in _PUNCT_CHARS:
                # Don't add space before punctuation
                result += word
            # Check if previous word ends with certain punctuation that shouldn't have space after
            elif prev_word and len(prev_word) > 0 and prev_word[-1] in ['(', '[', '{']:
                # Don't add space after opening brackets
                result += word
            else:
                # Add space between words
                result += " " + word
                
        return result

    # ────────────────────────────────────────────────────────────────────
    # Internals (unchanged w.r.t. algorithmic behaviour)
    # ────────────────────────────────────────────────────────────────────
    def _tokenize_texts(self, texts: Iterable[str]) -> List[List[str]]:
        out: List[List[str]] = []
        for txt in texts:
            txt = self._preprocess(txt)
            for raw in txt.split():
                if raw in self.special_tokens:
                    out.append([raw, END_WORD_MARK])
                    continue
                out.append(list(raw) + [END_WORD_MARK])
        return out

    @staticmethod
    def _rebuild_freqs(words: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for w in words:
            freq[tuple(w)] += 1
        return freq
    
    @staticmethod
    def _merge_pair(pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        first, second = pair
        merged = first + second
        for w in words:
            i = 0
            while i < len(w) - 1:
                if w[i] == first and w[i + 1] == second:
                    w[i : i + 2] = [merged]
                    if i > 0:
                        i -= 1
                else:
                    i += 1
        return words

    def _update_stats(self, words: List[List[str]], merged_pair: Tuple[str, str]) -> Set[Tuple[str, str]]:
        """ Update statistics after a merge operation."""
        affected: Set[Tuple[str, str]] = set()
        for w in words:
            prev = None
            for tok in w:
                if prev is not None:
                    affected.add((prev, tok))
                prev = tok
        affected.discard(merged_pair)

        for p in affected:
            self.pair_stats[p] = 0
        for word, f in self.word_freqs.items():
            for i in range(len(word)-1):
                p = (word[i], word[i+1])
                if p in affected:
                    self.pair_stats[p] += f
        return affected

    def _extract_vocab_symbols(self) -> List[str]:
        """Extract all unique symbols from the learned word frequencies."""
        if not self.word_freqs:
            raise RuntimeError("Call this only after `train()` finished.")
        token_freq: Dict[str, int] = defaultdict(int)
        for word, f in self.word_freqs.items():
            for tok in word:
                if tok == END_WORD_MARK:
                    continue
                token_freq[tok] += f
        sorted_symbols = sorted(token_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [sym for sym, _ in sorted_symbols]

    # ────────────────────────────────────────────────────────────────────
    # Helper: export vocabulary to disk
    # ────────────────────────────────────────────────────────────────────
    def _export_vocab(self) -> None:
        """Write the *full* vocabulary (id order) to `self.vocab_out`."""
        try:
            with open(self.vocab_out, "w", encoding="utf-8") as f:
                for idx in range(len(self.id_to_token)):
                    f.write(f"{self.id_to_token[idx]} {idx} \n")
            print(f"[BPE:{self.domain}] Vocabulary exported → {self.vocab_out}")
        except Exception as exc:
            print(f"[BPE:{self.domain}] Warning • failed to write vocab: {exc}")

    def _perform_word_merges(
        self, 
        original_texts: List[str], 
        tokenised_words: List[List[str]], 
        word_merges: int, 
        merges_done: int, 
        pbar
    ) -> int:
        """Perform whole-word bigram merges on the remaining 20% of merges.
        
        This optimized version uses the already-learned symbol-level merges to 
        efficiently tokenize texts, then operates on word-level patterns.
        """
        
        print(f"[BPE:{self.domain}] Word merge phase: processing {len(original_texts)} texts for {word_merges} merges...")
        
        def is_word(token: str) -> bool:
            """Check if a token is a word (contains letters) and not just punctuation."""
            if not token or token in self.special_tokens:
                return False
            # Check if token contains at least one letter
            return any(c.isalpha() for c in token)
        
        def create_word_sequences(words: List[str]) -> List[Tuple[str, str]]:
            """Create bigram sequences from complete words, filtering out punctuation."""
            # Filter to only include actual words (not punctuation)
            word_tokens = [w for w in words if is_word(w)]
            
            sequences = []
            for i in range(len(word_tokens) - 1):
                sequences.append((word_tokens[i], word_tokens[i + 1]))
            return sequences
        
        def reconstruct_words_from_tokens(tokens: List[str]) -> List[str]:
            """Reconstruct complete words from tokenized representation."""
            words = []
            current_word = []
            
            for token in tokens:
                if token.endswith(END_WORD_MARK):
                    # End of word - add accumulated characters plus this token
                    word_content = token[:-len(END_WORD_MARK)]
                    if word_content:
                        current_word.append(word_content)
                    
                    # Join the word and add to words list
                    if current_word:
                        complete_word = ''.join(current_word)
                        if complete_word.strip() and complete_word not in self.special_tokens:
                            words.append(complete_word)
                        current_word = []
                else:
                    # Accumulate characters for current word
                    current_word.append(token)
            
            # Handle any remaining accumulated word
            if current_word:
                complete_word = ''.join(current_word)
                if complete_word.strip() and complete_word not in self.special_tokens:
                    words.append(complete_word)
            
            return words
        
        # Keep track of learned merges for this phase
        learned_word_merges = []
        
        # Use the existing tokenizer state to efficiently encode texts to word level
        print(f"[BPE:{self.domain}] Efficiently tokenizing {len(original_texts)} texts using learned symbol merges...")
        current_processed_texts = []
        
        for text in original_texts:
            # Use the encode method with the current tokenizer state to get tokens
            try:
                # Temporarily disable word merges for encoding
                temp_merges = self.merges.copy()
                temp_ranks = self.ranks.copy()
                
                # Encode using symbol-level merges only
                tokens = []
                for word in self._preprocess(text).split():
                    if not word or word in self.special_tokens:
                        continue
                    
                    # Apply symbol-level BPE to this word
                    word_chars = list(word) + [END_WORD_MARK]
                    
                    # Apply BPE merges iteratively (only symbol-level ones we've learned so far)
                    while len(word_chars) > 1:
                        # Find the highest priority merge that can be applied
                        best_merge = None
                        best_rank = float('inf')
                        best_pos = -1
                        
                        for i in range(len(word_chars) - 1):
                            pair = (word_chars[i], word_chars[i + 1])
                            if pair in self.ranks and self.ranks[pair] < best_rank:
                                best_merge = pair
                                best_rank = self.ranks[pair]
                                best_pos = i
                        
                        # If no merge found, break
                        if best_merge is None:
                            break
                        
                        # Apply the merge
                        merged_token = best_merge[0] + best_merge[1]
                        word_chars = word_chars[:best_pos] + [merged_token] + word_chars[best_pos + 2:]
                    
                    tokens.extend(word_chars)
                
                # Convert tokens back to words
                words = reconstruct_words_from_tokens(tokens)
                current_processed_texts.append(words)
                
            except Exception as e:
                print(f"[BPE:{self.domain}] Warning: Error processing text, falling back to simple split: {e}")
                # Fallback to simple preprocessing
                preprocessed = self._preprocess(text)
                words = preprocessed.split()
                filtered_words = [w for w in words if w not in self.special_tokens and w.strip()]
                current_processed_texts.append(filtered_words)
        
        word_merges_completed = 0
        
        while word_merges_completed < word_merges:
            # Get all word sequences from current processed texts
            all_word_sequences = []
            for word_list in current_processed_texts:
                if len(word_list) > 1:
                    all_word_sequences.extend(create_word_sequences(word_list))
            
            # Convert to frequency dictionary
            word_bigram_freqs = defaultdict(int)
            for bigram in all_word_sequences:
                word_bigram_freqs[bigram] += 1
            
            if len(word_bigram_freqs) == 0:
                print(f"[BPE:{self.domain}] No more word bigrams found - stopping at {word_merges_completed} merges")
                break
            
            # Find the most frequent bigram with entity-aware scoring
            scored_bigrams = []
            for bigram, freq in word_bigram_freqs.items():
                if freq >= 2:  # minimum frequency threshold
                    score = self._score_word_bigram(bigram, freq)
                    scored_bigrams.append((bigram, freq, score))
            
            if not scored_bigrams:
                print(f"[BPE:{self.domain}] No suitable word bigrams found - stopping at {word_merges_completed} merges")
                break
                
            # Select best scoring bigram
            bigram, freq, best_score = max(scored_bigrams, key=lambda x: x[2])
            
            first_word, second_word = bigram
            merged_bigram = f"{first_word}_{second_word}"
            
            # Check if we've already seen this merge (shouldn't happen with proper updates)
            if bigram in learned_word_merges:
                print(f"[BPE:{self.domain}] Duplicate merge detected: {bigram} - skipping")
                break
            
            # Add merged bigram to vocabulary with END_WORD_MARK if not already present
            merged_token = merged_bigram + END_WORD_MARK
            if merged_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_token] = idx
                self.id_to_token[idx] = merged_token
            
            # Record this merge
            self.merges.append(bigram)
            self.ranks[bigram] = merges_done
            learned_word_merges.append(bigram)
            
            # Apply this merge to all processed texts efficiently
            for word_list in current_processed_texts:
                i = 0
                while i < len(word_list) - 1:
                    if word_list[i] == first_word and word_list[i + 1] == second_word:
                        # Replace the pair with the merged version
                        word_list[i:i+2] = [merged_bigram]
                    else:
                        i += 1
            
            word_merges_completed += 1
            merges_done += 1
            if pbar is not None:
                pbar.update(1)
            
            # Print progress
            if word_merges_completed % 10 == 0 or word_merges_completed <= 20:
                print(f"[BPE:{self.domain}] Word bigram merge {word_merges_completed}/{word_merges}: '{first_word}' + '{second_word}' → '{merged_bigram}' (freq: {freq})")
        
        # Store final bigram frequencies for reporting
        final_word_bigram_freqs = defaultdict(int)
        for word_list in current_processed_texts:
            if len(word_list) > 1:
                for bigram in create_word_sequences(word_list):
                    final_word_bigram_freqs[bigram] += 1
        
        self.word_bigram_freqs = dict(final_word_bigram_freqs)
        
        print(f"[BPE:{self.domain}] Completed {word_merges_completed} word merges")
        return merges_done

    def _apply_word_level_merges(self, tokens: List[str], original_words: List[str]) -> List[str]:
        """Apply word-level bigram merges to the tokenized sequence."""
        
        # Simple approach: just return tokens as-is since word merges are already in vocabulary
        # The greedy merge loop in encode() will handle them automatically via the ranks
        return tokens
    def report_word_bigrams(self) -> Tuple[List[Tuple[Tuple[str, str], int]], List[Tuple[Tuple[str, str], int]]]:
        """Return the 5 most and 5 least frequent word bigrams (excluding punctuation-only tokens)."""
        if not self.word_bigram_freqs:
            return [], []
        
        # Filter to only include bigrams where both words contain letters
        def is_word(token: str) -> bool:
            return any(c.isalpha() for c in token) and token not in self.special_tokens
        
        filtered_bigrams = {
            bigram: freq for bigram, freq in self.word_bigram_freqs.items()
            if is_word(bigram[0]) and is_word(bigram[1])
        }
        
        if not filtered_bigrams:
            return [], []
        
        # Sort bigrams by frequency (descending)
        sorted_bigrams = sorted(filtered_bigrams.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 5 and bottom 5
        top5 = sorted_bigrams[:5]
        bottom5 = sorted_bigrams[-5:] if len(sorted_bigrams) > 5 else []
        
        return top5, bottom5

    # ────────────────────────────────────────────────────────────────────
    # Helper: export vocabulary to disk
    # ────────────────────────────────────────────────────────────────────
    def _is_entity_preserving_merge(self, pair: Tuple[str, str]) -> bool:
        """Check if a merge pair is likely to preserve entity boundaries."""
        left, right = pair
        
        # Avoid merging across case boundaries (preserve proper noun starts)
        if (len(left) > 0 and len(right) > 0 and 
            left[-1].islower() and right[0].isupper()):
            return False
            
        # Prefer merging within consistent case patterns
        if (left.isupper() and right.isupper()) or (left.islower() and right.islower()):
            return True
            
        # Avoid splitting common entity patterns
        entity_patterns = [
            ('Mr', '.'), ('Mrs', '.'), ('Dr', '.'), ('St', '.'),
            ('Inc', '.'), ('Corp', '.'), ('Ltd', '.'), ('LLC', '.'),
        ]
        if pair in entity_patterns:
            return True
            
        return True  # Default to allowing merge

    def _get_merge_priority(self, pair: Tuple[str, str], freq: int) -> float:
        """Calculate merge priority considering entity preservation."""
        base_priority = freq
        
        # Boost priority for entity-preserving merges
        if self._is_entity_preserving_merge(pair):
            base_priority *= 1.1
            
        # Boost merges that form common entity suffixes/prefixes
        left, right = pair
        if (left.endswith("'") and right in ['s', 't', 're', 've', 'll', 'd']) or \
           (left in ['@', '#', '<'] or right in ['>', '.com', '.org', '.net']):
            base_priority *= 1.2
            
        return base_priority
    
    def _entity_aware_score(self, pair: Tuple[str, str], freq: int) -> float:
        """Score merge pairs with entity preservation in mind"""
        left, right = pair
        
        # Base score is frequency
        score = float(freq)
        
        # Boost score for pairs that commonly appear in entities
        entity_bonuses = []
        
        # 1. Capitalization patterns (proper nouns)
        if left and right:
            # Both parts are uppercase (acronyms like "U" + "S" -> "US")
            if left.isupper() and right.isupper():
                entity_bonuses.append(1.15)
            
            # Both start with uppercase (proper nouns like "New" + "York")  
            elif (len(left) > 0 and left[0].isupper()) and (len(right) > 0 and right[0].isupper()):
                entity_bonuses.append(1.12)
                
            # Don't break across case boundaries unnecessarily
            elif (len(left) > 0 and len(right) > 0 and 
                  left[-1].islower() and right[0].isupper()):
                entity_bonuses.append(0.9)  # slight penalty
        
        # 2. Special tokens should merge with adjacent content
        if (left in ['<USER>', '<URL>', '<HASHTAG>', '<EMOTICON>', '<EMOJI>'] or 
            right in ['<USER>', '<URL>', '<HASHTAG>', '<EMOTICON>', '<EMOJI>']):
            entity_bonuses.append(1.2)
        
        # 3. Preserve common entity suffixes/prefixes  
        if right in ['ing', 'ed', 'er', 's', 'ly', 'tion', 'sion']:
            entity_bonuses.append(0.95)  # slight penalty to preserve word boundaries
        elif left in ['un', 're', 'pre', 'anti', 'pro']:
            entity_bonuses.append(0.95)  # slight penalty to preserve word boundaries
            
        # 4. Boost for common contractions and punctuation
        if (left.endswith("'") and right in ['s', 't', 're', 've', 'll', 'd']) or \
           (left in ['@', '#', '<'] or right in ['>', '.com', '.org', '.net']):
            entity_bonuses.append(1.25)
            
        # 5. Avoid breaking number patterns
        if left.isdigit() and right.isdigit():
            entity_bonuses.append(1.1)
        
        # Apply all bonuses
        for bonus in entity_bonuses:
            score *= bonus
            
        return score
    
    def _score_word_bigram(self, bigram: Tuple[str, str], freq: int) -> float:
        """Score word bigrams for entity-aware word-level merging."""
        first_word, second_word = bigram
        
        # Base score is frequency
        score = float(freq)
        
        # Entity-related bonuses
        entity_bonuses = []
        
        # 1. Proper noun patterns (both words capitalized)
        if (first_word and first_word[0].isupper() and 
            second_word and second_word[0].isupper()):
            entity_bonuses.append(1.3)  # Strong boost for proper noun pairs
        
        # 2. Title patterns
        title_words = {'mr', 'mrs', 'ms', 'dr', 'prof', 'president', 'senator', 'governor'}
        if first_word.lower() in title_words:
            entity_bonuses.append(1.25)
            
        # 3. Company/Organization suffixes
        org_suffixes = {'inc', 'corp', 'ltd', 'llc', 'company', 'corporation', 'limited'}
        if second_word.lower() in org_suffixes:
            entity_bonuses.append(1.2)
            
        # 4. Location patterns
        location_words = {'new', 'north', 'south', 'east', 'west', 'san', 'los', 'las'}
        if first_word.lower() in location_words:
            entity_bonuses.append(1.15)
            
        # 5. Common entity bigrams
        common_entity_pairs = {
            ('new', 'york'), ('los', 'angeles'), ('san', 'francisco'), 
            ('united', 'states'), ('white', 'house'), ('wall', 'street'),
            ('apple', 'inc'), ('google', 'inc'), ('microsoft', 'corp')
        }
        if (first_word.lower(), second_word.lower()) in common_entity_pairs:
            entity_bonuses.append(1.4)
            
        # 6. Avoid common non-entity patterns
        common_non_entities = {
            ('the', 'and'), ('of', 'the'), ('in', 'the'), ('to', 'the'),
            ('and', 'the'), ('for', 'the'), ('on', 'the'), ('with', 'the')
        }
        if (first_word.lower(), second_word.lower()) in common_non_entities:
            entity_bonuses.append(0.7)  # penalty for non-entity patterns
            
        # Apply all bonuses
        for bonus in entity_bonuses:
            score *= bonus
            
        return score

    def _debug_log(self, message: str, data=None):
        """Log debug information if debugging is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")
            if data is not None:
                print(f"[DEBUG] Data: {data}")

    def _preprocess_text(self, text: str) -> str:
        """
        Basic preprocessing: remove extra whitespace and preserve entity boundaries.
        Also splits common concatenated words before tokenization.
        """
        original_text = text
        
        # Handle common concatenations and contractions
        text = text.replace("tothe", "to the")
        text = text.replace("onthe", "on the")
        text = text.replace("atthe", "at the")
        text = text.replace("inthe", "in the")
        text = text.replace("fromthe", "from the")
        text = text.replace("ofthe", "of the")
        text = text.replace("forthe", "for the")
        text = text.replace("withthe", "with the")
        text = text.replace("bythe", "by the")
        text = text.replace("overthe", "over the")
        text = text.replace("underthe", "under the")
        text = text.replace("throughthe", "through the")
        text = text.replace("betweenthe", "between the")
        text = text.replace("amongthe", "among the")
        text = text.replace("dont", "don't")
        text = text.replace("wont", "won't")
        text = text.replace("cant", "can't")
        text = text.replace("shouldnt", "shouldn't")
        text = text.replace("wouldnt", "wouldn't")
        text = text.replace("couldnt", "couldn't")
        text = text.replace("wasnt", "wasn't")
        text = text.replace("werent", "weren't")
        text = text.replace("isnt", "isn't")
        text = text.replace("arent", "aren't")
        text = text.replace("hasnt", "hasn't")
        text = text.replace("havent", "haven't")
        text = text.replace("hadnt", "hadn't")
        text = text.replace("didnt", "didn't")
        text = text.replace("doesnt", "doesn't")
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        debug = getattr(self, 'debug', False)
        if debug and text != original_text:
            debug_stats = getattr(self, 'debug_stats', {})
            if 'preprocessing_changes' not in debug_stats:
                debug_stats['preprocessing_changes'] = []
            debug_stats['preprocessing_changes'].append({
                'original': original_text,
                'processed': text
            })
            if hasattr(self, '_debug_log'):
                self._debug_log(f"Preprocessing changed text: '{original_text}' -> '{text}'")
        
        return text