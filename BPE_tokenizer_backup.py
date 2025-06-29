from __future__ import annotations

import heapq
# Printable ASCII punctuation + '…' (unicode ellipsis) + Unicode quotes  
_PUNCT_CHARS = list(r"""!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + ["…", """, """, "'", "'"]
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
    ) -> None:
        super().__init__()

        self.num_merges   = num_merges
        self.log_every    = log_every
        self.domain       = domain.lower()
        self.vocab_out    = vocab_out_path or f"{self.domain}_vocab.txt"

        # ── 1. Domain‑specific special tokens ─────────────────────────────
        domain_extras: List[str] = []
        if self.domain == "twitter":
            domain_extras = ["<URL>", "<USER>", "<HASHTAG>", "<EMOTICON>"]
        elif self.domain == "news": # add more as needed
            domain_extras = ["[DATE]"]
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
    # Preprocessing (domain aware)
    # ────────────────────────────────────────────────────────────────────
    def _preprocess(self, text: str) -> str:
        """Return a clean, space‑separated string suitable for tokenisation."""
        text = _EMOJI.sub('<EMOJI>', text)
        if self.domain == "twitter":
            text = unicodedata.normalize("NFKC", text)
            text = unescape(text)
            text = _TW_URL.sub("<URL>",  text)
            text = _TW_USER.sub("<USER>", text)
            text = _HASHTAG_RE.sub(_split_hashtag, text)
            text = EMOTICON_RE.sub("<EMOTICON>", text)
            # text = REP_CHARS.sub(r"\1\1", text)  # Commented out to preserve spacing
            text = text.translate(UNICODE_PUNCT_TABLE)
            
            # Filter out problematic Unicode sequences that cause encoding issues
            # Remove non-printable characters and corrupted Unicode
            text = ''.join(char for char in text if char.isprintable() or char.isspace())
            # Remove sequences that look like encoding artifacts
            text = re.sub(r'[^\x00-\x7F\u00A0-\u017F\u2000-\u206F\u2070-\u209F]+', ' ', text)
            
            text = PUNCT_PAD.sub(r" \1 ", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            def smart_case(tok): # preserve ALLCAPS, lowercase others
                return tok if tok.isupper() else tok.lower()

            # text = " ".join(smart_case(tok) for tok in text.split())
            text = " ".join(text.split())
            return text

        if self.domain == "news":
            text = _NEWS_PUN.sub(r" \1 ", text)
            # Replace dates using the new regex constant
            text = _NEWS_DATE.sub('[DATE]', text)
            text = unicodedata.normalize("NFKC", text)
            return text

        # Generic cleanup
        text = text.replace("\uFE0F", "")   # VS-16 (emoji “colour” selector)
        text = text.replace("\u200B", "")   # zero-width space
        text = text.replace("\uFEFF", "")   # byte-order mark

        # Unknown / generic domain – keep original casing
        return text.strip()    # ────────────────────────────────────────────────────────────────────
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
    def train(self, texts: List[str]) -> None:
        """Learn merges from *texts* belonging to **this** tokenizer's domain.
        First performs 80% symbol-level merges, then 20% whole-word merges (bigrams).
        """
        if self.merges:
            raise RuntimeError(f"Tokenizer for domain '{self.domain}' is already trained; create a new instance to retrain.")

        # Calculate merge distribution: 80% symbol merges, 20% word merges
        symbol_merges = int(0.8 * self.num_merges)
        word_merges = self.num_merges - symbol_merges        # Tokenise + build initial stats
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
        # Phase 1: Symbol-level merges (80%)
        # ────────────────────────────────────────────────────────────────
        merges_done, print_buf = 0, []
        pbar = tqdm(total=self.num_merges, desc=f"BPE:{self.domain}", unit="merge")
        
        print(f"[BPE:{self.domain}] Phase 1: Symbol merges ({symbol_merges}/{self.num_merges})")

        while merges_done < symbol_merges and self._heap:
            freq_neg, best_pair = heapq.heappop(self._heap)
            freq = -freq_neg
            if self.pair_stats.get(best_pair, 0) != freq:  # stale entry
                continue

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

            print_buf.append((best_pair, freq))
            merges_done += 1
            pbar.update(1)            # 5️⃣ safety rebuild every 100 merges (keeps stats fresh & heap small)
            if merges_done % 100 == 0:
                self.word_freqs = self._rebuild_freqs(tokenised_words)
                self.pair_stats = self._get_merge_stats(tokenised_words, self.word_freqs)
                self._heap = self._build_merge_heap(self.pair_stats)

        # ────────────────────────────────────────────────────────────────
        # Phase 2: Whole-word merges (20%) - bigrams
        # ────────────────────────────────────────────────────────────────
        if word_merges > 0:
            print(f"[BPE:{self.domain}] Phase 2: Word bigram merges ({word_merges}/{self.num_merges})")
            
            # Store original texts for whole-word analysis
            original_texts = list(texts) if not isinstance(texts, list) else texts
            
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
        """BPE‑encode *text* → list[int]; unknowns map to [UNK] with a warning.
        Applies both symbol-level and word-level merges based on training.
        """
        if not self.merges:
            raise RuntimeError(f"Tokenizer for domain '{self.domain}' has not been trained yet.")

        # 1️⃣ Pre‑tokenise
        preprocessed = self._preprocess(text)
        original_words = preprocessed.split()  # Keep track of original words for word-level merges
        
        tokens: List[str] = []
        for raw in original_words:
            if raw in self.special_tokens:            # placeholder becomes one token
                tokens.append(raw + END_WORD_MARK)
            else:
                tokens.extend(list(raw))
                tokens.append(END_WORD_MARK)

        # 2️⃣ Greedy merge loop for symbol-level merges (uses rank ordering)
        while True:
            best_pair, best_rank, best_idx = None, 1e9, -1
            for i in range(len(tokens) - 1):
                r = self.ranks.get((tokens[i], tokens[i + 1]))
                if r is not None and r < best_rank:
                    best_pair, best_rank, best_idx = (tokens[i], tokens[i + 1]), r, i
            if best_pair is None:
                break
            tokens[best_idx : best_idx + 2] = ["".join(best_pair)]

        # 3️⃣ Apply word-level merges (bigrams) if any exist
        tokens = self._apply_word_level_merges(tokens, original_words)

        # 4️⃣ Map to ids (unknown → [UNK])
        unk_id = self.token_to_id["[UNK]"]
        ids, unknowns = [], []
        for tok in tokens:
            tid = self.token_to_id.get(tok)
            if tid is None:
                unknowns.append(tok)
                tid = unk_id
            ids.append(tid)

        if unknowns:
            print(f"[BPE:{self.domain}] Warning • mapped to [UNK]: {unknowns}")
        return ids

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
        """Perform whole-word bigram merges on the remaining 20% of merges."""
        
        # Extract complete words from tokenized representation
        complete_words = self.extract_complete_words(tokenised_words)
          # Create word sequences (bigrams) from complete words
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
          # Get all word sequences from all texts
        all_word_sequences = []
        for text in original_texts:
            preprocessed = self._preprocess(text)
            words = preprocessed.split()
            # Filter out special tokens and ensure we have actual words
            filtered_words = [w for w in words if w not in self.special_tokens and w.strip()]
            if len(filtered_words) > 1:
                all_word_sequences.extend(create_word_sequences(filtered_words))
        
        # Convert to frequency dictionary
        word_bigram_freqs = defaultdict(int)
        for bigram in all_word_sequences:
            word_bigram_freqs[bigram] += 1
        
        # Store for reporting
        self.word_bigram_freqs = dict(word_bigram_freqs)
        
        # Build heap for word bigrams
        word_bigram_heap = self._build_merge_heap(word_bigram_freqs)
        
        word_merges_completed = 0
        
        while word_merges_completed < word_merges and word_bigram_heap:
            freq_neg, best_bigram = heapq.heappop(word_bigram_heap)
            freq = -freq_neg
            
            # Check if this bigram is still valid (not stale)
            if word_bigram_freqs.get(best_bigram, 0) != freq:
                continue
            
            first_word, second_word = best_bigram
            merged_bigram = f"{first_word}_{second_word}"  # Use underscore to separate
            
            # Add merged bigram to vocabulary with END_WORD_MARK if not already present
            merged_token = merged_bigram + END_WORD_MARK
            if merged_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged_token] = idx
                self.id_to_token[idx] = merged_token
            
            # Record this merge
            self.merges.append(best_bigram)
            self.ranks[best_bigram] = merges_done
            
            # Update bigram statistics
            word_bigram_freqs.pop(best_bigram, None)
            
            # Update frequencies for affected bigrams
            affected_bigrams = set()
            for bigram in list(word_bigram_freqs.keys()):
                if bigram[0] == second_word or bigram[1] == first_word:
                    affected_bigrams.add(bigram)
            
            # Recalculate frequencies for affected bigrams
            for bigram in affected_bigrams:
                new_freq = 0
                for seq in all_word_sequences:
                    if seq == bigram:
                        new_freq += 1
                if new_freq > 0:
                    word_bigram_freqs[bigram] = new_freq
                    heapq.heappush(word_bigram_heap, (-new_freq, bigram))
                else:
                    word_bigram_freqs.pop(bigram, None)
            
            word_merges_completed += 1
            merges_done += 1
            pbar.update(1)
            
            if word_merges_completed % 10 == 0:
                print(f"[BPE:{self.domain}] Word bigram merge {word_merges_completed}/{word_merges}: '{first_word}' + '{second_word}' → '{merged_bigram}' (freq: {freq})")
        
        return merges_done

    def _apply_word_level_merges(self, tokens: List[str], original_words: List[str]) -> List[str]:
        """Apply word-level bigram merges to the tokenized sequence."""
        
        # Find word boundaries in the token sequence
        def find_word_boundaries(tokens: List[str]) -> List[Tuple[int, int]]:
            """Find start and end indices of each word in the token sequence."""
            boundaries = []
            start = 0
            
            for i, token in enumerate(tokens):
                if token.endswith(END_WORD_MARK):
                    boundaries.append((start, i + 1))  # end is exclusive
                    start = i + 1
                    
            return boundaries
        
        # Reconstruct words from token sequence
        def reconstruct_words(tokens: List[str], boundaries: List[Tuple[int, int]]) -> List[str]:
            """Reconstruct complete words from tokenized sequence."""
            words = []
            for start, end in boundaries:
                word_tokens = tokens[start:end]
                if word_tokens and word_tokens[-1].endswith(END_WORD_MARK):
                    # Remove END_WORD_MARK and join tokens
                    word_tokens[-1] = word_tokens[-1][:-len(END_WORD_MARK)]
                    word = "".join(word_tokens)
                    if word and word not in self.special_tokens:
                        words.append(word)
                    else:
                        words.append(word_tokens[0] if word_tokens else "")
                else:
                    words.append("".join(word_tokens))
            return words
        
        # Apply word-level merges iteratively
        changed = True
        max_iterations = len(self.merges)  # Prevent infinite loops
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            boundaries = find_word_boundaries(tokens)
            words = reconstruct_words(tokens, boundaries)
            
            # Look for word bigrams that can be merged
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                
                # Check if this bigram has a learned merge (should be in the latter part of merges)
                if bigram in self.ranks:
                    # Found a word-level merge
                    merged_word = f"{words[i]}_{words[i + 1]}"
                    
                    # Check if merged word is in vocabulary
                    if merged_word in self.token_to_id:
                        # Apply the merge by replacing the two word tokens with the merged version
                        start1, end1 = boundaries[i]
                        start2, end2 = boundaries[i + 1]
                          # Replace the token sequence for these two words with merged word + END_WORD_MARK
                        new_tokens = (tokens[:start1] + 
                                    [merged_word + END_WORD_MARK] + 
                                    tokens[end2:])
                        
                        tokens = new_tokens
                        changed = True
                        break  # Restart the search after making a change
        
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
