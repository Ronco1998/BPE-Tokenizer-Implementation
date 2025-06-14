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
_NEWS_PUN = re.compile(r"([,.;:!?()\"'])")    # punctuation splitter
_NEWS_DATE = re.compile(r'\b\d{4}-\d{2}-\d{2}\b') # dates like 2023-10-05
REP_CHARS = re.compile(r"(.)\1{2,}")
PUNCT_PAD = re.compile(r"([,.;:!?()\"'])")
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
        log_every: int = 0,
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
        self.ranks:  Dict[Tuple[str, str], int] = {}     # quick look‑up → rank

        # Working state (used only during `train()`)
        self.word_freqs: Dict[Tuple[str, ...], int] = {}
        self.pair_stats: Dict[Tuple[str, str], int] = {}
        self._heap: List[Tuple[int, Tuple[str, str]]] = []

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
            text = REP_CHARS.sub(r"\1\1", text)
            text = text.translate(UNICODE_PUNCT_TABLE)
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
        return text.strip()

    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────
    def train(self, texts: Iterable[str]) -> None:
        """Learn merges from *texts* belonging to **this** tokenizer's domain."""
        if self.merges:
            raise RuntimeError(f"Tokenizer for domain '{self.domain}' is already trained; create a new instance to retrain.")

        # Tokenise + build initial stats
        tokenised_words = self._tokenize_texts(texts)
        self.word_freqs = self._rebuild_freqs(tokenised_words)

        # Protect user‑supplied special tokens from merging away
        for tok in self.special_tokens:
            self.word_freqs[(tok, END_WORD_MARK)] = 10**9

        self.pair_stats = self._get_stats(tokenised_words, self.word_freqs)
        self._rebuild_heap()

        # Seed character‑level tokens into the vocab
        for ch in {c for w in tokenised_words for c in w if c != END_WORD_MARK}:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        merges_done, print_buf = 0, []
        pbar = tqdm(total=self.num_merges, desc=f"BPE:{self.domain}", unit="merge")

        while merges_done < self.num_merges and self._heap:
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
            pbar.update(1)

            # 4️⃣ periodic print to terminal
            if merges_done % self.log_every == 0 or merges_done == self.num_merges:
                start = merges_done - len(print_buf) + 1
                tqdm.write(f"[BPE:{self.domain}] merges {start}–{merges_done} →")
                for i, (pair, f) in enumerate(print_buf, 1):
                    print(f"  {i}. {pair} (freq {f})")
                print_buf = []

            # 5️⃣ safety rebuild every 100 merges (keeps stats fresh & heap small)
            if merges_done % 100 == 0:
                self.word_freqs = self._rebuild_freqs(tokenised_words)
                self.pair_stats = self._get_stats(tokenised_words, self.word_freqs)
                self._rebuild_heap()

        pbar.close()

        # Final inventory cleanup (symbols that never merged yet appeared)
        for sym in self._extract_vocab_symbols():
            if sym not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[sym] = idx
                self.id_to_token[idx] = sym

        # 🚀 Export learned vocabulary to textfile
        self._export_vocab()

    # ────────────────────────────────────────────────────────────────────
    # Public API – encode / decode
    # ────────────────────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """BPE‑encode *text* → list[int]; unknowns map to [UNK] with a warning."""
        if not self.merges:
            raise RuntimeError(f"Tokenizer for domain '{self.domain}' has not been trained yet.")

        # 1️⃣ Pre‑tokenise
        preprocessed = self._preprocess(text)
        tokens: List[str] = []
        for raw in preprocessed.split():
            if raw in self.special_tokens:            # placeholder becomes one token
                tokens.append(raw + END_WORD_MARK)
            else:
                tokens.extend(list(raw))
                tokens.append(END_WORD_MARK)

        # 2️⃣ Greedy merge loop (uses rank ordering)
        while True:
            best_pair, best_rank, best_idx = None, 1e9, -1
            for i in range(len(tokens) - 1):
                r = self.ranks.get((tokens[i], tokens[i + 1]))
                if r is not None and r < best_rank:
                    best_pair, best_rank, best_idx = (tokens[i], tokens[i + 1]), r, i
            if best_pair is None:
                break
            tokens[best_idx : best_idx + 2] = ["".join(best_pair)]

        # 3️⃣ Map to ids (unknown → [UNK])
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
        """Reverse of `encode()`.  Raises on truly unknown ids."""
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
                current.append(tok[:-len(END_WORD_MARK)])
                words.append("".join(current))
                current = []
            else:
                current.append(tok)
        if current:                                   # unterminated last word
            words.append("".join(current))
        return " ".join(words)

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
    def _get_stats(
        words: List[List[str]],
        word_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, str], int]:
        stats: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, f in word_freqs.items():
            for i in range(len(word)-1):
                stats[(word[i], word[i+1])] += f
        return stats

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

    def _rebuild_heap(self) -> None:
        self._heap = [(-f, p) for p, f in self.pair_stats.items() if f > 0]
        heapq.heapify(self._heap)

    def _extract_vocab_symbols(self) -> List[str]:
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
                    f.write(self.id_to_token[idx] + "\n")
            print(f"[BPE:{self.domain}] Vocabulary exported → {self.vocab_out}")
        except Exception as exc:
            print(f"[BPE:{self.domain}] Warning • failed to write vocab: {exc}")
