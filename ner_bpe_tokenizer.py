from __future__ import annotations
"""
High‑performance **Byte‑Pair Encoding (BPE) tokenizer** tailored for your
assignment.  It inherits from `BaseTokenizer` and now explicitly
prioritises **word‑bigrams** (tokens containing a single space) so that
*even with a small vocabulary* (e.g. 300 tokens) you still get
multi‑word merges like “New York”.

Key points
==========
*   **Fast training** – global pair counting with `collections.Counter`.
*   **Progress prints** – every percent during training.
*   **Bigram boost** – merge‑selection now weights pairs that include the
    ASCII space byte *twice* as high, ensuring some bigrams appear early
    in the limited merge budget.
*   No external imports beyond **standard library**, `regex`, and
    `base_tokenizer`.
"""

from typing import List, Dict, Tuple
from collections import Counter
from time import perf_counter

# ---------------------------------------------------------------------------
# Optional dependency – *regex* is required for GPT‑style Unicode splitting
# ---------------------------------------------------------------------------
try:
    import regex as re  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – clearer error msg
    raise ModuleNotFoundError(
        "Install the 'regex' package to use BPETokenizer (pip install regex)."
    ) from exc

from base_tokenizer import BaseTokenizer  # only allowed external import

# ---------------------------------------------------------------------------
# Tiny helpers (local – no forbidden imports)
# ---------------------------------------------------------------------------

def merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of *pair* in *seq* with *new_id* and return new list."""
    a, b = pair
    out: List[int] = []
    i, L = 0, len(seq)
    while i < L:
        if i < L - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _pair_stats(corpus: List[List[int]]) -> Counter[Tuple[int, int]]:
    """Fast global adjacent‑pair frequency counter (single pass)."""
    stats: Counter[Tuple[int, int]] = Counter()
    for seq in corpus:
        stats.update(zip(seq, seq[1:]))
    return stats

# ---------------------------------------------------------------------------
# GPT‑4 token‑split regex (unchanged)
# ---------------------------------------------------------------------------
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|"
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|"
    r"\s*[\r\n]|"
    r"\s+(?!\S)|"
    r"\s+"
)

# ---------------------------------------------------------------------------
# Tokenizer implementation
# ---------------------------------------------------------------------------

class BPETokenizer(BaseTokenizer):
    """Fast GPT‑style BPE tokenizer with progress prints and bigram bias."""

    def __init__(self, *, vocab_size: int = 10_000, pattern: str | None = None, space_bonus: float = 2.0):
        """Args
        -----
        vocab_size : int
            Total tokens including 256 bytes & 4 special tokens (min 260).
        pattern : str | None
            Optional regex override for GPT‑4 split pattern.
        space_bonus : float
            Weight multiplier for pairs that include **space** – higher
            makes bigrams appear earlier.  Default **2.0**.
        """
        super().__init__()
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("vocab_size must be at least 260 (256 bytes + 4 specials).")

        self.vocab_size = vocab_size
        self._split_re = re.compile(pattern or GPT4_SPLIT_PATTERN)
        self._byte_offset = len(self.special_tokens)  # ids 0‑3 reserved
        self._space_id = self._byte_offset + 32        # ASCII space byte
        self._space_bonus = space_bonus

        # Learnt state (populated by .train())
        self._id2bytes: Dict[int, bytes] = {}
        self._merges: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _score_pair(self, pair: Tuple[int, int], freq: int) -> float:
        """Return weighted score to choose best merge (space gets bonus)."""
        if self._space_id in pair:
            return freq * self._space_bonus
        return float(freq)

    def train(self, texts: List[str]) -> None:  # noqa: D401
        if not texts:
            raise ValueError("'texts' cannot be empty.")

        t0 = perf_counter()
        # 1) Build initial corpus (list of byte‑id sequences)
        corpus: List[List[int]] = [
            [self._byte_offset + b for b in chunk.encode("utf-8")]
            for doc in texts for chunk in self._split_re.findall(doc)
        ]

        # 2) Seed vocabulary with 256 byte tokens
        for b in range(256):
            tid = self._byte_offset + b
            bt = bytes((b,))
            self._id2bytes[tid] = bt
            token_str = bt.decode("latin-1")
            self.token_to_id[token_str] = tid
            self.id_to_token[tid] = token_str

        max_new = self.vocab_size - (self._byte_offset + 256)
        last_pct = -1
        bigram_formed = 0

        for step in range(max_new):
            stats = _pair_stats(corpus)
            if not stats:
                break

            # Choose the pair with maximum *weighted* score
            best_pair = max(stats.items(), key=lambda kv: self._score_pair(kv[0], kv[1]))[0]

            new_id = self._byte_offset + 256 + step
            self._merges[best_pair] = new_id
            self._id2bytes[new_id] = (
                self._id2bytes[best_pair[0]] + self._id2bytes[best_pair[1]]
            )
            token_str = self._id2bytes[new_id].decode("latin-1")
            self.token_to_id[token_str] = new_id
            self.id_to_token[new_id] = token_str

            # track bigrams created
            if self._space_id in best_pair:
                bigram_formed += 1

            corpus = [merge(seq, best_pair, new_id) for seq in corpus]

            pct = int((step + 1) * 100 / max_new)
            if pct != last_pct:
                print(f"[BPE] {pct}% ({step + 1}/{max_new} merges, bigrams={bigram_formed})", flush=True)
                last_pct = pct
        print(f"[BPE] Training complete in {perf_counter() - t0:.2f}s.  Bigrams formed: {bigram_formed}", flush=True)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_bytes(self, raw: bytes) -> List[int]:
        ids = [self._byte_offset + b for b in raw]
        while len(ids) >= 2:
            pair_freqs = Counter(zip(ids, ids[1:]))
            candidates = [p for p in pair_freqs if p in self._merges]
            if not candidates:
                break
            best = min(candidates, key=self._merges.__getitem__)
            ids = merge(ids, best, self._merges[best])
        return ids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:  # noqa: D401
        if not self._merges:
            raise RuntimeError("Tokenizer not trained; call .train().")
        specials_re = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        parts = re.split(specials_re, text)
        out: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue
            for chunk in self._split_re.findall(part):
                out.extend(self._encode_bytes(chunk.encode("utf-8")))
        return out

    def decode(self, token_ids: List[int]) -> str:  # noqa: D401
        if not self._id2bytes:
            raise RuntimeError("Tokenizer not trained; call .train().")
        pieces: List[bytes] = []
        append = pieces.append
        for tid in token_ids:
            if tid in self._id2bytes:
                append(self._id2bytes[tid])
            elif tid in self.id_to_token:
                append(self.id_to_token[tid].encode("utf-8"))
            else:
                append(b"[UNK]")
        return b"".join(pieces).decode("utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Self‑test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    corpus = [
        "New York is great. Los Angeles too!",
        "New York City lights.",
    ]
    tok = BPETokenizer(vocab_size=300)
    tok.train(corpus)

    test = "New York City"
    enc = tok.encode(test)
    print("Encoded:", enc)
    print("Decoded:", tok.decode(enc))
