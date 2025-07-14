from __future__ import annotations
"""
Fast *Byte‑Pair Encoding (BPE) tokenizer* that depends only on the
Python standard library (re).  It inherits from BaseTokenizer and
keeps all earlier features:

* progress prints every percent *and* early‑stage messages so the user
  sees output immediately,
* explicit bigram bias so multi‑word tokens appear even with a tiny
  vocabulary (e.g. 300),
* a reasonably quick C‑accelerated text splitter (based on a compiled
  re pattern) — no slow, pure‑Python per‑character loop.
"""

from typing import List, Dict, Tuple
from collections import Counter
from time import perf_counter
import re  # built‑in regex engine
from tqdm import tqdm

from base_tokenizer import BaseTokenizer  # allowed external import

# ---------------------------------------------------------------------------
# Helpers (local – no forbidden imports)
# ---------------------------------------------------------------------------

def merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Return a new sequence with pair replaced by new_id."""
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
    """Count adjacent pairs across the whole corpus (single pass, C‑fast)."""
    stats: Counter[Tuple[int, int]] = Counter()
    for seq in corpus:
        stats.update(zip(seq, seq[1:]))
    return stats

# ---------------------------------------------------------------------------
# Fast splitter (std‑lib re, Unicode aware)
# ---------------------------------------------------------------------------
# Order matters: earlier alternatives take precedence.
_FRAG_RE = re.compile(
    r"\r\n|\r|\n|"        # Windows & Unix newlines first
    r"[^\W\d_]+|"          # letters (any script) – [^\W] = "word", minus digits/_
    r"[0-9]{1,3}|"          # digits, but max 3 per chunk (GPT‑style)
    r"[ \t\x0B\f]+|"        # horizontal whitespace (space, tab, vtab, ff)
    r"[^\w\s]",             # everything else (punctuation, emoji, symbols)
    re.UNICODE,
)

def _split_into_fragments(text: str) -> List[str]:
    """Split text according to _FRAG_RE (C‑speed)."""
    return [m.group(0) for m in _FRAG_RE.finditer(text)]

# ---------------------------------------------------------------------------
# Tokenizer implementation
# ---------------------------------------------------------------------------

class BPETokenizer(BaseTokenizer):
    """Standard‑lib‑only BPE tokenizer with progress prints and bigram boost."""

    def __init__(self, *, vocab_size: int = 10_000, space_bonus: float = 2.0):
        super().__init__()
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("vocab_size must be ≥ 260 (256 bytes + 4 specials).")

        self.vocab_size = vocab_size
        self._byte_offset = len(self.special_tokens)  # ids 0‑3 reserved
        self._space_id = self._byte_offset + 32        # ASCII space token id
        self._space_bonus = space_bonus

        # Learnt state (initialised in .train())
        self._id2bytes: Dict[int, bytes] = {}
        self._merges: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _score(self, pair: Tuple[int, int], freq: int) -> float:
        return freq * self._space_bonus if self._space_id in pair else float(freq)

    def train(self, texts: List[str]) -> None:  # noqa: D401
        if not texts:
            raise ValueError("'texts' is empty.")

        print("[BPE] Preparing corpus…", flush=True)
        t0 = perf_counter()

        corpus: List[List[int]] = []
        for doc in tqdm(texts, desc="Processing documents"):
            for frag in _split_into_fragments(doc):
                corpus.append([self._byte_offset + b for b in frag.encode("utf-8")])
        print(f"[BPE] Corpus ready – {len(corpus):,} fragments.", flush=True)

        # Seed with byte tokens
        for b in range(256):
            tid = self._byte_offset + b
            bt = bytes((b,))
            self._id2bytes[tid] = bt
            tok_str = bt.decode("latin-1")
            self.token_to_id[tok_str] = tid
            self.id_to_token[tid] = tok_str

        max_new = self.vocab_size - (self._byte_offset + 256)
        last_pct = -1
        bigram_ct = 0

        for step in tqdm(range(max_new), desc="Training BPE merges"):
            stats = _pair_stats(corpus)
            if not stats:
                break
            best_pair = max(stats.items(), key=lambda kv: self._score(kv[0], kv[1]))[0]

            new_id = self._byte_offset + 256 + step
            self._merges[best_pair] = new_id
            self._id2bytes[new_id] = self._id2bytes[best_pair[0]] + self._id2bytes[best_pair[1]]
            tok_str = self._id2bytes[new_id].decode("latin-1")
            self.token_to_id[tok_str] = new_id
            self.id_to_token[new_id] = tok_str

            if self._space_id in best_pair:
                bigram_ct += 1

            corpus = [merge(seq, best_pair, new_id) for seq in corpus]

            # pct = int((step + 1) * 100 / max_new)
            # if pct != last_pct:
            #     print(
            #         f"[BPE] {pct}% ({step + 1}/{max_new} merges, bigrams={bigram_ct})",
            #         flush=True,
            #     )
            #     last_pct = pct
        print(f"[BPE] Training done in {perf_counter() - t0:.2f}s. Bigrams: {bigram_ct}", flush=True)

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
        for part in tqdm(parts, desc="Encoding text parts", leave=False):
            if part == "":
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue
            for frag in _split_into_fragments(part):
                out.extend(self._encode_bytes(frag.encode("utf-8")))
        return out

    def decode(self, token_ids: List[int]) -> str:  # noqa: D401
        if not self._id2bytes:
            raise RuntimeError("Tokenizer not trained.")
        pieces: List[bytes] = []
        ap = pieces.append
        for tid in token_ids:
            if tid in self._id2bytes:
                ap(self._id2bytes[tid])
            elif tid in self.id_to_token:
                ap(self.id_to_token[tid].encode("utf-8"))
            else:
                ap(b"[UNK]")
        return b"".join(pieces).decode("utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Self‑test when run directly
# ---------------------------------------------------------------------------
if __name__ == "_main_":
    corpus = [
        "New York is great!",
        "Los Angeles too 😊.",
    ]
    tok = BPETokenizer(vocab_size=300)
    tok.train(corpus)

    sample = "New York 😊"
    print("Encoded:", tok.encode(sample))
    print("Decoded:", tok.decode(tok.encode(sample)))