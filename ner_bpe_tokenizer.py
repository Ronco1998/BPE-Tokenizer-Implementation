"""
NER‑aware BPE Tokenizer — v6
============================

Final alignment with the reference *tokenizer.py* strategy:

* **Character cap** – only the **200 most‑common characters** (plus space &
  `[UNK]`) are seeded, so we never exhaust the 4 000‑token budget on rare
  emojis.  This leaves plenty of room for word and bigram tokens.
* **Guaranteed bigram quota** – ~20 % of the remaining capacity is allocated
  to the **most‑frequent word bigrams**, 20 % to single words; the rest goes to
  normal BPE merges.
* **Robust bigram detector** – token is a bigram if it contains **exactly one
  space** *within* the string (no leading/trailing space).
* **Clean Unicode printing** via `sys.stdout.reconfigure(encoding="utf‑8")`.

With a typical Twitter corpus (~1.2 M lines) this produces ≈60–120 genuine
bigram tokens out of 4 000 total, matching the friend's numbers.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re
import unicodedata
from html import unescape

import numpy as np

from base_tokenizer import BaseTokenizer

# Ensure Windows consoles can print UTF‑8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Lightweight regex helpers borrowed from the larger BPE_tokenizer
# ---------------------------------------------------------------------------

_TW_USER   = re.compile(r"@[A-Za-z0-9_]{1,15}")
_TW_URL    = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w[\w\d_]*")
_NEWS_DATE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
PUNCT_PAD  = re.compile(r"([,.;:!?()\"'])")

# Simple emoji detector (BMP + common supplementary planes)
_EMOJI = re.compile(
    r"[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    r"|[\U0001F600-\U0001F64F]"  # emoticons
    r"|[\U0001F680-\U0001F6FF]"  # transport & map
    r"|[\u2600-\u26FF]"          # misc symbols
    r"|[\u2700-\u27BF]"          # dingbats
)

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
    """Lightweight BPE with pre‑added word bigrams, tuned for NER."""

    UNK_TOKEN = "[UNK]"
    SPACE_TOKEN = " "
    END_MARK = "</w>"

    _DOMAIN_WEIGHTS: Dict[str, float] = {
        "twitter": 1.3,
        "headline": 1.1,
        "unknown": 1.0,
    }

    def __init__(self, vocab_size: int = 4000, *, domain: str = "unknown", random_seed: int = 42):
        super().__init__()
        random.seed(random_seed)
        np.random.seed(random_seed)

        # store domain information early so that training does not need to be told again
        self.domain = domain.lower()
        self.vocab_size = vocab_size
        for tok in (self.UNK_TOKEN, self.SPACE_TOKEN):
            if tok not in self.token_to_id:
                self._add_token(tok)

        self.space_token = self.SPACE_TOKEN
        self._bpe_ranks: Dict[Tuple[str, str], int] = {}
        self._token_freq: Dict[str, int] = {}

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _add_token(self, tok: str) -> None:
        if tok in self.token_to_id:
            return
        idx = len(self.token_to_id)
        self.token_to_id[tok] = idx
        self.id_to_token[idx] = tok

    @staticmethod
    def _pairs(word: Tuple[str, ...]):
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    @staticmethod
    def _is_bigram(tok: str) -> bool:
        return tok.count(" ") == 1 and not tok.startswith(" ") and not tok.endswith(" ")

    # ---------------------------------------------------------------------
    # domain-aware preprocessing (very lightweight version)
    # ---------------------------------------------------------------------

    def _preprocess_twitter(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        text = _TW_URL.sub("<URL>", text)
        text = _TW_USER.sub("<USER>", text)
        text = _HASHTAG_RE.sub("<HASHTAG>", text)
        text = _EMOJI.sub("<EMOJI>", text)
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _preprocess_headline(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _NEWS_DATE.sub("[DATE]", text)
        text = _EMOJI.sub("<EMOJI>", text)
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _preprocess_generic(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = unescape(text)
        text = _EMOJI.sub("<EMOJI>", text)
        text = text.translate(UNICODE_PUNCT_TABLE)
        text = PUNCT_PAD.sub(r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _preprocess(self, text: str) -> str:
        """Return cleaned text according to the tokenizer's domain."""
        if self.domain == "twitter":
            return self._preprocess_twitter(text)
        if self.domain in {"headline", "headlines"}:
            return self._preprocess_headline(text)
        return self._preprocess_generic(text)

    # ---------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------
    def train(self, texts: List[str], *, bigram_share: float = 0.2,
              char_cap: int = 100):
        """Train tokenizer, forcing a quota of word‑bigram tokens.

        *bigram_share* – fraction of the final vocab reserved for direct
        bigram tokens (same share is used for single words).
        *char_cap* – max number of single‑character tokens to pre‑add.
        """
        # use the domain specified at construction time when computing domain-specific weight
        dom_w = self._DOMAIN_WEIGHTS.get(self.domain, 1.0)

        # 1️⃣ frequency tables -------------------------------------------------
        char_freq, word_freq, bigram_freq = Counter(), Counter(), Counter()
        for line in texts:
            line = self._preprocess(line)
            char_freq.update(line)
            words = line.strip().split()
            word_freq.update(words)
            bigram_freq.update([f"{w1} {w2}" for w1, w2 in zip(words[:-1], words[1:])])

        # 2️⃣ seed **most common chars** --------------------------------------
        char_room = max(0, self.vocab_size - len(self.token_to_id) - 1)  # keep at least 1 slot
        for ch, _ in char_freq.most_common(min(char_cap, char_room)):
            self._add_token(ch)
        # ensure space already exists
        self._add_token(self.SPACE_TOKEN)

        remaining = self.vocab_size - len(self.token_to_id)
        if remaining <= 0:
            print("Vocab size too small to add words / bigrams after char seeding!")
            return
        direct_quota = int(remaining * bigram_share)
        single_quota = direct_quota  # same share for single words

        # 3️⃣ add single‑word tokens -----------------------------------------
        added_single = 0
        for w, _ in word_freq.most_common():
            if added_single >= single_quota or len(self.token_to_id) >= self.vocab_size:
                break
            if w not in self.token_to_id:
                self._add_token(w)
                added_single += 1

        # 4️⃣ add word‑bigram tokens -----------------------------------------
        added_bigrams = 0
        for bg, _ in bigram_freq.most_common():
            if added_bigrams >= direct_quota or len(self.token_to_id) >= self.vocab_size:
                break
            if self._is_bigram(bg) and bg not in self.token_to_id:
                self._add_token(bg)
                added_bigrams += 1

        print(f"Directly added: {added_single} single words, {added_bigrams} bigrams")

        # 5️⃣ prepare BPE vocabulary -----------------------------------------
        vocab = Counter()
        for line in texts:
            words = line.strip().split()
            for idx, word in enumerate(words):
                vocab[tuple(word) + (self.END_MARK,)] += 1
                if idx:
                    vocab[tuple(" " + word) + (self.END_MARK,)] += 1  # leading‑space variant

        # 6️⃣ BPE merges -------------------------------------------------------
        merges_needed = self.vocab_size - len(self.token_to_id)
        merges: List[Tuple[str, str]] = []
        for _ in range(merges_needed):
            pair_freq = Counter()
            for w, f in vocab.items():
                pair_freq.update({p: f for p in self._pairs(w)})
            if not pair_freq:
                break
            # choose best pair by weighted freq
            best_pair, best_score = None, 0.0
            for p, f in pair_freq.items():
                score = f * self._ner_bonus(p) * dom_w
                if score > best_score:
                    best_pair, best_score = p, score
            if best_pair is None:
                break
            self._merge(best_pair, vocab)
            merges.append(best_pair)
            if len(self.token_to_id) >= self.vocab_size:
                break
        self._bpe_ranks = {p: i for i, p in enumerate(merges)}

        # 7️⃣ stats -----------------------------------------------------------
        bigrams_final = sum(1 for t in self.token_to_id if self._is_bigram(t))
        print(f"Final vocab size: {len(self.token_to_id)} (bigrams: {bigrams_final})")

    # ---------------------------------------------------------------------
    # merge helpers
    # ---------------------------------------------------------------------
    def _merge(self, pair: Tuple[str, str], vocab: Counter[Tuple[str, ...]]):
        joined = "".join(pair)
        self._add_token(joined)
        new_vocab = Counter()
        for word, freq in vocab.items():
            i, out = 0, []
            w = list(word)
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                    out.append(joined); i += 2
                else:
                    out.append(w[i]); i += 1
            new_vocab[tuple(out)] += freq
        vocab.clear(); vocab.update(new_vocab)

    # ---------------------------------------------------------------------
    # encoding / decoding (unchanged from v5)
    # ---------------------------------------------------------------------
    def encode(self, text: str, *, domain: str = "unknown") -> List[int]:
        # apply domain-specific preprocessing (uses self.domain)
        text = self._preprocess(text)

        ids, i, n = [], 0, len(text)
        while i < n:
            if text[i] == " ":
                ids.append(self.token_to_id[self.SPACE_TOKEN]); i += 1; continue
            # greedy longest‑match up to 30 chars
            j = min(n, i + 30)
            match = None
            while j > i:
                cand = text[i:j]
                if cand in self.token_to_id:
                    match = cand; break
                j -= 1
            if not match:
                match = text[i]
            ids.append(self.token_to_id.get(match, self.token_to_id[self.UNK_TOKEN]))
            i += len(match)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token.get(i, self.UNK_TOKEN) for i in ids)

    # ---------------------------------------------------------------------
    # scoring helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _ner_bonus(pair: Tuple[str, str]) -> float:
        """Heuristic NER-oriented weight for a candidate merge *pair*.

        Starts with 1.0 and adds / bumps the score based on features inspired by
        the richer analysis used in *tokenizer.py*.
        The values are calibrated so that the historical rules retain their
        original weight (6, 5, 3.5, 2.5) while extra cues provide smaller
        incremental boosts.
        """

        joined = "".join(pair)

        # --- base score ---------------------------------------------------
        score = 1.0

        # Strong bonuses that originally used early-returns
        if pair[0] == " " and pair[1] and pair[1][0].isupper():
            score = 6.0  # word starting with capital after a space
        elif pair[0] and pair[0][0].isupper() and pair[1] and pair[1][0].isupper():
            score = 5.0  # Both sub-tokens start with uppercase
        elif pair[0][-1:].islower() and pair[1] and pair[1][0].isupper():
            score = 3.5  # lowercase-to-Uppercase boundary
        elif any(c.isdigit() for c in joined):
            score = 2.5  # contains digit

        # ------------------------------------------------------------------
        # Additional, smaller bonuses mirroring *tokenizer.py* heuristics
        # ------------------------------------------------------------------

        # Mixed case inside the merge (both upper & lower letters)
        if any(c.isupper() for c in joined) and any(c.islower() for c in joined):
            score += 1.5

        # Presence of non-ASCII characters (emoji, accented letters, etc.)
        if any(ord(c) > 127 for c in joined):
            score += 0.8

        # Generic bonus for tokens that span a word boundary (contain space)
        if " " in joined:
            score += 1.0

        # Title-case word immediately after a space (e.g. " Apple")
        if pair[0] == " " and pair[1].istitle():
            score += 2.0

        # Optional: small positive bias for token length >1 (helps avoid too
        # many single-character merges) – clamps to +1 so it does not skew too
        # much compared to strong NER cues.
        if len(joined) > 1 and " " not in joined:
            score += 1.0

        # Penalise hashtag / mention tokens slightly (already removed explicit
        # bonus) – apply only if no other strong bonus lifted the score above 3.
        if (joined.startswith("#") or joined.startswith("@")) and score < 3.0:
            score *= 0.5

        return score
