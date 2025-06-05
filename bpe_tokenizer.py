
from __future__ import annotations

import heapq, re, logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Iterable, Optional

from tqdm.auto import tqdm  # nice progress bar in notebooks & terminals


# ────────────────────────────────────────────────────────────────────────────
# Helper regexes for domain‑specific preprocessing
# ────────────────────────────────────────────────────────────────────────────
_TW_USER  = re.compile(r"@[A-Za-z0-9_]{1,15}")  # Twitter @handle
_TW_URL   = re.compile(r"https?://\S+")         # http(s) URLs
_TW_HASH  = re.compile(r"#[\w\d_]+")           # #hashtag
_NEWS_PUN = re.compile(r"([,.;:!?()\"'])")     # punct splitter


def _preprocess(text: str, domain: str) -> str:
    """Minimal clean‑up that returns a space‑separated string."""
    if domain == "twitter":
        text = text.lower()
        text = _TW_USER.sub("<USER>", text)
        text = _TW_URL.sub("<URL>",  text)
        text = _TW_HASH.sub(lambda m: m.group(0).lower(), text)
        return text

    if domain == "news":
        text = _NEWS_PUN.sub(r" \1 ", text)
        return text.lower()

    return text.strip().lower()


# ────────────────────────────────────────────────────────────────────────────
# BPETokenizer class
# ────────────────────────────────────────────────────────────────────────────
class BPETokenizer:
    """Byte‑Pair‑Encoding tokenizer with optional per‑domain special tokens."""

    def __init__(
        self,
        vocab_size: int = 30_000,
        log_every: int = 20,
        domain: str = "generic",
        special_tokens: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.vocab_size   = vocab_size
        self.log_every    = log_every
        self.domain       = domain
        self.special_toks = special_tokens or []

        # basic logger (prints to stdout if none provided)
        if logger is None:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logger = logging.getLogger(f"BPE.{domain}")
        self.logger = logger

        # learned artefacts *per domain*
        self.merges_by_domain: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.ranks_by_domain:  Dict[str, Dict[Tuple[str, str], int]] = defaultdict(dict)

        # working state during *current* training session
        self.word_freqs: Dict[Tuple[str, ...], int] = {}
        self.pair_stats: Dict[Tuple[str, str], int] = {}
        self._heap: List[Tuple[int, Tuple[str, str]]] = []

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────
    def train(self, texts: Iterable[str]) -> None:
        """Learn merges from *texts* belonging to `self.domain`."""
        domain = self.domain
        words  = self._tokenize_texts(texts, domain)
        self.word_freqs  = self._rebuild_freqs(words)

        # lock‑in special tokens
        for tok in self.special_toks:
            self.word_freqs[(tok, "</w>")] = 10**9  # massive count so they never merge away

        self.pair_stats  = self._get_stats(words, self.word_freqs)
        self._rebuild_heap()

        merges_done, print_buf = 0, []
        pbar = tqdm(total=self.vocab_size, desc=f"BPE:{domain}", unit="merge")

        while merges_done < self.vocab_size and self._heap:
            freq_neg, best_pair = heapq.heappop(self._heap)
            freq = -freq_neg
            if self.pair_stats.get(best_pair, 0) != freq:  # stale entry
                continue

            # 1️⃣ merge everywhere
            words = self._merge_pair(best_pair, words)
            self.pair_stats.pop(best_pair, None)

            # 2️⃣ update local stats & heap
            changed = self._update_stats(words, best_pair)
            for p in changed:
                heapq.heappush(self._heap, (-self.pair_stats[p], p))

            # 3️⃣ bookkeeping
            self.merges_by_domain[domain].append(best_pair)
            self.ranks_by_domain[domain][best_pair] = merges_done
            print_buf.append((best_pair, freq))
            merges_done += 1
            pbar.update(1)

            # 4️⃣ periodic log + print
            if merges_done % self.log_every == 0 or merges_done == self.vocab_size:
                start = merges_done - len(print_buf) + 1
                self.logger.info(f"[BPE:{domain}] merges {start}–{merges_done} →")
                tqdm.write(f"[BPE:{domain}] merges {start}–{merges_done} →")
                for i, (pair, f) in enumerate(print_buf, 1):
                    self.logger.info(f"  {i}. {pair} (freq {f})")
                print_buf = []

            # 5️⃣ safety rebuild every 1000 merges
            if merges_done % 1000 == 0:
                self.word_freqs = self._rebuild_freqs(words)
                self.pair_stats = self._get_stats(words, self.word_freqs)
                self._rebuild_heap()

        pbar.close()

    # Convenience encode (uses merges of current domain)
    def encode(self, word: str) -> List[str]:
        merges = self.merges_by_domain.get(self.domain)
        if not merges:
            raise RuntimeError("Tokenizer not trained for this domain.")
        tokens = list(word) + ["</w>"]
        merge_rank = {m: i for i, m in enumerate(merges)}
        while True:
            best = None
            min_rank = 1e9
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                r = merge_rank.get(pair)
                if r is not None and r < min_rank:
                    best, min_rank, idx = pair, r, i
            if best is None:
                break
            tokens[idx:idx+2] = [best[0]+best[1]]
        if tokens[-1] == "</w>":
            tokens.pop()
        return tokens

    # ────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize_texts(texts: Iterable[str], domain: str) -> List[List[str]]:
        out: List[List[str]] = []
        for txt in texts:
            txt = _preprocess(txt, domain)
            for raw in txt.split():
                out.append(list(raw) + ["</w>"])
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
                if w[i] == first and w[i+1] == second:
                    w[i:i+2] = [merged]
                i += 1
        return words

    def _update_stats(
        self,
        words: List[List[str]],
        merged_pair: Tuple[str, str],
    ) -> Set[Tuple[str, str]]:
        affected: Set[Tuple[str, str]] = set()
        merged_tok = merged_pair[0] + merged_pair[1]
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
