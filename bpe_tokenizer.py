
from __future__ import annotations

import heapq, re, logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Iterable, Optional
from base_tokenizer import BaseTokenizer
from tqdm.auto import tqdm


# ────────────────────────────────────────────────────────────────────────────
# Helper regexes for domain‑specific preprocessing
# ────────────────────────────────────────────────────────────────────────────
_TW_USER  = re.compile(r"@[A-Za-z0-9_]{1,15}")  # Twitter @handle
_TW_URL   = re.compile(r"https?://\S+")         # http(s) URLs
_TW_HASH  = re.compile(r"#[\w\d_]+")           # #hashtag
_NEWS_PUN = re.compile(r"([,.;:!?()\"'])")     # punct splitter
_REGEX_CACHE: Dict[Tuple[str, str], re.Pattern] = {}




# ────────────────────────────────────────────────────────────────────────────
# BPETokenizer class
# ────────────────────────────────────────────────────────────────────────────
class BPETokenizer(BaseTokenizer):
    """Byte‑Pair‑Encoding tokenizer with optional per‑domain special tokens."""

    def __init__(
        self,
        vocab_size: int = 5_000,
        log_every: int = 20,
        domain: str = "generic",
        special_tokens: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
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
    
    
    def _preprocess(self, text: str, domain: str) -> str:
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

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────
    def train(self, texts: Iterable[str]) -> None:
        """Learn merges from *texts* belonging to `self.domain`."""
        domain = self.domain
        words  = self._tokenize_texts(texts, domain)
        self.word_freqs  = self._rebuild_freqs(words)
        if self.merges_by_domain[self.domain]:
            raise RuntimeError(
                f"BPETokenizer for domain '{self.domain}' is already trained; "
                "create a new instance if you need to retrain.")

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

        for sym in self._extract_vocab_symbols():
            if sym not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[sym] = idx
                self.id_to_token[idx] = sym

    # Convenience encode (uses merges of current domain)
    # ─── Public API ────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """
        BPE-encode *text* → list[int] where each int is a vocabulary id.
        Raises:
            RuntimeError – if tokenizer not trained for this domain
            ValueError   – if a produced sub-token is missing from the vocab
        """
        merges = self.merges_by_domain.get(self.domain)
        if not merges:
            raise RuntimeError(
                f"Tokenizer for domain '{self.domain}' has not been trained yet."
            )

        # 1️⃣ run BPE merges (identical logic, just broken out into helper here)
        tokens: List[str] = list(self._preprocess(text, self.domain)) + ["</w>"]
        merge_rank = self.ranks_by_domain[self.domain]
        while True:
            best_pair, best_rank, best_idx = None, 1e9, -1
            for i in range(len(tokens) - 1):
                r = merge_rank.get((tokens[i], tokens[i + 1]))
                if r is not None and r < best_rank:
                    best_pair, best_rank, best_idx = (tokens[i], tokens[i + 1]), r, i
            if best_pair is None:
                break
            tokens[best_idx : best_idx + 2] = ["".join(best_pair)]

        if tokens and tokens[-1] == "</w>":  # drop sentence end marker
            tokens.pop()

        # 2️⃣ map to ids – failure here is *very* informative
        ids: List[int] = []
        for tok in tokens:
            try:
                ids.append(self.token_to_id[tok])
            except KeyError as e:
                raise ValueError(
                    f"Sub-token '{tok}' is not in vocabulary "
                    f"(domain='{self.domain}'). Did you forget to train / load?"
                ) from e
        return ids


    def decode(self, token_ids: List[int]) -> str:
        """
        Reverse of `encode`.
        Raises:
            ValueError – if an id is unknown
        """
        if not token_ids:
            return ""

        words, current = [], []
        for i in token_ids:
            tok = self.id_to_token.get(i)
            if tok is None:
                raise ValueError(f"Unknown token-id {i} (vocab size={len(self.id_to_token)})")
            if tok in self.special_tokens:
                continue                                # skip [PAD]/[BOS]/[EOS]/[UNK]
            if tok.endswith("</w>"):
                current.append(tok[:-4])                # strip suffix
                words.append("".join(current))
                current = []
            else:
                current.append(tok)

        if current:                                    # safety – unterminated last word
            words.append("".join(current))
        return " ".join(words)


    # ────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────
    def _tokenize_texts(self, texts: Iterable[str], domain: str) -> List[List[str]]:
        out: List[List[str]] = []
        for txt in texts:
            txt = self._preprocess(txt, domain)
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



    # … and overwrite the existing _merge_pair:
    @staticmethod
    def _merge_pair(pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        first, second = pair
        merged = first + second

        # 1️⃣ compile / reuse regex  (word chars separated by single spaces)
        pat = _REGEX_CACHE.get(pair)
        if pat is None:
            pat = re.compile(
                rf"(?:(?<=\s)|^){re.escape(first)} {re.escape(second)}(?=\s)"
            )
            _REGEX_CACHE[pair] = pat

        # 2️⃣ flatten → replace → rebuild
        joined = " ".join(" ".join(w) for w in words)
        joined = pat.sub(merged, joined)

        # 3️⃣ split back into nested list structure
        flat = joined.split(" ")
        new_words, cur = [], []
        for tok in flat:
            cur.append(tok)
            if tok == "</w>":          # word boundary marker
                new_words.append(cur)
                cur = []
        return new_words
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

        # ------------------------------------------------------------------
    # Build final symbol inventory after training
    # ------------------------------------------------------------------
    def _extract_vocab_symbols(self) -> List[str]:
        """
        Return *all* sub-word symbols that survived the BPE merges,
        sorted by total frequency (high → low).

        It walks over the final `self.word_freqs` dictionary, which
        should already reflect the last merge state (remember we
        rebuild it every 1000 merges and once more at the very end).
        """
        if not self.word_freqs:
            raise RuntimeError("Call this only after `train()` finished.")

        # 1️⃣ accumulate token frequencies
        token_freq: Dict[str, int] = defaultdict(int)
        for word, f in self.word_freqs.items():              # :contentReference[oaicite:0]{index=0}
            for tok in word:
                if tok == "</w>":            # sentence-ending marker ≠ real vocab
                    continue
                token_freq[tok] += f

        # 2️⃣ sort by frequency (desc), then lexicographically (stable)
        sorted_symbols = sorted(
            token_freq.items(),
            key=lambda kv: (-kv[1], kv[0])
        )
        return [sym for sym, _ in sorted_symbols]

