from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re
from base_tokenizer import BaseTokenizer
import unicodedata
from tqdm import tqdm
import heapq

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000, domain: str = ""):
        """
        Initialize the BPE tokenizer

        Args:
            vocab_size: Maximum size of the vocabulary
        """
        super().__init__()
        self.space_token = "_"
        self.vocab_size = vocab_size
        self.merges = {}  # Store the learned BPE merges
        self.word_frequencies = Counter()  # Store word frequencies during training
        self.domain = domain
        self.pair_stats = {}  # New attribute to store pair statistics
        self._heap: list[tuple[int, tuple[str, str]]] = []

        # Add Twitter-specific special tokens if domain is twitter
        if self.domain == 'twitter':
            special_token_list = ['HASHTAG_TOKEN', 'MENTION_TOKEN', 'URL_TOKEN', 'EMOJI_TOKEN']
            for idx, token in enumerate(special_token_list, start=-10):
                if token not in self.special_tokens:
                    self.special_tokens[token] = idx
        elif self.domain == 'news':
            self.preserve_case = True
        elif self.domain == 'unknown':
            # Robust fallback: canonicalise visually identical chars
            self._normalize = lambda s: unicodedata.normalize('NFKC', s)
        else:
            raise ValueError(f"Unsupported domain {self.domain}")


    def _get_stats(self, words: List[List[str]], freqs: Counter) -> Dict[Tuple[str, str], int]:
        """
        Count bigrams, weighted by word frequency.

        Args:
            words: List of words (as lists of tokens) to count pairs from.
            freqs: Word frequencies as a Counter object.

        Returns:
            Dictionary of pairs and their weighted frequencies.
        """
        pair_counts = defaultdict(int)
        for tokens in words:  # tokens is a *list*, no split()
            w_freq = freqs[tuple(tokens)]
            for a, b in zip(tokens, tokens[1:]):
                pair_counts[(a, b)] += w_freq
        return pair_counts

    def _update_stats(self, words: List[List[str]], last_pair: Tuple[str, str]) -> Tuple[Counter, Counter]:
        """
        Update only pairs adjacent to the merged token in the saved pair statistics.
        Subtract counts for pairs that vanish, so fewer full rebuilds are needed.

        Args:
            words: List of words (as lists of tokens) after the merge
            last_pair: The last merged pair (e.g. ('a', 'b'))

        This method updates self.pair_stats in-place. It increments counts for new pairs
        and subtracts counts for pairs that vanish due to the merge.
        """
        merged_token = last_pair[0] + last_pair[1]
        # Track pairs that vanish due to the merge
        vanished_pairs = Counter()
        appeared_pairs = Counter()

        for word in words:
            prev = None
            for idx, token in enumerate(word):
                if token == merged_token:
                    # The merged token replaces last_pair, so last_pair vanishes
                    vanished_pairs[last_pair] += 1
                    # New pairs may appear to the left and right
                    if idx - 1 >= 0:
                        left_pair = (word[idx - 1], merged_token)
                        appeared_pairs[left_pair] += 1
                    if idx + 1 < len(word):
                        right_pair = (merged_token, word[idx + 1])
                        appeared_pairs[right_pair] += 1
                # Track pairs that vanish due to the merge
                if prev is not None:
                    pair = (prev, token)
                    if pair == last_pair:
                        vanished_pairs[pair] += 1
                prev = token
        return vanished_pairs, appeared_pairs

        # Subtract vanished pairs
        for pair, count in vanished_pairs.items():
            if pair in self.pair_stats:
                self.pair_stats[pair] -= count
                if self.pair_stats[pair] <= 0:
                    del self.pair_stats[pair]
        # Add appeared pairs
        for pair, count in appeared_pairs.items():
            self.pair_stats[pair] = self.pair_stats.get(pair, 0) + count

    def _merge_pair(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        """
        Merge pair in-place and return the modified list-of-lists.

        Args:
            pair: The pair to merge.
            words: List of words (as lists of tokens) to merge the pair in.

        Returns:
            List of words (as lists of tokens) with the pair merged.
        """
        p0, p1 = pair
        repl = p0 + p1
        new_words = []
        for tokens in words:
            i = 0
            out = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    out.append(repl)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            new_words.append(out)
        return new_words
    
    def _preprocess_twitter(self, text: str) -> str:
        """
        Preprocess Twitter text: normalize, lowercase, separate hashtags, mentions, URLs, emojis, and punctuation as tokens.
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Replace URLs with URL_TOKEN
        text = re.sub(r'https?://\S+|www\.\S+', ' URL_TOKEN ', text)
        # Replace mentions with MENTION_TOKEN
        text = re.sub(r'@[\w_]+', ' MENTION_TOKEN ', text)
        # Replace hashtags with HASHTAG_TOKEN
        text = re.sub(r'#[\w_]+', ' HASHTAG_TOKEN ', text)
        # Replace emojis with EMOJI_TOKEN (simple unicode emoji range)
        emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' EMOJI_TOKEN ', text)
        # Separate punctuation as in news
        text = re.sub(r'([,.;:!?()\[\]«»"“”‘’])', r' \1 ', text)
        # Collapse runs of whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        # No need to fix case for special tokens anymore
        return text

    def _preprocess_news(self, text: str) -> str:
        """
        Minimal, NER-friendly normalisation for news headlines.
        - keeps original case (capitals often mark named entities)
        - separates headline punctuation ,.:;!?()[]«»"“”‘’ so each becomes its own token
        - collapses weird unicode quotes/dashes to ASCII equivalents
        - squeezes multiple blanks into a single space
        """
        # Canonicalise look-alike unicode chars (e.g. “smart quotes” → ")
        text = unicodedata.normalize('NFKC', text)
        # Ensure punctuation is whitespace-delimited so BPE sees it as a stand-alone symbol
        text = re.sub(r'([,.;:!?()\[\]«»"“”‘’])', r' \1 ', text)
        # Collapse runs of whitespace that steps 2 may have introduced
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _preprocess(self, text: str) -> str:
        if self.domain == 'twitter':
            return self._preprocess_twitter(text)
        elif self.domain == 'news':
            return self._preprocess_news(text)
        elif self.domain == 'unknown':
            # Normalize, separate punctuation, collapse whitespace
            text = unicodedata.normalize('NFKC', text)
            text = re.sub(r'([,.;:!?()\[\]«»"“”‘’])', r' \1 ', text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            return text
        else:
            return text  # Fallback to returning the original text

    def _split_once(self, word_str: str) -> List[str]:
        """
        Cache the split result instead of calling .split() multiple times.
        """
        return word_str.split()

    def _word_to_list(self, word_str: str) -> List[str]:
        """
        Keep word as a list of symbols from the beginning.
        """
        return list(word_str)

    def _rebuild_freqs(self, words: List[List[str]]) -> Counter:
        """
        Rebuild the word frequencies from the list of words.
        """
        freqs = Counter()
        for w in words:
            freqs[tuple(w)] += 1
        return freqs

    def train(self, texts: List[str]) -> None:
        """
        Train the BPE tokenizer on the given texts
        
        Args:
            texts: List of training texts
        """
        # Initialize vocabulary with characters
        words = []
        for text in texts:
            text = self._preprocess(text)  # Bug fix: keep the return value
            for w in text.split():
                token_list = self._word_to_list(w)  # ['_', 'h', 'u', 'g']
                words.append(token_list)
                self.word_frequencies[tuple(token_list)] += 1  # Use tuple as key
        print("[BPE] Finished preprocessing and collecting initial words.")

        # Initialize vocabulary with characters
        vocab = set()
        for word in words:
            vocab.update(word)
        print(f"[BPE] Initial vocabulary built with {len(vocab)} unique tokens.")

        # Add special tokens to vocabulary
        for token in self.special_tokens:
            vocab.add(token)
        print(f"[BPE] Added {len(self.special_tokens)} special tokens to vocabulary.")

        # Convert vocabulary to token_to_id mappings
        for token in vocab:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        domain_merge_factor = {
            'twitter': 1.2,   # 20 % more merges
            'news':    0.8,   # 20 % fewer
            'unknown': 1.0,
        }.get(self.domain, 1.0)

        # Calculate number of merges based on vocab size and domain factor
        num_merges = int((self.vocab_size - len(self.token_to_id)) * domain_merge_factor)

        # Initialize variables before merge loop
        new_pairs_this_block = []
        pair_stats = self._get_stats(words, self.word_frequencies)
        self.pair_stats = dict(pair_stats)

        # Build a max-heap for pair frequencies (negate freq for max-heap)
        heap = [(-freq, pair) for pair, freq in pair_stats.items()]
        heapq.heapify(heap)
        pair_latest_freq = dict(pair_stats)  # Track latest freq for each pair

        for i in tqdm(range(num_merges), desc="BPE"):
            # Pop max, skip stale
            while heap:
                neg_freq, best_pair = heapq.heappop(heap)
                freq = -neg_freq
                # Only accept if this is the latest freq for this pair
                if pair_latest_freq.get(best_pair, 0) == freq and freq > 0:
                    break
            else:
                break  # Heap empty

            
            words = self._merge_pair(best_pair, words)

            vanished_pairs, appeared_pairs = self._update_stats(words, best_pair)
            new_pairs_this_block.append((best_pair, freq))

            for pair in set(vanished_pairs) | set(appeared_pairs):
                new_freq = self.pair_stats.get(pair, 0)   # already updated inside _update_stats
                pair_latest_freq[pair] = new_freq
                if new_freq > 0:
                    heapq.heappush(heap, (-new_freq, pair))


            # Print every 20 iterations the merged pairs made
            if (i + 1) % 20 == 0 or i == num_merges - 1:
                print(f"[BPE] Merges for iterations {i-18 if i-18>0 else 1} to {i+1}:")
                for idx, (pair, freq) in enumerate(new_pairs_this_block[-20:], start=1):
                    print(f"  {idx}. {pair} (frequency: {freq})")
                new_pairs_this_block = []

            # Rebuild frequencies and heap every 1000 merges
            if (i + 1) % 1000 == 0:
                self.word_frequencies = self._rebuild_freqs(words)
                pair_stats = self._get_stats(words, self.word_frequencies)
                heap = [(-freq, pair) for pair, freq in pair_stats.items()]
                heapq.heapify(heap)
                pair_latest_freq = dict(pair_stats)
        # Check for at least one bigram (character pair) merge
        bigram_count = sum(1 for pair in self.merges if len(pair[0]) == 1 and len(pair[1]) == 1)
        if bigram_count > 0:
            print(f"[BPE] SUCCESS: At least one bigram (character pair) was merged ({bigram_count} found).")
        else:
            print("[BPE] WARNING: No bigram merges found! This will lead to a penalty.")

        # Ensure vocab size does not exceed limit (remove excess tokens if any)
        if len(self.token_to_id) > self.vocab_size:
            # Remove tokens with highest ids until vocab size matches limit
            excess = len(self.token_to_id) - self.vocab_size
            for _ in range(excess):
                max_id = max(self.id_to_token.keys())
                del_token = self.id_to_token.pop(max_id)
                self.token_to_id.pop(del_token)

        # Re-join tokens into strings for saving the vocabulary
        final_vocab = set(''.join(word) for word in words)  # Ensure unique tokens
        try:
            with open("final_vocab.txt", "w", encoding="utf-8") as f:
                for token in sorted(final_vocab):  # Sort for consistent output
                    f.write(f"{token}\n")
            print("[BPE] Final vocabulary saved to 'final_vocab.txt'.")
        except Exception as e:
            print(f"[BPE] Could not save final vocabulary: {e}")

        # Remove duplicate bigram frequency calculation
        str_words = self._words_to_strings(words)
        bigram_freqs = self._get_stats([list(w) for w in str_words], self.word_frequencies)
        if bigram_freqs:
            sorted_bigrams = sorted(bigram_freqs.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_bigrams[:5]
            bottom5 = sorted_bigrams[-5:]
            print("Top 5 bigrams by frequency after training:")
            for (bigram, freq) in top5:
                print(f"  {bigram}: {freq}")
            print("Bottom 5 bigrams by frequency after training:")
            for (bigram, freq) in bottom5:
                print(f"  {bigram}: {freq}")
            # Optionally, save to file
            try:
                with open("bigrams_stats.txt", "w", encoding="utf-8") as f:
                    f.write("Top 5 bigrams by frequency after training:\n")
                    for (bigram, freq) in top5:
                        f.write(f"  {bigram}: {freq}\n")
                    f.write("Bottom 5 bigrams by frequency after training:\n")
                    for (bigram, freq) in bottom5:
                        f.write(f"  {bigram}: {freq}\n")
            except Exception as e:
                print(f"Could not save bigram stats: {e}")
        bigram_freqs = self._get_stats([w.split() for w in str_words], self.word_frequencies)
        if bigram_freqs:
            sorted_bigrams = sorted(bigram_freqs.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_bigrams[:5]
            bottom5 = sorted_bigrams[-5:]
            print("\nSummary of bigrams after all merges:")
            print("Top 5 bigrams:")
            for (bigram, freq) in top5:
                print(f"  {bigram}: {freq}")
            print("Bottom 5 bigrams:")
            for (bigram, freq) in bottom5:
                print(f"  {bigram}: {freq}")
        print(f"[BPE] Final vocabulary size: {self.get_vocab_size()}")

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs using BPE
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token IDs
        """
        text = self._preprocess(text)
        text = f"[BOS] {text} [EOS]"
        words = text.split()
        token_ids = []
        for word in words:
            if word in self.special_tokens:
                token_ids.append(self.token_to_id[word])
                continue
            # Splitting to individual letters here:
            current_word = list(word)
            # Apply BPE merges
            while True:
                pairs = self._get_stats([current_word], Counter({tuple(current_word): 1}))  # pairs possible within the current word
                if not pairs:
                    break
                best_pair = None
                for pair in pairs:
                    if pair in self.merges:
                        best_pair = pair
                        break
                if not best_pair:
                    break
                current_word = self._merge_pair(best_pair, [current_word])[0]
            # Convert the final word into token IDs
            for token in current_word:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    token_ids.append(self.token_to_id["[UNK]"])
        print(f"[BPE] Encoding '{text}' → token IDs: {token_ids}")
        return token_ids

    # Helper for reporting: convert list of token lists to list of space-joined strings
    def _words_to_strings(self, words: List[List[str]]) -> List[str]:
        return [' '.join(word) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            The decoded text string
        """
        # Convert token IDs to tokens
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Join intelligently: insert space unless token is punctuation or a placeholder
        text = ''
        for tok in tokens:
            if tok == self.space_token:
                text += ' '
            elif text and not tok.startswith("'") and tok not in ',.!?:;':
                text += tok
            else:
                text += tok
        print(f"[BPE] Decoding {token_ids} → '{text}'")
        return text.strip()