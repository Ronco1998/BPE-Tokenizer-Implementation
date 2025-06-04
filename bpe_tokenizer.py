from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re
from base_tokenizer import BaseTokenizer
import unicodedata
from tqdm import tqdm

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

        # Add Twitter-specific special tokens if domain is twitter
        if self.domain == 'twitter':
            special_token_list = ['[HASHTAG]', '[MENTION]', '[URL]', '[EMOJI]']
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


    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent pairs in the words and save them as an attribute.

        Args:
            words: List of words (as lists of tokens) to count pairs from
        Returns:
            Dictionary of pairs and their frequencies
        """
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        self.pair_stats = dict(pairs)
        return self.pair_stats

    def _update_stats(self, words: List[List[str]], last_pair: Tuple[str, str]) -> None:
        """
        Update only pairs adjacent to the merged token in the saved pair statistics.

        Args:
            words: List of words (as lists of tokens) after the merge
            last_pair: The last merged pair (e.g. ('a', 'b'))

        This method updates self.pair_stats in-place. It only increments the counts for pairs
        that are adjacent to the newly merged token (i.e., pairs that could have changed due to the merge).
        It also removes the merged pair from the stats, as it no longer exists as a pair.
        """
        merged_token = last_pair[0] + last_pair[1]
        # For each word, look for occurrences of the merged token
        for word in words:
            for idx, token in enumerate(word):
                if token != merged_token:
                    continue
                # If there is a token to the left, update the left pair count
                if idx - 1 >= 0:
                    left_pair = (word[idx - 1], merged_token)
                    # Increment the count for this left pair
                    self.pair_stats[left_pair] = self.pair_stats.get(left_pair, 0) + 1
                # If there is a token to the right, update the right pair count
                if idx + 1 < len(word):
                    right_pair = (merged_token, word[idx + 1])
                    self.pair_stats[right_pair] = self.pair_stats.get(right_pair, 0) + 1
        # Remove the old merged pair from the stats, as it no longer exists
        if last_pair in self.pair_stats:
            del self.pair_stats[last_pair]

    def _merge_pair(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        """
        Merge all occurrences of the pair in the words (token lists)
        
        Args:
            pair: The pair to merge
            words: List of words (as lists of tokens) to merge the pair in
            
        Returns:
            List of words (as lists of tokens) with the pair merged
        """
        new_words = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words
    
    def _preprocess_twitter(self, text: str) -> str:
        """
        Preprocess Twitter text: normalize, lowercase, separate hashtags, mentions, URLs, emojis, and punctuation as tokens.
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Replace URLs with [URL]
        text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
        # Replace mentions with [MENTION]
        text = re.sub(r'@[\w_]+', ' [MENTION] ', text)
        # Replace hashtags with [HASHTAG]
        text = re.sub(r'#[\w_]+', ' [HASHTAG] ', text)
        # Replace emojis with [EMOJI] (simple unicode emoji range)
        emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' [EMOJI] ', text)
        # Separate punctuation as in news
        text = re.sub(r'([,.;:!?()\[\]«»"“”‘’])', r' \1 ', text)
        # Collapse runs of whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
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

    def train(self, texts: List[str]) -> None:
        """
        Train the BPE tokenizer on the given texts
        
        Args:
            texts: List of training texts
        """
        # Initialize vocabulary with characters
        words = []
        for text in texts:
            text = self._preprocess(text)
            # Split text into words and add special tokens
            text_words = text.split()
            for word in text_words:
                # If the word is a special token, add as a single token
                if word in self.special_tokens:
                    words.append([word])
                    self.word_frequencies[word] += 1
                else:
                    # Add word boundary markers
                    word = list(word)
                    words.append(word)
                    self.word_frequencies[word] += 1
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

        # Perform BPE merges
        num_merges = int(domain_merge_factor * (self.vocab_size - len(self.token_to_id))) # how many merges we need to perform :) (this is the k we talked about)

        self._get_stats(words)  # Save stats in self.pair_stats
        for i in tqdm(range(num_merges), desc="BPE Merges"):  # tqdm progress bar
            if not self.pair_stats:
                break

            # Find the most frequent pair
            best_pair = max(self.pair_stats.items(), key=lambda x: x[1])[0]
            freq = self.pair_stats[best_pair]

            # Merge the pair in all words
            words = self._merge_pair(best_pair, words)

            # Add the merge to our merges dictionary
            self.merges[best_pair] = best_pair[0] + best_pair[1]

            # Add the merged token to vocabulary if not already present
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                if len(self.token_to_id) >= self.vocab_size:
                    break  # Stop if vocab size limit reached
                token_id = len(self.token_to_id)
                self.token_to_id[merged_token] = token_id
                self.id_to_token[token_id] = merged_token

            self._update_stats(words, best_pair)

            # Print the new bigram and its frequency after every 20 merges
            if i % 20 == 0:
                print(f"Merged bigram: {best_pair} (frequency: {freq}) after {i+1} merges")

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
    
        # For reporting, convert words back to string for stats
        str_words = self._words_to_strings(words)
        bigram_freqs = self._get_stats([w.split() for w in str_words])
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
        bigram_freqs = self._get_stats([w.split() for w in str_words])
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
            # Split word into characters (as list)
            current_word = list(word)
            # Apply BPE merges
            while True:
                pairs = self._get_stats([current_word])  # pairs possible within the current word
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