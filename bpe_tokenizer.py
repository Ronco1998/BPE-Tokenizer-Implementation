from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re
from base_tokenizer import BaseTokenizer
import unicodedata

#TODO: Actually do this

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

        # Add Twitter-specific special tokens if domain is twitter
        if self.domain == 'twitter':
            special_token_list = ['[HASHTAG]', '[MENTION]', '[URL]', '[EMOJI]']
            for idx, token in enumerate(special_token_list, start=-10):
                if token not in self.special_tokens:
                    self.special_tokens[token] = idx
        # For news headlines, no lowercasing or aggressive normalization (for NER)
        if self.domain == 'news':
            self.preserve_case = True
        elif self.domain == 'unknown':
        # Robust fallback: canonicalise visually identical chars
            self._normalize = lambda s: unicodedata.normalize('NFKC', s)
        else:
            raise ValueError(f"Unsupported domain {self.domain}")


    def _get_stats(self, words: List[str]) -> Dict[Tuple[str, str], int]:
        """x
        Count frequency of adjacent pairs in the words
        
        Args:
            words: List of words to count pairs from
            
        Returns:
            Dictionary mapping pairs to their frequency
        """
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += 1
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], words: List[str]) -> List[str]:
        """
        Merge all occurrences of the pair in the words
        
        Args:
            pair: The pair to merge
            words: List of words to merge the pair in
            
        Returns:
            List of words with the pair merged
        """
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_words = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word.split()):
                if i < len(word.split()) - 1 and word.split()[i] == pair[0] and word.split()[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word.split()[i])
                    i += 1
            new_words.append(' '.join(new_word))
        return new_words
    
    def _preprocess_twitter(self, text: str) -> str:
        """
        Preprocess Twitter text: separate hashtags, mentions, URLs, and emojis as tokens.
        """
        # Replace URLs with [URL]
        text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
        # Replace mentions with [MENTION]
        text = re.sub(r'@[\w_]+', ' [MENTION] ', text)
        # Replace hashtags with [HASHTAG]
        text = re.sub(r'#[\w_]+', ' [HASHTAG] ', text)
        # Replace emojis with [EMOJI] (simple unicode emoji range)
        emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' [EMOJI] ', text)
        return text

    def _preprocess_news(self, text: str) -> str:
        """
        Preprocess news headline text: minimal normalization, preserve case for NER.
        """
        # Optionally, you could add more sophisticated punctuation splitting here if needed
        return text  # No lowercasing or normalization

    def _preprocess(self, text: str) -> str:
        if self.domain == 'twitter':
            return self._preprocess_twitter(text.lower())
        elif self.domain == 'news':
            return self._preprocess_news(text)          # keep case
        elif self.domain == 'unknown':
            return self._normalize(text)
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
            self._preprocess(text)
            # Split text into words and add special tokens
            text_words = text.split()
            for word in text_words:
                # Add word boundary markers
                word = ' '.join(list(word))
                words.append(word)
                self.word_frequencies[word] += 1
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in words:
            vocab.update(word.split())
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            vocab.add(token)

        #TODO: add special tokens according to the domain we're in rn!
        
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

        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            # Find the most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Merge the pair in all words
            words = self._merge_pair(best_pair, words)
            
            # Add the merge to our merges dictionary
            self.merges[best_pair] = ''.join(best_pair)
            
            # Add the merged token to vocabulary if not already present
            merged_token = ''.join(best_pair)
            if merged_token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[merged_token] = token_id
                self.id_to_token[token_id] = merged_token

            #TODO: how do we keep track of where we are in the vocabulary size?
    
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs using BPE
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token IDs
        """
        # Add BOS and EOS tokens, and preprocess for Twitter if needed
        text = self._preprocess(text)
        text = f"[BOS] {text} [EOS]"
        
        # Split text into words
        words = text.split()
        token_ids = []
        
        for word in words:
            if word in self.special_tokens:
                token_ids.append(self.token_to_id[word])
                continue
                
            # Split word into characters
            chars = list(word)
            current_word = ' '.join(chars)
            
            # Apply BPE merges
            while True:
                pairs = self._get_stats([current_word]) # pairs possible within the current word
                if not pairs:
                    break
                    
                # Find the most frequent pair that exists in our merges
                best_pair = None
                for pair in pairs:
                    if pair in self.merges:
                        best_pair = pair
                        break
                
                if not best_pair:
                    break
                    
                # Merge the pair
                current_word = self._merge_pair(best_pair, [current_word])[0]
            
            # Convert the final word into token IDs
            for token in current_word.split():
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    token_ids.append(self.token_to_id["[UNK]"])
        
        return token_ids
    
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
        return text.strip()