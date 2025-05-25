from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re
from base_tokenizer import BaseTokenizer

#TODO: Actually do this

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize the BPE tokenizer

        Args:
            vocab_size: Maximum size of the vocabulary
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.merges = {}  # Store the learned BPE merges
        self.word_frequencies = Counter()  # Store word frequencies during training
        
    def _get_stats(self, words: List[str]) -> Dict[Tuple[str, str], int]:
        """
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
    
    def train(self, texts: List[str]) -> None:
        """
        Train the BPE tokenizer on the given texts
        
        Args:
            texts: List of training texts
        """
        # Initialize vocabulary with characters
        words = []
        for text in texts:
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
        
        # Convert vocabulary to token_to_id mapping
        for token in vocab:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)
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
    
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs using BPE
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token IDs
        """
        # Add BOS and EOS tokens
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
                pairs = self._get_stats([current_word])
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
        
        # Join tokens and remove spaces between characters
        text = ''.join(tokens)
        
        return text 