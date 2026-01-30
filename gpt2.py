import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from collections import defaultdict, Counter
import itertools

import multiprocessing as mp
import time
import typing
from pathlib import Path

_worker_tokenizer = None  # process-local


def worker_init():
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained("gpt2")


def preprocess_and_wordfreqs(text):
    county = Counter()
    words_with_offsets = _worker_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        county[word] += 1
    return county
    
class BPE():
    
    def __init__(self, dataset: str, vocab_size: int = 50257):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self._pretokenizer = AutoTokenizer.from_pretrained("gpt2")

        vocab_file_path = 'BPE_stuff/vocab.txt'
        merge_file_path = 'BPE_stuff/merge_list_rules.txt'

        """
        try:
            token_to_id = {}
            with open(vocab_file_path, 'r') as vocab_file:
                for idx, line in enumerate(vocab_file):
                    token = line.rstrip("\n")
                    token_to_id[token] = idx
            id_to_token = {v: k for k, v in token_to_id.items()}
            self.token_to_id = token_to_id
            self.id_to_token = id_to_token
        
            pair_to_rank = {}
            with open(merge_file_path, 'r') as merge_file:
                for rank, line in enumerate(merge_file):
                    pair = tuple(line.rstrip("\n").split())
                    pair_to_rank[pair] = rank
            self.pair_to_rank = pair_to_rank
        
            self.eos_token = "<|endoftext|>"
            self.eos_id = self.token_to_id[self.eos_token]
        
        except Exception as e:
            raise(e)
        """

    def get_max_pair(self, split_tuple) -> tuple[str, str]:
        word_pair_freqs = defaultdict(int)
        for tup in split_tuple:
            for i in range(len(tup[0]) - 1):
                pair = (tup[0][i], tup[0][i+1])
                word_pair_freqs[pair] += tup[1]
        
        max_pair = max(word_pair_freqs, key=word_pair_freqs.get)
        return max_pair

    def training_merge(self, split_tuple, max_pair) -> list[tuple[list, int]]:
        new_split = [([], freq) for symb,freq in split_tuple]
        i = 0
        j = 0

        while j < len(split_tuple):
            while i < len(split_tuple[j][0]) - 1:
                if (split_tuple[j][0][i], split_tuple[j][0][i+1]) == max_pair:
                    new_split[j][0].append("".join(max_pair))
                    i += 2
                else:
                    new_split[j][0].append(split_tuple[j][0][i])
                    i += 1
            if i == len(split_tuple[j][0]) - 1:
                new_split[j][0].extend(split_tuple[j][0][i:])
            i = 0
            j += 1
        return new_split

    def create_vocab_and_merge(self, chunk_size):
        dataset = load_dataset(f'{self.dataset}')
        corpus = dataset['train']

        word_freqs = Counter()
        alphabet = []

        num_workers = mp.cpu_count()
        pool = mp.Pool(processes=num_workers, initializer=worker_init)

        for i in range(0, len(corpus), chunk_size):
            if i % 100000 == 0:
                print(f"CURRENT CHUNK: {i}")

            local_county = Counter()
            
            mini_batch = corpus[i:i+chunk_size]['text']

            processed_result = pool.map(preprocess_and_wordfreqs, mini_batch)
            for county in processed_result:
                local_county.update(county)

            word_freqs.update(local_county)

        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)

        alphabet.sort()

        vocab = ["<|endoftext|>"] + alphabet.copy()
        merge_list_rules = []

        training_data = [([letter for letter in key], val) for key, val in word_freqs.items()]

        i = 0

        while i < self.vocab_size:
            max_pair = self.get_max_pair(training_data)
            vocab.append("".join(max_pair))
            merge_list_rules.append(" ".join(max_pair))
            new_list = self.training_merge(training_data, max_pair)
            print(len(new_list))
            breakpoint()
            training_data = new_list
            print(f'ADDED TO VOCAB AND MERGE: {i}')
            i += 1
        
        with open("vocab.txt", "w", encoding="utf-8") as vocab_txt:
            for token in vocab:
                vocab_txt.write(f"{token}\n")
        
        with open("merge_list_rules.txt", "w", encoding="utf-8") as merge_txt:
            for merge in merge_list_rules:
                merge_txt.write(f"{merge}\n")

    def inference_merge(self, split, pair):
        i = 0
        while i < len(split)-1:
            if (split[i], split[i+1]) == pair:
                split[i:i+2] = [''.join(pair)]
            i += 1
        return split

    
    def tokenize(self, text):
        words_with_offsets = self._pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        split_words = [[letter for letter in w] for w in new_words]
        new_splits = []

        # self.pair_to_rank -> {('a', 'b') : 1, ('cr', 'y') : 2, ('b', 'ad') : 1}
        # split_words = [ [('t', 'h'), ('h', 'e')], [('Ġ', 'c'), ('c', 'a'), ('a', 't')] ]

        for split in split_words:
            while True:
                if len(split) < 2: break
                tup_split = [(split[j], split[j+1]) for j in range(len(split)-1)]
                pair = min(tup_split, key=lambda p: self.pair_to_rank.get(p, float('inf')))
                if pair not in self.pair_to_rank:
                    break
                split = self.inference_merge(split, pair)
            new_splits.append(split)
        
        flattened = list(itertools.chain.from_iterable(new_splits))
        tokenized = list(map(lambda t: self.token_to_id[t], flattened))

        return tokenized

    def decode(self, ids):
        return self._pretokenizer.decode(ids)

if __name__ == '__main__':

    bpe = BPE('Skylion007/openwebtext', 50257)

    bpe.create_vocab_and_merge(chunk_size=1000)
