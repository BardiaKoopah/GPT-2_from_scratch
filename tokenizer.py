import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPT2Tokenizer

from collections import defaultdict, Counter
import itertools
import os

import multiprocessing as mp
import time
from typing import Optional
from pathlib import Path

import heapq

import json


def preprocess_and_wordfreqs(text, tokenizer):
    county = Counter()
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        county[word] += 1
    return county
    
class BPE():
    
    def __init__(self, dataset: Optional[str], vocab_size: int = 50257):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self._pretokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._gpt2_slow = GPT2Tokenizer.from_pretrained("gpt2")
        self._byte_decoder = self._gpt2_slow.byte_decoder
        self.byte_alphabet = list(self._gpt2_slow.byte_encoder.values())

        vocab_file_path = 'BPE_stuff/vocab.json'
        merge_file_path = 'BPE_stuff/merges.txt'

        if os.path.exists(vocab_file_path) and os.path.exists(merge_file_path):
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                self.token_to_id = json.load(f)

            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            assert "<|endoftext|>" in self.token_to_id
            self.eos_token = "<|endoftext|>"
            self.eos_id = self.token_to_id[self.eos_token]

            pair_to_rank = {}
            rank = 0

            with open(merge_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    a, b = line.split()
                    pair_to_rank[(a, b)] = rank
                    rank += 1

            self.pair_to_rank = pair_to_rank
        else:
            print("VOCAB AND MERGE FILES DON'T EXIST!!")

    def to_byte_unicode(self, s):
        b = s.encode("utf-8")
        return [self._gpt2_slow.byte_encoder[byte] for byte in b]


    def create_vocab_and_merge(self):
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        dataset = load_dataset(f'{self.dataset}', streaming=True)
        corpus = dataset['train']
        heap_of_pairs = []
        pair_counts = defaultdict(int)
        pair_positions = defaultdict(list)
        word_freqs = Counter() #this is just a counter with the word and its associated frequency -> Counter{'test': 124, ...}
        symbols = [] #list containing lists where each sub-list is a split token -> [['c', 'a', 't'], ['a', 'r', 't']]
        freqs = [] #list containing the freqs of each of the sublists inside "symbols" (an index in freqs matches with an index in symbols and vice versa)

        for index, dic in enumerate(corpus): #this is just where we load each document one by one (streaming=True) and update word_freqs
            text = dic['text']
            processed_county = preprocess_and_wordfreqs(text, tokenizer=tokenizer)
            word_freqs.update(processed_county)

            if index > 0 and index % 100000 == 0:
                print(f"ON CURRENT DOCUMENT: {index}")

        print("SUCCESSFULLY CREATED WORD_FREQS!")
        print(len(word_freqs))

        for token, freq in word_freqs.items():
            symbols.append(self.to_byte_unicode(token))
            freqs.append(freq)

        vocab = ["<|endoftext|>"] + self.byte_alphabet
        merge_list_rules = []

        for seq_id in range(len(symbols)):
            for i in range(len(symbols[seq_id]) - 1):
                pair = (symbols[seq_id][i], symbols[seq_id][i+1]) # get an adjacent pair from symbols -> ('a', 'b')
                pair_counts[pair] += freqs[seq_id] # add the respective frequency that that pair occurs to pair counts dict -> {}
                pair_positions[pair].append((seq_id, i)) #seq_id tells us which sublist in symbol and i tells us the position in that sublist the pair occurs

        for key, val in pair_counts.items():
            heap_of_pairs.append((-val, key))
        
        heapq.heapify(heap_of_pairs)

        i = 0

        num_merges = self.vocab_size - len(vocab)

        while i < num_merges:
            if i > 0 and i % 100 == 0:
                print(f"ON RUN {i}")
            popped_node = heapq.heappop(heap_of_pairs) #looks like: (-frequency, pair)
            while -pair_counts[popped_node[-1]] != popped_node[-2]: #staleness check. Check to see if freq of heap node for a pair matches global pair_counts freq for that pair
                popped_node = heapq.heappop(heap_of_pairs)
            pair = popped_node[-1]
            occurrences = pair_positions[pair] #gives us back of list of (seq_id, idx) where we can find that pair
            pair_positions[pair] = [] 
            joined_pair = "".join(pair)
            for seq_id, index in sorted(occurrences, key=lambda x: x[1], reverse=True):
                if index + 1 >= len(symbols[seq_id]): continue
                if symbols[seq_id][index] == pair[0] and symbols[seq_id][index+1] == pair[1]:
                    if index > 0:
                        xa = (symbols[seq_id][index-1], symbols[seq_id][index])
                        xc = (symbols[seq_id][index-1], joined_pair)
                        pair_counts[xa] -= freqs[seq_id] # decrement 'X A' if X exists
                        pair_counts[xc] += freqs[seq_id] # increment 'X C' if X exists
                        pair_positions[xc].append((seq_id, index-1))
                        heapq.heappush(heap_of_pairs, (-pair_counts[xc], xc))
                    if index + 2 < len(symbols[seq_id]):
                        by = (symbols[seq_id][index+1], symbols[seq_id][index+2])
                        cy = (joined_pair, symbols[seq_id][index+2])
                        pair_counts[by] -= freqs[seq_id] # decrement 'B Y' if Y exists
                        pair_counts[cy] += freqs[seq_id] # increment 'C Y' if Y exists
                        pair_positions[cy].append((seq_id, index))
                        heapq.heappush(heap_of_pairs, (-pair_counts[cy], cy))
                    ab = (symbols[seq_id][index], symbols[seq_id][index+1])
                    pair_counts[ab] -= freqs[seq_id] # decrement 'A B' (it's now C)
                    symbols[seq_id][index] = joined_pair #changes 'A B' to 'C'
                    symbols[seq_id].pop(index+1) # gets rid of B

            i += 1
            merge_list_rules.append(pair)
            vocab.append(joined_pair)
        
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}

        with open("BPE_stuff/vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)
        
        with open("BPE_stuff/merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in merge_list_rules:
                f.write(f"{a} {b}\n")

    def inference_merge(self, split, pair):
        i = 0
        while i < len(split)-1:
            if (split[i], split[i+1]) == pair:
                split[i:i+2] = [''.join(pair)]
                i = max(i - 1, 0)
            i += 1
        return split

    
    def tokenize(self, text):
        words_with_offsets = self._pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        split_words = [self.to_byte_unicode(w) for w in new_words]
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
        toks = [self.id_to_token[i] for i in ids]
        text = "".join(toks)
        byte_arr = bytearray([self._byte_decoder[c] for c in text])
        return byte_arr.decode("utf-8", errors="replace")

if __name__ == '__main__':

    #bpe = BPE('Skylion007/openwebtext', 50257)
    #bpe.create_vocab_and_merge()
    
    bpe = BPE("test")
    s = "Traditional stocks made from a single piece of timber , or laminated stocks."
    ids = bpe.tokenize(s)
    print("decoded:", bpe.decode(ids))

