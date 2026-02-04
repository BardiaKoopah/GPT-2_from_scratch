import torch
import torch.nn as nn

from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from tokenizers import Tokenizer
from transformers import AutoTokenizer

import random

global_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DataLoader():

    def __init__(self, tokens, batch_size, L, is_train=True):
        self.batch_size = batch_size
        self.L = L
        self.tokens = tokens
        
        if is_train:
            pass
        else:
            random.seed(69)

    def __call__(self):
        N = len(self.tokens)
        assert N >= self.L + 1
        assert N - (self.L+1) + 1 >= self.batch_size

        s = random.sample(range(0, (N - self.L + 1)), self.batch_size)

        inputs = []
        labels = []
        for si in s:
            inputs.append(self.tokens[si : si + self.L])
            labels.append(self.tokens[si + 1: si + self.L + 1])
        
        tensor_input = torch.tensor(inputs, dtype=torch.int64).to(device=global_device)
        tensor_labels = torch.tensor(labels, dtype=torch.int64).to(device=global_device)

        assert tensor_input.shape == (self.batch_size, self.L)
        assert tensor_labels.shape == (self.batch_size, self.L)

        return tensor_input, tensor_labels
