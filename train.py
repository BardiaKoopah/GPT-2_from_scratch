from tokenizer import BPE
from model import FullGPT
from dataloader import DataLoader

import torch
from torch import optim
import torch.nn as nn
import math
import torch.nn.functional as F

from math import cos, pi, exp

from datasets import load_dataset

global_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def build_token_pool(split_texts, eos_id, bpe):
    inside_doc = False
    tokens_1d = []

    for idx, row in enumerate(split_texts):

        if not bool(row.strip()):
            if inside_doc:
                tokens_1d.append(eos_id)
                inside_doc = False
            else:
                continue
        else:
            token_ids = bpe.tokenize(row)
            tokens_1d.extend(token_ids)
            inside_doc = True
    
    return tokens_1d


def training_loop():
    total_updates = 100000

    bpe = BPE('test')
    gpt2 = FullGPT(vocab_size=50257, seq_len=128, d_model=512, num_heads=8)

    yes_decay = []
    no_decay = []
    for param in gpt2.named_parameters():
        name = param[0]

        if 'gamma' in name or 'beta' in name or name.endswith('bias'):
            no_decay.append(param[1])
        else:
            yes_decay.append(param[1])
    
    optimizer = optim.AdamW([
        {"params": yes_decay, 'weight_decay': 0.01},
        {"params": no_decay,  'weight_decay': 0}], 
        lr=2.5e-4, betas=(0.90, 0.98), eps=1e-9)
    
    linear_lr = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-8, end_factor=1, total_iters=2000)
    cosine_annealing_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=98000, eta_min=0)
    
    scheduler = optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[linear_lr, cosine_annealing_lr], milestones=[2000])
    

    eos = bpe.eos_id
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = ds["train"]["text"]
    valid_texts = ds["validation"]["text"]

    train_tokens = build_token_pool(train_texts, eos, bpe=bpe)
    valid_tokens = build_token_pool(valid_texts, eos, bpe=bpe)

    train_loader = DataLoader(tokens=train_tokens, batch_size=8, L=128, is_train=True)
    valid_loader = DataLoader(tokens=valid_tokens, batch_size=8, L=128, is_train=False)

    gpt2.train()
    for i in range(total_updates):
        print(f"ON {i}")
        train_inputs, train_labels = train_loader.__call__()
        optimizer.zero_grad()

        logits = gpt2(train_inputs)
        V = logits.shape[-1]
        
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loss = loss_fn(logits.reshape(-1, V), train_labels.reshape(-1))

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if i > 0 and i % 100 == 0:
            print(f'UPDATE: {i} | TRAINING LOSS: {train_loss.item()} | TRAINING PPL: {exp(min(20, train_loss.item()))}')


        if i > 0 and i % 300 == 0:
            
            gpt2.eval()
            with torch.no_grad():
                valid_inputs, valid_labels = valid_loader.__call__()

                logits = gpt2(valid_inputs)
                V = logits.shape[-1]

                sample_index = 6 #sample index
                target_seq = valid_labels[sample_index].detach().cpu().tolist()
                pred_seq = logits[sample_index].argmax(dim=-1).detach().cpu().tolist()

                def strip(ids):
                    out = []
                    for t in ids:
                        if t == eos:
                            break
                        out.append(t)
                    return out
                
                target_seq_cleaned = strip(target_seq)
                pred_seq_cleaned = strip(pred_seq)

                target_text = bpe.decode(target_seq_cleaned)
                pred_text = bpe.decode(pred_seq_cleaned)

                print("\n=== SAMPLE ===")
                print("TGT: ", target_text)
                print("PRED:", pred_text)

        
                loss_fn = torch.nn.CrossEntropyLoss()
                valid_loss = loss_fn(logits.reshape(-1, V), valid_labels.reshape(-1))

                print(f'UPDATE: {i} | VALIDATION LOSS: {valid_loss.item()} | VALIDATION PPL: {exp(min(20, valid_loss.item()))}')

            gpt2.train()

training_loop()