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

@torch.no_grad()
def eval_ppl_stream(model, tokens, seq_len, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    ids = torch.tensor(tokens, dtype=torch.long)

    for i in range(0, len(ids) - 1, seq_len):
        x = ids[i:i+seq_len]
        y = ids[i+1:i+seq_len+1]

        if x.numel() < seq_len or y.numel() < seq_len:
            continue

        x = x.to(device)
        y = y.to(device)

        logits = model(x.unsqueeze(0))
        V = logits.size(-1)

        loss = F.cross_entropy(
            logits.view(-1, V),
            y.view(-1),
            reduction="sum"
        )

        total_nll += loss.item()
        total_tokens += y.numel()

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    model.train()
    return avg_nll, ppl

def strip_at_eos(ids, eos_id):
    out = []
    for t in ids:
        if t == eos_id:
            break
        out.append(t)
    return out

def sample_next(logits_1d, temperature=1.0, top_k=50):
    temperature = max(float(temperature), 1e-8)
    logits_1d = logits_1d / temperature

    if top_k is not None:
        top_k = min(int(top_k), logits_1d.numel())
        v, ix = torch.topk(logits_1d, top_k)
        masked = torch.full_like(logits_1d, float("-inf"))
        masked.scatter_(0, ix, v)
        logits_1d = masked

    probs = torch.softmax(logits_1d, dim=-1)
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, seq_len, device, temperature=1.0, top_k=50, eos_id=None):
    model.eval()
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        ctx = ids[-seq_len:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        next_logits = logits[0, -1]
        next_id = sample_next(next_logits, temperature=temperature, top_k=top_k)

        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    model.train()
    return ids

def pretty(text: str) -> str:
    return (
        text.replace("Ġ", " ")
            .replace("\t", "\\t")
            .replace("\n", "\\n\n")
    )


def training_loop():
    total_updates = 100000

    bpe = BPE('test')
    gpt2 = FullGPT(vocab_size=50257, seq_len=128, d_model=768, num_heads=12)

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
            val_loss, val_ppl = eval_ppl_stream(gpt2, valid_tokens, 128, global_device)
            print(f"VAL STREAM NLL: {val_loss:.4f} | VAL STREAM PPL: {val_ppl:.2f}")

            gpt2.eval()
            with torch.no_grad():
                valid_inputs, valid_labels = valid_loader.__call__()
                logits = gpt2(valid_inputs)

                sample_index = 0
                prompt_ids = valid_inputs[sample_index].detach().cpu().tolist()
                tgt_ids = valid_labels[sample_index].detach().cpu().tolist()
                pred_ids = logits[sample_index].argmax(dim=-1).detach().cpu().tolist()

                print("\n=== TEACHER-FORCED ===")
                print("PROMPT:", pretty(bpe.decode(strip_at_eos(prompt_ids, eos))))
                print("TGT   :", pretty(bpe.decode(strip_at_eos(tgt_ids, eos))))
                print("PRED  :", pretty(bpe.decode(strip_at_eos(pred_ids, eos))))

            gpt2.train()

            print("\n=== GENERATION ===")
            gen_ids = generate(
                model=gpt2,
                prompt_ids=prompt_ids,
                max_new_tokens=80,
                seq_len=128,
                device=global_device,
                temperature=1.0,
                top_k=50,
                eos_id=eos
            )
            print("GEN:", pretty(bpe.decode(strip_at_eos(gen_ids, eos))))
            print()


training_loop()