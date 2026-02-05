import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import random

global_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(global_device)
random.seed(69)


class TextEmbedding(nn.Module):
    """
    Creates Embedding Matrix Lookup. Fills in tensor
    with row vectors of floats which are the embeddings
    of each token in the seq.
    
    Shape Returned: [batch_size, seq_len, d_model]
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedded = nn.Embedding(vocab_size, d_model, device=global_device)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.device == next(self.embedded.parameters()).device
        return self.embedded(input)
    
class PositionalEmbedding(nn.Module):
    """
    Adds Positional Encoding vectors to text embeddings. Retains shape.
    These PE embeddings are learned; different than SinusoidalPE used in AIAYN paper.

    Shape Returned: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len: int, d_model: int, device=global_device) -> None:
        super().__init__()
        self.PE = nn.Parameter(torch.randn(seq_len, d_model).to(device).unsqueeze(0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.PE
    
class MultiHeadAttention(nn.Module):
    """
    Masked Mulithead Attention. Exactly like the Decoder MHA implementation from
    AIAYN paper.

    Shape Returned: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len, d_model, num_heads, mask, device=global_device):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.mask = mask
        scale = 1 / sqrt(12)

        self.Wq = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wk = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wv = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wo = nn.Linear(num_heads * self.d_k, d_model, bias=False, device=device)
        self.attn_dropout = nn.Dropout(p=0.1)
        with torch.no_grad():
            self.Wo.weight.mul_(scale)

    def sdpa(self, Q, K, V, d_k):

        first_mul = torch.div(Q @ K.mT, d_k ** .5)

        mask_matrix = torch.triu(torch.full(first_mul.shape, float('-inf')), diagonal=1).to(global_device)

        masked = first_mul + mask_matrix

        attention = F.softmax(masked, dim=-1)

        attention = self.attn_dropout(attention)

        out = attention @ V

        return attention, out

    def forward(self, X, mask=False):
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        B, S, H, dh = X.size()[0], X.size()[1], self.num_heads, self.d_k

        Q = (torch.reshape(Q, (B, S, H, dh))).transpose(1, 2)
        K = (torch.reshape(K, (B, S, H, dh))).transpose(1, 2)
        V = (torch.reshape(V, (B, S, H, dh))).transpose(1, 2)

        attention, output = self.sdpa(Q, K, V, self.d_k)
        print(output.shape)

        reshaped = (torch.transpose(output, 1, 2)).reshape(B, S, H * dh)
        
        return self.Wo(reshaped)

class LayerNorm(nn.Module):
    """
    LayerNorm implementation from ~scratch. Follows formula in Pytorch docs.

    Shape Returned: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len, d_model, eps=1e-05, device=global_device):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model, device=device))
        self.beta  = nn.Parameter(torch.zeros(d_model, device=device))


    def forward(self, x):
        z = (x - torch.mean(x, dim=-1, keepdim=True)) / ((torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps) ** .5)
        y = (z * self.gamma) + self.beta
        return y

class PositionWiseFFN(nn.Module):
    """
    PositionWiseFFN from ~scratch. 
    
    Shape Returned: [batch_size, seq_len, d_model]
    """

    def __init__(self, seq_len, d_model, inner_dimension=3072, device=global_device):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.inner_dimension = inner_dimension
        scale = 1 / sqrt(12)

        self.W1 = nn.Linear(d_model, inner_dimension, bias=True, device=device)
        self.W2 = nn.Linear(inner_dimension, d_model, bias=True, device=device)
        with torch.no_grad():
            self.W2.weight.mul_(scale)

    def forward(self, x):
        z = F.gelu(self.W1(x))
        out = self.W2(z)
        return out

class DecoderBlock(nn.Module):
    """
    Decoder Block Structure based on GPT paper (with model spec. changes based on GPT-2 paper)
    
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    """

    def __init__(self, seq_len, d_model, num_heads, mask):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask

        self.masked_multi_self_attention = MultiHeadAttention(seq_len=seq_len, d_model=d_model, num_heads=num_heads, mask=mask)
        self.layer_norm_one = LayerNorm(seq_len=seq_len, d_model=d_model)
        self.layer_norm_two = LayerNorm(seq_len=seq_len, d_model=d_model)
        self.feed_forward = PositionWiseFFN(seq_len=seq_len, d_model=d_model)
        self.resid_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        y = x + self.resid_dropout(self.masked_multi_self_attention(self.layer_norm_one(x)))

        z = y + self.resid_dropout(self.feed_forward(self.layer_norm_two(y)))

        return z

class FullGPT(nn.Module):
    """
    Full 12 Layers of GPT like mentioned in the paper. Also handles initializing embeddings and positional embedding.
    
    Includes weight tying at the end to return output back to original vocab space (logits).
    """

    def __init__(self, vocab_size, seq_len, d_model, num_heads, mask):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = LayerNorm(seq_len=seq_len, d_model=d_model)
        self.embedding = TextEmbedding(vocab_size, d_model)
        self.PE = PositionalEmbedding(seq_len, d_model)
        self.decoders = nn.ModuleList([DecoderBlock(seq_len=seq_len, d_model=d_model, num_heads=num_heads, mask=mask) for _ in range(12)])
        self.embd_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        pe = self.PE(embedded)
        positionally_encoded = self.embd_dropout(pe)

        for decoder in self.decoders:
            final_decoder_output = decoder(positionally_encoded)
            positionally_encoded = final_decoder_output
        
        final_decoder_output_ln = self.layer_norm(final_decoder_output)

        logits = (final_decoder_output_ln @ self.embedding.embedded.weight.t()) * (1.0 / sqrt(self.d_model))
        
        return logits

"""
vocab_size = 10
d_model = 128
seq_len = 4

embedding = TextEmbedding(vocab_size, d_model)
PE = PositionalEmbedding(seq_len, d_model)

example = torch.randint(0, vocab_size, (2, seq_len)).to(global_device)
x = embedding(example)
z = PE(x)

mha = MultiHeadAttention(4, 128, 8, False)
yur = mha(z)
layer = LayerNorm(seq_len=seq_len, d_model=d_model)
out = layer.forward(yur)
pwffn = PositionWiseFFN(seq_len=seq_len, d_model=d_model)
pwffn_output = pwffn.forward(out)
print("YURRRRRTOEEE", torch.std_mean(pwffn_output))
breakpoint()
outy = F.dropout(pwffn_output, p=.1)


full_gpt = FullGPT(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, num_heads=8, mask=1)
print(full_gpt(example).shape)

"""