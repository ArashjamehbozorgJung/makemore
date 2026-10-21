"""
Lab 8: the Transformer. Where its parameters live, and one attention layer replayed by hand.
run:  python course/labs/lab08_transformer.py
optional, after you trained a transformer with makemore.py -o names:
      python course/labs/lab08_transformer.py names/model.pt
"""

import sys
import math
import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import Transformer, ModelConfig

torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)
train, test = load_data()
V = train.get_vocab_size()
config = ModelConfig(vocab_size=V, block_size=train.get_output_length(), n_layer=4, n_head=4, n_embd=64)

print("=== 1. where the parameters are (the constructor prints its own count first)")
torch.manual_seed(0)
model = Transformer(config)
groups = {}
for name, p in model.named_parameters():
    key = name.rsplit('.', 1)[0]  # drop the trailing .weight / .bias
    if key.startswith('transformer.h.') and not key.startswith('transformer.h.0.'):
        key = 'transformer.h.1..3 (3 more blocks, same as block 0)'
    groups[key] = groups.get(key, 0) + p.numel()
for key, n in groups.items():
    print(f"  {key:52s} {n:7d}")
total = sum(p.numel() for p in model.parameters())
print(f"  total: {total}  (the 0.20M printed above leaves out lm_head)")

if len(sys.argv) > 1:
    model.load_state_dict(torch.load(sys.argv[1]))
    print(f"\nloaded trained weights from {sys.argv[1]}")

print("\n=== 2. replay block 0's attention by hand on '.emma'")
word = 'emma'
idx = torch.tensor([[0] + [train.stoi[c] for c in word]])
B, T = idx.shape
with torch.no_grad():
    x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(T)).unsqueeze(0)
    blk = model.transformer.h[0]
    a = blk.ln_1(x)
    C, nh = a.size(-1), blk.attn.n_head
    hs = C // nh
    q, k, v = blk.attn.c_attn(a).split(C, dim=2)
    print(f"x (B,T,C) = {tuple(x.shape)};  q, k, v each {tuple(q.shape)}")
    q = q.view(B, T, nh, hs).transpose(1, 2)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    v = v.view(B, T, nh, hs).transpose(1, 2)
    print(f"split into {nh} heads of size {hs}: q, k, v each (B, nh, T, hs) = {tuple(q.shape)}")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
    print(f"scores = q @ k^T / sqrt({hs}): (B, nh, T, T) = {tuple(scores.shape)}")
    scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
    att = F.softmax(scores, dim=-1)
    y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
    y = blk.attn.c_proj(y)
    print("max |hand-computed - CausalSelfAttention output| =", (y - blk.attn(a)).abs().max().item())
    labels = ['.'] + list(word)
    for h in range(nh):
        print(f"\nhead {h}: row = position reading, column = position read from (rows sum to 1)")
        print("      " + "     ".join(labels))
        for t in range(T):
            print(f"  {labels[t]}  " + " ".join(f"{w:5.2f}" for w in att[0, h, t].tolist()))
    if len(sys.argv) == 1:
        print("\n(untrained weights: the pattern is near-uniform noise. pass a trained model.pt to see learned patterns)")
