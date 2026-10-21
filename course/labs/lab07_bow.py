"""
Lab 7: the Bag-of-Words model. A masked softmax of zeros is a running average.
run:  python course/labs/lab07_bow.py
"""

import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import BoW, CausalBoW, ModelConfig

torch.set_printoptions(precision=3, sci_mode=False)
train, test = load_data()
V = train.get_vocab_size()
config = ModelConfig(vocab_size=V, block_size=train.get_output_length(), n_embd=64, n_embd2=64)

T = 5
print("=== 1. CausalBoW.forward step by step, for T=5 positions")
att = torch.zeros((1, T, T))
print("start: all zeros\n", att[0])
mask = torch.tril(torch.ones(T, T))
att = att.masked_fill(mask == 0, float('-inf'))
print("after masking the future (above the diagonal) with -inf\n", att[0])
att = F.softmax(att, dim=-1)
print("after softmax along each row: exp(-inf)=0, the rest share equally\n", att[0])

print("\n=== 2. so att @ x is the running mean of x over positions 0..t")
x = torch.randn(1, T, 4)
y = CausalBoW(ModelConfig(vocab_size=V, block_size=T))(x)
running_mean = x.cumsum(dim=1) / torch.arange(1, T + 1).view(1, T, 1)
print("max |CausalBoW(x) - running mean| =", (y - running_mean).abs().max().item())

print("\n=== 3. the full BoW model")
torch.manual_seed(0)
model = BoW(config)
for name, p in model.named_parameters():
    print(f"  {name:32s} {str(tuple(p.shape)):10s} {p.numel():6d}")
print("  total:", sum(p.numel() for p in model.parameters()))
X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))
logits, loss = model(X, Y)
print(f"input {tuple(X.shape)} -> logits {tuple(logits.shape)} | loss at init {loss.item():.4f}")
print("note: CausalBoW has no parameters. the averaging weights are fixed, not learned.")
