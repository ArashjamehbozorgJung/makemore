"""
Lab 5: the MLP model. How the torch.roll loop builds a fixed window of previous characters.
run:  python course/labs/lab05_mlp.py
"""

import torch
from _setup import load_data, show
from makemore import MLP, ModelConfig

train, test = load_data()
V = train.get_vocab_size()
letters = '.abcdefghijklmnopqrstuvwxyz_'  # '.' = token 0, '_' = token 27 = <BLANK> (only the MLP has it)

print("=== 1. replay MLP.forward's gathering loop on '.emma' with a window of 3")
idx = torch.tensor([[0] + [train.stoi[c] for c in 'emma']])
block_size = 3
for k in range(block_size):
    print(f"k={k}: idx = {idx[0].tolist()}  shown as {''.join(letters[i] for i in idx[0].tolist())!r}")
    idx = torch.roll(idx, 1, 1)
    idx[:, 0] = V  # the <BLANK> token, integer 27
print("\nread the three rows column by column: column t is the window the MLP sees at position t")
rows = []
idx = torch.tensor([[0] + [train.stoi[c] for c in 'emma']])
for k in range(block_size):
    rows.append(idx[0].tolist())
    idx = torch.roll(idx, 1, 1)
    idx[:, 0] = V
for t in range(5):
    window = ''.join(letters[rows[k][t]] for k in reversed(range(block_size)))
    print(f"  position {t}: oldest..newest = {window!r}")

print("\n=== 2. the real MLP as makemore.py builds it (block_size=16, n_embd=64, n_embd2=64)")
config = ModelConfig(vocab_size=V, block_size=train.get_output_length(), n_embd=64, n_embd2=64)
model = MLP(config)
for name, p in model.named_parameters():
    print(f"  {name:14s} {str(tuple(p.shape)):12s} {p.numel():6d} numbers")
print("  total:", sum(p.numel() for p in model.parameters()))
X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))
logits, loss = model(X, Y)
print("input", tuple(X.shape), "-> logits", tuple(logits.shape), f"| loss at init {loss.item():.4f}")
