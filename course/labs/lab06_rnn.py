"""
Lab 6: the RNN and GRU models. A hidden state carried forward one character at a time.
run:  python course/labs/lab06_rnn.py
"""

import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import RNN, ModelConfig

torch.set_printoptions(precision=3, sci_mode=False)
train, test = load_data()
V = train.get_vocab_size()
config = ModelConfig(vocab_size=V, block_size=train.get_output_length(), n_embd=64, n_embd2=64)

for cell_type in ['rnn', 'gru']:
    torch.manual_seed(0)
    model = RNN(config, cell_type=cell_type)
    print(f"=== {cell_type.upper()} parameters")
    for name, p in model.named_parameters():
        print(f"  {name:22s} {str(tuple(p.shape)):10s} {p.numel():6d}")
    print("  total:", sum(p.numel() for p in model.parameters()))
    X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))
    logits, loss = model(X, Y)
    print(f"  input {tuple(X.shape)} -> logits {tuple(logits.shape)} | loss at init {loss.item():.4f}\n")

print("=== stepping the RNN cell by hand on '.emma': the hidden state h is the model's only memory")
torch.manual_seed(0)
model = RNN(config, cell_type='rnn')
idx = torch.tensor([[0] + [train.stoi[c] for c in 'emma']])
emb = model.wte(idx)                          # (1, 5, 64)
h = model.start                               # (1, 64), a learned vector, not zeros (zeros at init)
for t in range(idx.size(1)):
    h = model.cell(emb[:, t, :], h)           # new h from (this character, previous h)
    print(f"t={t} read {'.emma'[t]!r}: h[:5] = {h[0, :5].detach()}  |h| = {h.norm().item():.3f}")
print("every h value is squashed into (-1, 1) by tanh")

print("\n=== GRU gates on one step: r (reset) and z (update) are sigmoids, so each entry is in (0, 1)")
torch.manual_seed(0)
model = RNN(config, cell_type='gru')
xt = model.wte(torch.tensor([train.stoi['e']]))
hprev = torch.randn(1, 64).tanh()
xh = torch.cat([xt, hprev], dim=1)
r = F.sigmoid(model.cell.xh_to_r(xh))
z = F.sigmoid(model.cell.xh_to_z(xh))
print("r[:6] =", r[0, :6].detach())
print("z[:6] =", z[0, :6].detach())
ht = model.cell(xt, hprev)
print("check the blend ht = (1-z)*hprev + z*hbar is what the cell returned:")
hbar = torch.tanh(model.cell.xh_to_hbar(torch.cat([xt, r * hprev], dim=1)))
print("  max difference:", ((1 - z) * hprev + z * hbar - ht).abs().max().item())
