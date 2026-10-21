"""
Lab 4: how generate() turns a model into new names: softmax, temperature, top-k, sampling.
run:  python course/labs/lab04_sampling.py
"""

import torch
from torch.nn import functional as F
from _setup import load_data, show
from makemore import Bigram, ModelConfig, generate

torch.set_printoptions(precision=3, sci_mode=False)

train, test = load_data()
V = train.get_vocab_size()

# skip training: set the Bigram's logits to log(counts+1). softmax(log(c)) = c / sum(c),
# so the model's probabilities become exactly the smoothed counted probabilities
N = torch.ones(V, V)
for w in train.words:
    s = [0] + [train.stoi[ch] for ch in w] + [0]
    for a, b in zip(s, s[1:]):
        N[a, b] += 1
model = Bigram(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
model.logits.data = N.log()

print("=== 1. the distribution over the first letter (row <START>), at different temperatures")
letters = '.abcdefghijklmnopqrstuvwxyz'
for temperature in [0.5, 1.0, 2.0]:
    p = F.softmax(model.logits[0] / temperature, dim=-1)
    top = torch.topk(p, 5)
    pretty = ', '.join(f"{letters[i]}={v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"T={temperature}: top-5 {pretty}")
print("lower T sharpens the distribution toward the favourite; higher T flattens it toward uniform")

print("\n=== 2. top-k: keep the k largest logits, set the rest to -inf, so they get probability 0")
logits = model.logits[0].clone()
v, _ = torch.topk(logits, 3)
logits[logits < v[-1]] = -float('Inf')
p = F.softmax(logits, dim=-1)
print("nonzero probabilities with top_k=3:", {letters[i]: round(p[i].item(), 3) for i in p.nonzero().flatten().tolist()})

def decode(row):
    row = row[1:]                                    # drop the <START> token we fed in
    row = row[:row.index(0)] if 0 in row else row     # crop at the first <STOP> token
    return train.decode(row)

steps = train.get_output_length() - 1
print("\n=== 3. greedy decoding (do_sample=False): always take the single most likely next token")
X = generate(model, torch.zeros(3, 1, dtype=torch.long), steps, do_sample=False)
print([decode(r) for r in X.tolist()], "<- identical every time, and a poor name")

print("\n=== 4. sampling (do_sample=True): draw the next token at random from the distribution")
torch.manual_seed(42)
X = generate(model, torch.zeros(10, 1, dtype=torch.long), steps, do_sample=True)
print("raw output row 0:", X[0].tolist())
print("shown with '.' for token 0:", show(X[0].tolist()))
print("decoded names:", [decode(r) for r in X.tolist()])
print("\nwith top_k=3:")
X = generate(model, torch.zeros(10, 1, dtype=torch.long), steps, do_sample=True, top_k=3)
print([decode(r) for r in X.tolist()])
