"""
Lab 2: logits, softmax, cross-entropy, and the Bigram model.
run:  python course/labs/lab02_bigram.py
"""

import math
import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import Bigram, ModelConfig

torch.set_printoptions(precision=4, sci_mode=False)

print("=== 1. softmax by hand: logits (any real numbers) -> probabilities (positive, sum to 1)")
logits = torch.tensor([2.0, 1.0, 0.1])
probs = logits.exp() / logits.exp().sum()
print("logits          :", logits)
print("probs by hand   :", probs, "sum =", probs.sum().item())
print("F.softmax       :", F.softmax(logits, dim=-1))

print("\n=== 2. cross-entropy by hand: -log(probability the model gave to the correct answer)")
for target in [0, 2]:
    by_hand = -probs[target].log()
    by_torch = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
    print(f"correct answer = {target}: p = {probs[target]:.4f}, -log p = {by_hand:.4f}, F.cross_entropy = {by_torch:.4f}")

print("\n=== 3. the Bigram model before any training")
train, test = load_data()
V = train.get_vocab_size()
model = Bigram(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))  # a batch of 32 examples
logits, loss = model(X, Y)
print("its only parameter: model.logits, shape", tuple(model.logits.shape), "- all zeros at the start")
print("input X shape", tuple(X.shape), "-> output logits shape", tuple(logits.shape))
print(f"loss at init = {loss.item():.4f}   ln(27) = {math.log(27):.4f}   (uniform guess over 27 tokens)")

print("\n=== 4. the best possible bigram, computed by counting (no neural net, no gradients)")
def seqs(words):
    for w in words:
        yield [0] + [train.stoi[ch] for ch in w] + [0]  # <START> name <STOP>
N = torch.zeros(V, V)
for s in seqs(train.words):
    for a, b in zip(s, s[1:]):
        N[a, b] += 1
def avg_nll(P, words):
    nll, n = 0.0, 0
    for s in seqs(words):
        for a, b in zip(s, s[1:]):
            nll -= P[a, b].log().item()
            n += 1
    return nll / n
P = N / N.sum(1, keepdim=True)
P1 = (N + 1) / (N + 1).sum(1, keepdim=True)  # add-one smoothing so unseen pairs are not probability 0
print(f"count-based bigram, train loss: {avg_nll(P, train.words):.4f}  <- the lowest loss ANY bigram can reach on this training set")
print(f"count-based bigram (add-one smoothed), test loss: {avg_nll(P1, test.words):.4f}")
print(f"pairs never seen in training: {(N == 0).sum().item()} of {V*V}")

print("\n=== 5. train the Bigram 'neural net' by gradient descent and watch it reach the counting answer")
Xall, Yall = map(torch.stack, zip(*[train[i] for i in range(len(train))]))  # whole training set as one batch
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
for step in range(501):
    _, loss = model(Xall, Yall)
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step:3d} | train loss {loss.item():.4f}")
learned = F.softmax(model.logits.detach(), dim=-1)
print(f"largest |learned prob - counted prob| over all 27x27 entries: {(learned - P).abs().max().item():.4f}")
print("row for <START> (which letter starts a name), counted vs learned, first 6 letters a..f:")
print("  counted:", P[0, 1:7])
print("  learned:", learned[0, 1:7])
