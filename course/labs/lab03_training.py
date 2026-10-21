"""
Lab 3: what loss.backward() and optimizer.step() actually compute.
run:  python course/labs/lab03_training.py
"""

import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import Bigram, ModelConfig

torch.set_printoptions(precision=5, sci_mode=False)

train, test = load_data()
V = train.get_vocab_size()
model = Bigram(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))

print("=== 1. loss.backward() fills .grad: how the loss changes if you nudge each parameter")
_, loss = model(X, Y)
model.zero_grad(set_to_none=True)
loss.backward()
print("model.logits.grad has the same shape as model.logits:", tuple(model.logits.grad.shape))

# the gradient of mean cross-entropy w.r.t. the logits has a closed form:
# for every counted position: (softmax(logits) - onehot(target)) / (number of counted positions)
mask = Y != -1
M = mask.sum()
probs = F.softmax(model.logits.detach()[X[mask]], dim=-1)
onehot = F.one_hot(Y[mask], V).float()
manual = torch.zeros(V, V).index_add_(0, X[mask], (probs - onehot) / M)
print("counted positions in this batch:", M.item())
print("max |autograd grad - hand-derived grad| =", (model.logits.grad - manual).abs().max().item())
print("grad row for <START>, columns a..f:", model.logits.grad[0, 1:7])
print("negative entries = 'raise this logit to lower the loss' (these letters started names in this batch)")

print("\n=== 2. optimizer.step(): AdamW's very first step moves each parameter by about lr, whatever the grad size")
lr = 5e-4  # makemore.py's default learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)
before = model.logits.detach().clone()
optimizer.step()
delta = model.logits.detach() - before
moved = model.logits.grad != 0
print("parameters with nonzero grad:", moved.sum().item(), "of", moved.numel())
print("their |change| ranges from", delta[moved].abs().min().item(), "to", delta[moved].abs().max().item())
print("parameters with zero grad changed by:", delta[~moved].abs().max().item(), "(weight decay of a zero is zero)")

print("\n=== 3. the makemore.py loop: batch -> forward -> zero_grad -> backward -> step, repeated")
from makemore import InfiniteDataLoader
torch.manual_seed(0)
model = Bigram(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)
loader = InfiniteDataLoader(train, batch_size=32, num_workers=0)
for step in range(3001):
    X, Y = loader.next()
    _, loss = model(X, Y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step:4d} | batch loss {loss.item():.4f}")
print("with lr=5e-4 the bigram is still above its best (2.454) after 3000 steps: each logit moves at most ~lr per step")
