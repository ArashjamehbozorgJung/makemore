"""
Lab 9: all six models side by side. Size, loss before training, and a test that
none of them can see the future.
run:  python course/labs/lab09_all_models.py
"""

import torch
from _setup import load_data
from makemore import Bigram, MLP, RNN, BoW, Transformer, ModelConfig

train, test = load_data()
V = train.get_vocab_size()
config = ModelConfig(vocab_size=V, block_size=train.get_output_length(),
                     n_layer=4, n_head=4, n_embd=64, n_embd2=64)  # makemore.py's defaults
builders = {
    'bigram':      lambda: Bigram(config),
    'mlp':         lambda: MLP(config),
    'rnn':         lambda: RNN(config, cell_type='rnn'),
    'gru':         lambda: RNN(config, cell_type='gru'),
    'bow':         lambda: BoW(config),
    'transformer': lambda: Transformer(config),
}
X, Y = map(torch.stack, zip(*[train[i] for i in range(32)]))
j = 6  # we will change the input token at this position
X2 = X.clone()
X2[:, j] = (X2[:, j] + 1) % V

rows = []
for name, build in builders.items():
    torch.manual_seed(0)
    model = build()
    with torch.no_grad():
        _, loss = model(X, Y)
        # the test must hold for ANY parameter values, so randomize them first.
        # (without this the Bigram would pass trivially: its logits all start at zero)
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        logits, _ = model(X)
        logits2, _ = model(X2)
    before = (logits[:, :j] - logits2[:, :j]).abs().max().item()
    after = (logits[:, j:] - logits2[:, j:]).abs().max().item()
    rows.append((name, sum(p.numel() for p in model.parameters()), loss.item(), before, after))

print(f"\nchanged the input at position {j}; a model that cannot see the future must leave positions 0..{j-1} untouched")
print(f"{'model':12s} {'params':>8s} {'loss@init':>10s} {'max change pos<'+str(j):>18s} {'max change pos>='+str(j):>19s}")
for name, n, loss, before, after in rows:
    print(f"{name:12s} {n:8d} {loss:10.4f} {before:18.6f} {after:19.6f}")
print("ln(27) = 3.2958 is the loss of a uniform guess")
