"""
Lab 9b: split a trained model's test loss into names that also occur in the
training set (leaked through duplicate lines in names.txt) and truly unseen names.
run (pass the SAME --type, sizes and --seed you trained with):
      python course/labs/lab09_leak_split.py --type transformer names/model.pt
      python course/labs/lab09_leak_split.py --type gru --seed 1 g1/model.pt
"""

import argparse
import torch
from torch.nn import functional as F
from _setup import load_data
from makemore import Transformer, Bigram, MLP, RNN, BoW, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--type', default='transformer')
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--n-layer', type=int, default=4)
parser.add_argument('--n-head', type=int, default=4)
parser.add_argument('--n-embd', type=int, default=64)
parser.add_argument('--n-embd2', type=int, default=64)
args = parser.parse_args()

train, test = load_data(args.seed)  # the split depends on the seed, so it must match training
config = ModelConfig(vocab_size=train.get_vocab_size(), block_size=train.get_output_length(),
                     n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, n_embd2=args.n_embd2)
model = {
    'transformer': lambda: Transformer(config),
    'bigram':      lambda: Bigram(config),
    'mlp':         lambda: MLP(config),
    'rnn':         lambda: RNN(config, cell_type='rnn'),
    'gru':         lambda: RNN(config, cell_type='gru'),
    'bow':         lambda: BoW(config),
}[args.type]()
model.load_state_dict(torch.load(args.model_path))
model.eval()

X, Y = map(torch.stack, zip(*[test[i] for i in range(len(test))]))
with torch.no_grad():
    logits, _ = model(X)
V = logits.size(-1)
nll = F.cross_entropy(logits.view(-1, V), Y.view(-1), ignore_index=-1, reduction='none').view(Y.shape)
mask = (Y != -1).float()
per_word, n = (nll * mask).sum(1), mask.sum(1)
train_set = set(train.words)
leaked = torch.tensor([w in train_set for w in test.words])

def loss(sel):
    return (per_word[sel].sum() / n[sel].sum()).item()
print(f"all {len(test)} test names       : {loss(torch.ones_like(leaked)):.4f}")
print(f"{leaked.sum().item():4d} also in training set : {loss(leaked):.4f}")
print(f"{(~leaked).sum().item():4d} truly unseen        : {loss(~leaked):.4f}  <- the honest number")
