"""
Lab 10: reproduce the weak spots discussed in lesson 10, so you can check them yourself.
run:  python course/labs/lab10_findings.py
"""

import subprocess
import sys
import torch
from _setup import load_data, set_args, REPO
import makemore
from makemore import Bigram, ModelConfig, evaluate

train, test = load_data()
V = train.get_vocab_size()

print("=== 1. evaluate(max_batches=10) looks at 11 batches, not 10")
set_args(device='cpu')
model = Bigram(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
calls = []
original_forward = model.forward
def counting_forward(idx, targets=None):
    calls.append(idx.size(0))
    return original_forward(idx, targets)
model.forward = counting_forward
evaluate(model, train, batch_size=100, max_batches=10)
print(f"train: {len(calls)} batches, {sum(calls)} words out of {len(train)}")
calls.clear()
evaluate(model, test, batch_size=100, max_batches=10)
print(f"test : {len(calls)} batches, {sum(calls)} words out of {len(test)} (the test set simply ran out)")

print("\n=== 2. names that appear twice in names.txt can land in both train and test")
train_set = set(train.words)
overlap = [w for w in test.words if w in train_set]
print(f"{len(overlap)} of {len(test.words)} test words ({100*len(overlap)/len(test.words):.1f}%) are also training words")

print("\n=== 3. weight decay is applied to every parameter, including biases and LayerNorm")
model = makemore.Transformer(ModelConfig(vocab_size=V, block_size=train.get_output_length()))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)
print("parameter groups:", len(optimizer.param_groups), "| weight_decay of that group:", optimizer.param_groups[0]['weight_decay'])

print("\n=== 4. 'PyTorch is the only requirement' (README): importing makemore without tensorboard")
# hide the installed tensorboard package from a fresh python, the way a machine without it would look
code = """
import sys, importlib.abc
class Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'tensorboard':
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
sys.meta_path.insert(0, Hide())
import makemore
"""
r = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True)
lines = r.stderr.strip().splitlines()
print('\n'.join(l for l in lines if 'File "' in l and ('makemore.py' in l or 'tensorboard' in l)))
print(lines[-1])
