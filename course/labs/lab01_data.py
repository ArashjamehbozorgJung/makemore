"""
Lab 1: how makemore.py turns a text file of names into tensors of integers.
run:  python course/labs/lab01_data.py
"""

from _setup import load_data, show
from makemore import CharDataset

print("=== 1. what create_datasets() prints (this is makemore.py's own output)")
train, test = load_data()

print("\n=== 2. the character -> integer table (stoi)")
print(train.stoi)
print("integer 0 is not in the table: it is reserved for the special <START>/<STOP> token")
print("vocab_size =", train.get_vocab_size(), "(26 letters + token 0)")
print("block_size =", train.get_output_length(), "(longest name + 1 slot for the <START> token)")

print("\n=== 3. one example, exactly as __getitem__ builds it")
ds = CharDataset(['emma'], train.chars, train.max_word_length)
x, y = ds[0]
print("x =", x.tolist())
print("y =", y.tolist())
print("\nposition t: the model has seen x[0..t] -> it must predict y[t]")
for t in range(len(x)):
    target = y[t].item()
    if target == -1:
        print(f"  t={t:2d}: seen {show(x[:t+1].tolist()):17s} -> y=-1  (padding: loss ignores this position)")
    else:
        print(f"  t={t:2d}: seen {show(x[:t+1].tolist()):17s} -> must predict {show([target])!r}")

print("\n=== 4. the same names file contains repeated lines")
print("training words:", len(train.words), "| distinct training words:", len(set(train.words)))
train_set = set(train.words)
overlap = [w for w in test.words if w in train_set]
print(f"{len(overlap)} of {len(test.words)} test words also appear in the training set, e.g. {overlap[:6]}")
