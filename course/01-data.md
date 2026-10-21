# Lesson 1 — How names become numbers

**This lesson promises:** you can write down, by hand, the exact two tensors `x` and `y` that `makemore.py` builds for any name. You will know why each number is there.

**Code covered:** `create_datasets` (lines 545–572) and `CharDataset` (lines 506–543).
**Lab:** `python course/labs/lab01_data.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **string** | Text in Python, e.g. `'emma'`. |
| **encode / decode** | Encode: turn a string into a list of integers. Decode: turn integers back into a string. A neural network can only do arithmetic on numbers, so text must be encoded first. |
| **`stoi` / `itos`** | Two Python dictionaries. `stoi` ("string to integer") maps `'a'→1, 'b'→2, … 'z'→26`. `itos` is the reverse. |
| **special token** | An integer that doesn't stand for a character. Here token `0` means "start of name" when it is input, and "end of name" when it is the answer. |
| **training set / test set** | Two disjoint lists of lines from `names.txt`. The model learns from the training set only. The test set measures how well it does on lines it never learned from. |
| **target** | The correct answer the model should have predicted. |
| **padding** | Filler positions added so every example has the same length. Same length is required to stack examples into one tensor. |
| **mask / `ignore_index`** | Marking positions whose prediction must not count in the loss. Here the marker is `-1`. |
| **`Dataset`** | A PyTorch class with two methods: `__len__` (how many examples) and `__getitem__(i)` (return example number `i`). |

---

## 1. What the model is asked to do

A language model here does exactly one thing. Given the characters of a name so far, it outputs a probability for every possible next character.

For "emma" the questions are:

| The model has seen | It must predict |
|---|---|
| (start) | `e` |
| (start) `e` | `m` |
| (start) `em` | `m` |
| (start) `emm` | `a` |
| (start) `emma` | (end) |

Five questions from one four-letter name. "Start" and "end" are not letters, but the model needs them. It must know where a name begins in order to predict the first letter. It must be able to say "stop" so that generated names have an end.

## 2. The vocabulary: lines 545–559

```python
words = data.splitlines()
words = [w.strip() for w in words]     # remove spaces/newline at the ends of each line
words = [w for w in words if w]        # drop empty lines
chars = sorted(list(set(''.join(words))))
max_word_length = max(len(w) for w in words)
```

- `''.join(words)` glues all names into one long string. `set(...)` keeps each distinct character once. `sorted(...)` puts them in a fixed order. For `names.txt` the result is the 26 lowercase letters.
- Nothing here is specific to names or to English. Feed it a file containing `Ä`, `-`, or spaces *inside* lines, and those become vocabulary entries too. Only spaces at the ends of a line are stripped.

## 3. The integer table: line 512

```python
self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
```

`enumerate` numbers the characters from 0. The `+1` shifts them to 1…26. **Integer 0 is deliberately left free** for the special start/end token.

```python
def get_vocab_size(self):
    return len(self.chars) + 1   # 26 letters + token 0 = 27
```

## 4. One example: `__getitem__`, lines 535–543

This is the most important function in the file to understand exactly. Here it is with "emma" (`e=5, m=13, a=1`) and `max_word_length = 15`:

```python
ix = self.encode(word)                                   # [5, 13, 13, 1]
x = torch.zeros(self.max_word_length + 1, dtype=torch.long)  # 16 zeros
y = torch.zeros(self.max_word_length + 1, dtype=torch.long)  # 16 zeros
x[1:1+len(ix)] = ix      # write the name into x starting at position 1
y[:len(ix)] = ix         # write the name into y starting at position 0
y[len(ix)+1:] = -1       # everything after the end-token position becomes -1
```

Result, with positions numbered:

```
position:  0   1   2   3   4   5   6  ...  15
x       :  0   5  13  13   1   0   0  ...   0
y       :  5  13  13   1   0  -1  -1  ...  -1
```

Read it column by column. **At position `t`, the model has read `x[0..t]` and must output `y[t]`.**

- Column 0: it has read `0` (start). Answer `5` (`e`).
- Column 4: it has read `0 e m m a`. Answer `0`. Here `0` means **end**.
- Columns 5–15: answer `-1`. These are padding. The loss ignores them (lesson 2 shows how).

So `y` is `x` shifted left by one position. That shift is the entire trick. One example of length 16 holds 5 separate prediction questions, all answered in one pass.

Why does `x` still contain zeros at positions 5–15? Every example must be exactly 16 long to be stacked into a `(32, 16)` batch. The value there doesn't matter, because:
1. its answer is `-1`, so it never counts, and
2. every model in this file is built so that position `t` can only see positions `0..t`. Padding sits *after* the real name, so no real question can see it. Lab 9 proves this property for all six models.

## 5. `block_size` = 16

```python
def get_output_length(self):
    return self.max_word_length + 1   # <START> token followed by words
```

The longest name has 15 letters. Plus 1 for the start token gives 16. Every model is told `block_size = 16`. It is the longest input sequence the models will ever see.

> **Careful:** the end token of a 15-letter name doesn't fit. For `muhammadibrahim`, `x` is `0` plus 15 letters, filling all 16 slots. `y` is the 15 letters, and `y[15] = 0` is the end token. That one does fit, because `y` is shifted. So training is fine. But a generated name can be at most 15 characters. When the model hasn't produced an end token by then, the generator just stops (lesson 4).

## 6. The train/test split: lines 561–566

```python
test_set_size = min(1000, int(len(words) * 0.1))   # 1000 here
rp = torch.randperm(len(words)).tolist()           # a random shuffle of 0..32032
train_words = [words[i] for i in rp[:-test_set_size]]
test_words  = [words[i] for i in rp[-test_set_size:]]
```

- `torch.randperm` uses PyTorch's random generator. `makemore.py` seeds it with `--seed` (default 3407) on line 623, just before this runs. So the split is **the same on every run with the same seed**. That is how `--sample-only` gets the same split back. All labs use seed 3407, so they see exactly the split that `makemore.py` uses.
- The test set is capped at 1,000 lines. With a bigger file you still get 1,000 test lines.

> **Careful — a real data problem:** `names.txt` has 32,033 lines but only **29,494 distinct names**. 2,539 names appear exactly twice (`emma`, `olivia`, `ava`, …). The split shuffles *lines*, not distinct names. So one copy can land in train and the other in test. With seed 3407, **153 of the 1,000 test names (15.3%) are also training names.** The "test" set is therefore not entirely unseen data. Lesson 10 measures how much this flatters the test loss. (It matters: it flips the conclusion of lesson 9.)

---

## Lab 1

**Where:** terminal, in the repo folder, with the virtual environment active.

```bash
python course/labs/lab01_data.py
```

**What you should see** (my real output):

```
=== 2. the character -> integer table (stoi)
{'a': 1, 'b': 2, 'c': 3, ... 'y': 25, 'z': 26}
integer 0 is not in the table: it is reserved for the special <START>/<STOP> token
vocab_size = 27 (26 letters + token 0)
block_size = 16 (longest name + 1 slot for the <START> token)

=== 3. one example, exactly as __getitem__ builds it
x = [0, 5, 13, 13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y = [5, 13, 13, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

position t: the model has seen x[0..t] -> it must predict y[t]
  t= 0: seen .                 -> must predict 'e'
  t= 1: seen .e                -> must predict 'm'
  t= 2: seen .em               -> must predict 'm'
  t= 3: seen .emm              -> must predict 'a'
  t= 4: seen .emma             -> must predict '.'
  t= 5: seen .emma.            -> y=-1  (padding: loss ignores this position)
  ...
=== 4. the same names file contains repeated lines
training words: 31033 | distinct training words: 28654
153 of 1000 test words also appear in the training set, e.g. ['willow', 'masiah', 'kobie', 'knowledge', 'winn', 'everette']
```

**Decode it:**

| Line | Meaning |
|---|---|
| `{'a': 1, ...}` | The `stoi` table. No entry maps to 0. |
| `x = [0, 5, 13, 13, 1, 0, ...]` | Start token, then e-m-m-a, then padding zeros. |
| `y = [5, 13, 13, 1, 0, -1, ...]` | e-m-m-a, then the **end** token 0, then `-1` (ignored). |
| `seen .emma -> must predict '.'` | The lab prints token 0 as `.`. That is only a display choice in the lab; `makemore.py` has no `.` character. |
| `distinct training words: 28654` | 31,033 − 28,654 = 2,379 training lines are second copies of a name already in the training set. |
| `153 of 1000 test words also appear...` | The overlap described above. |

The first block of output (`number of examples in the dataset: 32033` …) is `create_datasets` printing, exactly as in lesson 0.

---

## What counts as success

| You can... | Test |
|---|---|
| Build `x` and `y` by hand | Do question 1 below without running code |
| Say what token 0 means in `x` vs in `y` | Question 2 |
| Explain why padding can't leak into predictions | Question 3 |
| Explain why the test set isn't fully "unseen" | Question 4 |

## Check yourself

1. Write `x` and `y` for the name `ava` (`a=1, v=22`).
2. Token `0` appears in both `x` and `y`. What does it mean in each?
3. Padding positions in `x` hold `0`. Why can't the model "cheat" by using them?
4. Someone says "test loss is measured on names the model has never seen." Is that true for this repo?
5. You train on a file whose longest line is 40 characters. What is `block_size`?

<details><summary>Answers</summary>

1. `x = [0, 1, 22, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` and `y = [1, 22, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]`. Four counted questions: start→a, a→v, v→a, a→end.
2. In `x` at position 0 it means "start of name". In `y` it means "end of name, stop here". (In `x` after the name it is just filler.)
3. Two reasons. Their targets are `-1`, so they never count in the loss. And every model only lets position `t` see positions `0..t`. Padding comes after the name, so no counted position can see it.
4. Not fully. 153 of the 1,000 test names (15.3%) also occur in the training set, because `names.txt` lists 2,539 names twice and the split shuffles lines.
5. 41.

</details>

Next: [Lesson 2 — the Bigram model and the loss](02-bigram-and-loss.md).
