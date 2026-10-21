# Lesson 4 — Making new names: `generate`, temperature, top-k

**This lesson promises:** you can trace how a trained model produces a name one character at a time. You can predict what temperature and top-k do, and read the "in train / in test / new" report correctly.

**Code covered:** `generate` (lines 428–457) and `print_samples` (lines 459–485).
**Lab:** `python course/labs/lab04_sampling.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **sampling** | Picking an outcome at random, where each outcome's chance equals its probability. If `a` has probability 0.138, it is picked about 13.8% of the time. |
| **`torch.multinomial(probs, 1)`** | PyTorch's function that does one such random pick per row of `probs`. |
| **greedy decoding** | Always picking the single most likely token instead of sampling. |
| **temperature** | A number `T` that every logit is divided by before softmax. `T < 1` makes the distribution sharper. `T > 1` makes it flatter. |
| **top-k** | Keep only the `k` highest logits. Set the rest to −∞, so softmax gives them probability exactly 0. |
| **`@torch.no_grad()`** | A decorator that turns off gradient recording inside the function. Generating needs no training, so this saves time and memory. |
| **inference** | Using a trained model to make predictions, as opposed to training it. |

---

## 1. The generation loop: lines 436–455

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]   # (a)
    logits, _ = model(idx_cond)                                             # (b)
    logits = logits[:, -1, :] / temperature                                 # (c)
    if top_k is not None:                                                   # (d)
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)                                       # (e)
    if do_sample:
        idx_next = torch.multinomial(probs, num_samples=1)                  # (f)
    else:
        _, idx_next = torch.topk(probs, k=1, dim=-1)
    idx = torch.cat((idx, idx_next), dim=1)                                 # (g)
```

One pass of the loop adds one token to every row:

- **(a)** Keep at most the last `block_size` tokens. The models can't take longer inputs: the Transformer and BoW have position tables of size 16, and the Bigram's `get_block_size()` is 1. So the Bigram is fed only the last token.
- **(b)** Run the model on the whole sequence so far. It returns logits for **every** position, shape `(B, T, 27)`.
- **(c)** Keep only the **last** position, `[:, -1, :]`. That is the prediction for the next token. Divide by temperature.
- **(d)** Optionally knock out everything except the top `k`.
- **(e)** Softmax: logits → probabilities.
- **(f)** Draw one token per row at random. If `do_sample=False`, take the most likely one instead.
- **(g)** Append it. The next pass sees it as input. This feeding back of outputs as inputs is what **autoregressive** means.

> **Careful:** (b) recomputes the model over the *entire* prefix on every pass, then throws away all but the last position. For 16-token names that waste doesn't matter. For long text it does, and real systems cache intermediate results instead. This code doesn't.

## 2. Temperature, mechanically

Dividing logits by `T` before softmax:

- `T = 1`: unchanged.
- `T → 0`: differences between logits get multiplied up. The top token's probability goes to 1. The result approaches greedy decoding.
- `T → ∞`: all logits go to 0, so the distribution goes to uniform: random letters.

The lab shows it on the "first letter of a name" distribution:

```
T=0.5: top-5 a=0.298, k=0.132, m=0.097, j=0.089, s=0.064
T=1.0: top-5 a=0.138, k=0.092, m=0.079, j=0.075, s=0.064
T=2.0: top-5 a=0.080, k=0.065, m=0.060, j=0.059, s=0.054
```

> **Careful:** `generate` accepts `temperature`, but `print_samples` never passes it, and there is no `--temperature` command-line flag. From the command line you always sample at `T = 1.0`. To change it, edit line 464 (a good first modification; see lesson 9).

## 3. Top-k, mechanically

```python
v, _ = torch.topk(logits, top_k)            # the k largest values, sorted, per row
logits[logits < v[:, [-1]]] = -float('Inf') # anything below the k-th largest -> -inf
```

`exp(−∞) = 0`, so those tokens get probability exactly 0. The surviving `k` share all the probability. With `--top-k 3` at the start of a name, only `a`, `k`, `m` can be chosen (lab: `{'a': 0.447, 'k': 0.298, 'm': 0.255}`).

What it buys you: rare, odd characters can never be picked. What it costs: less variety. The lab's top-k=3 Bigram names were `a, ari, kin, a, ma, ma, mian, a, men, an`. Plausible-looking, but short and repetitive.

## 4. Greedy decoding is worse than you'd think

`do_sample=False` takes the most likely token every time. The lab's greedy Bigram output is the name **`a`**, three times. Why: the most likely first letter is `a` (13.8%), and after `a` the most likely next token is **end** (many names end in `a`). Each step is locally the best, and the result is a useless name. Sampling is how you get variety.

## 5. `print_samples`: lines 459–485

```python
X_init = torch.zeros(num, 1, dtype=torch.long)       # every row starts with token 0 = <START>
steps = train_dataset.get_output_length() - 1          # 16 - 1 = 15 new tokens
X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True)
...
row = X_samp[i, 1:].tolist()                          # drop the <START> token
crop_index = row.index(0) if 0 in row else len(row)   # cut at the first <STOP> token
row = row[:crop_index]
```

- It always generates 15 tokens, even after an end token appears. Everything after the first `0` is thrown away. (The model is never trained on what follows an end token, because those targets are `-1`, so that part is garbage anyway.)
- Then each name is filed into one of three lists:

```python
if train_dataset.contains(word_samp):   train_samples.append(...)
elif test_dataset.contains(word_samp):  test_samples.append(...)
else:                                   new_samples.append(...)
```

> **Careful — reading the three lists:**
> - **"in train"** is checked first. A name in both train and test is counted as "in train". Only 840 distinct names exist in the test set alone, against 28,654 distinct training names. That is why "in test" is almost always 0. It is not a sign that something is broken.
> - **"in train" is not proof of memorization.** `lila` or `nix` are short, common, high-probability names. A model that never saw them would still produce them often. Memorization shows up in the train/test loss gap (lesson 9), not here.
> - **"new"** includes junk like `brsen`. New only means "not in the file".

My 20,000-step Transformer gave 17 "in train", 0 "in test", 33 "new" out of 50.

---

## Lab 4

```bash
python course/labs/lab04_sampling.py
```

The lab skips training. It sets the Bigram's logits to `log(count + 1)`. Softmax of `log(c)` equals `c / Σc`, so the model's probabilities become exactly the smoothed counted frequencies from lesson 2.

**What you should see** (real output; dataset lines omitted):

```
=== 1. the distribution over the first letter (row <START>), at different temperatures
T=0.5: top-5 a=0.298, k=0.132, m=0.097, j=0.089, s=0.064
T=1.0: top-5 a=0.138, k=0.092, m=0.079, j=0.075, s=0.064
T=2.0: top-5 a=0.080, k=0.065, m=0.060, j=0.059, s=0.054
lower T sharpens the distribution toward the favourite; higher T flattens it toward uniform

=== 2. top-k: keep the k largest logits, set the rest to -inf, so they get probability 0
nonzero probabilities with top_k=3: {'a': 0.447, 'k': 0.298, 'm': 0.255}

=== 3. greedy decoding (do_sample=False): always take the single most likely next token
['a', 'a', 'a'] <- identical every time, and a poor name

=== 4. sampling (do_sample=True): draw the next token at random from the distribution
raw output row 0: [0, 1, 19, 1, 12, 1, 9, 4, 1, 14, 14, 5, 14, 9, 14, 0]
shown with '.' for token 0: .asalaidannenin.
decoded names: ['asalaidannenin', 'j', 'ffaylazora', 'gan', 'de', 'ein', 'n', 'vurona', 'test', 'jaimel']

with top_k=3:
['a', 'ari', 'kin', 'a', 'ma', 'ma', 'mian', 'a', 'men', 'an']
```

**Decode it:**

| Line | Meaning |
|---|---|
| `raw output row 0: [0, 1, 19, ...]` | 16 integers: the start token plus 15 generated. |
| `.asalaidannenin.` | Here the end token happened to be the 15th generated token. |
| `asalaidannenin`, `ffaylazora` | Typical Bigram output: each letter pair is plausible, but the whole is not a name. A Bigram has no memory beyond one letter, so it can't track length or structure. Compare the Transformer's `lomira`, `noren` in lesson 0. |
| `test` | An accidental English word. |

Your section 4 names will be identical to mine only if your PyTorch version produces the same random numbers. The *kind* of output will be the same.

---

## What counts as success

| You can... | Test |
|---|---|
| Trace one pass of the loop | Question 1 |
| Predict temperature's effect | Question 2 |
| Explain why greedy gives `a` | Question 3 |
| Read the three sample lists without over-interpreting | Question 4 |

## Check yourself

1. During generation, the Transformer returns logits of shape `(10, 7, 27)`. How many tokens has each row generated so far (not counting the start token)? Which slice is used?
2. What happens to the samples at `T = 0.01`? At `T = 100`?
3. Why can greedy decoding produce a name that sampling would rarely produce?
4. A model's samples are "30 in train, 20 new". A colleague says "it memorized 60% of its output". Respond.
5. Why does `generate` feed only the last token to the Bigram, but up to 16 tokens to the Transformer?

<details><summary>Answers</summary>

1. 7 positions = start token + 6 generated. Only `logits[:, -1, :]`, the 7th position, is used to choose token 7.
2. `T = 0.01`: practically greedy, so nearly every sample is the same few names. `T = 100`: all probabilities nearly 1/27, so random letter soup of random length.
3. Greedy takes the top token at *each* step. The resulting sequence can have low probability as a whole, and the name `a` is in fact rare. Sampling follows the overall distribution.
4. "In train" means the output matches a training name, not that it was copied. Common names are high-probability for any good model, including one that never saw them. To measure memorization, compare train and test loss, or check names the model could only know from training.
5. `idx_cond` crops to `model.get_block_size()`. The Bigram's returns 1. The Transformer's returns 16, the size of its position table.

</details>

Next: [Lesson 5 — the MLP](05-mlp.md).
