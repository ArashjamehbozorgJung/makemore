# Lesson 2 — The Bigram model: logits, softmax, and the loss

**This lesson promises:** you will know what the number printed as `loss` is, compute it by hand, and say why it starts near 3.30. You will also see that "training" for the simplest model gives exactly the same answer as counting.

**Code covered:** `Bigram` (lines 396–423), and the loss line shared by all six models (e.g. line 421).
**Lab:** `python course/labs/lab02_bigram.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **bigram** | A pair of adjacent tokens, like `e→m`. A bigram model predicts the next token from **only the one token before it**. |
| **logit** | A raw score, any real number. One per possible next token. Bigger means "more likely". |
| **`exp(x)`**, written eˣ | e ≈ 2.718 raised to the power x. Always positive. `exp(0) = 1`. It grows fast: `exp(2) ≈ 7.39`. |
| **`log(x)`**, natural log | The inverse of `exp`: `log(exp(x)) = x`. `log(1) = 0`. For x between 0 and 1 the log is negative, and `log(x)` → −∞ as x → 0. |
| **probability distribution** | A list of non-negative numbers that add up to 1. Here: 27 numbers, one per possible next token. |
| **softmax** | `softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`. Turns any list of logits into a probability distribution. |
| **negative log-likelihood (NLL)** | `−log(p)`, where p is the probability the model gave to the **correct** answer. 0 if p = 1, large if p is small. |
| **cross-entropy loss** | The average NLL over all counted positions. This is the `loss` that `makemore.py` prints. |
| **nat** | The unit of a loss computed with the natural log. (With log base 2 the unit would be bits; 1 nat ≈ 1.443 bits.) |
| **perplexity** | `exp(loss)`. A loss of L means the model is as uncertain, on average, as a uniform choice among `exp(L)` options. |
| **parameter (`nn.Parameter`)** | A tensor that PyTorch marks as "trainable": the optimizer will change it. |

---

## 1. The whole model is a 27 × 27 table

```python
class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))   # 27 x 27, all zeros

    def get_block_size(self):
        return 1  # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):
        logits = self.logits[idx]   # 'forward pass', lol
        ...
```

- **Row** `i` of the table holds the 27 logits for "what comes after token `i`".
- `self.logits[idx]` is plain indexing. `idx` has shape `(B, T)` (batch × positions). For every integer in `idx`, PyTorch fetches that row of 27 numbers. The result has shape `(B, T, 27)`. That is the entire "forward pass", hence the author's "lol".
- 27 × 27 = **729 parameters**. That matches `model #params: 729` when you run `--type bigram`.

## 2. From logits to probabilities: softmax

Logits can be negative and needn't sum to anything. Probabilities must be ≥ 0 and sum to 1. Softmax does both:

1. `exp` each logit. Every result is now positive.
2. Divide each by their sum. Now they sum to 1.

Example (lab section 1): logits `[2.0, 1.0, 0.1]` → `exp` → `[7.389, 2.718, 1.105]` → sum 11.212 → probabilities `[0.659, 0.242, 0.099]`.

One property used later: **adding the same number to every logit doesn't change the result.** All-zero logits therefore mean "every token equally likely": `exp(0)/27 = 1/27` each.

## 3. The loss: lines 419–421 (identical in all six models)

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

What it computes, step by step:

1. `logits.view(-1, 27)` reshapes `(B, T, 27)` into `(B·T, 27)`: one row of 27 scores per position. `targets.view(-1)` reshapes `(B, T)` into `(B·T,)`: one correct answer per position. (`view` changes the shape without copying or changing the numbers.)
2. Rows whose target is `-1` are **dropped**. That is the padding mask from lesson 1.
3. For each remaining row: softmax, then take the probability `p` of the correct answer, then compute `−log(p)`.
4. Average over the remaining rows.

So **`loss` = the average, over every real prediction in the batch, of −log(probability given to the right answer).**

| The model gave the correct answer probability | −log p |
|---|---|
| 1.0 (certain and right) | 0 |
| 0.5 | 0.693 |
| 1/27 ≈ 0.037 (uniform guess) | **3.296** |
| 0.01 | 4.605 |
| → 0 (certain and wrong) | → ∞ |

### Why the first loss of the Bigram is exactly 3.2958

All logits start at 0, so every probability is 1/27, and the loss is `−log(1/27) = log(27) = 3.2958`. You saw `step 0 | loss 3.2958` in lesson 0's bigram run. That is no coincidence. It is this formula.

The Transformer's first loss was **3.5123**, worse than a uniform guess. Its parameters start random, not zero. Random logits are not equal, so the model starts out confidently wrong in random directions. Anything above 3.296 at step 0 means "worse than knowing nothing".

### Reading a loss as a number of choices

`exp(loss)` is the **perplexity**. `exp(3.296) = 27`: as unsure as picking among all 27 tokens. `exp(2.454) ≈ 11.6`. `exp(1.966) ≈ 7.1`. So the trained Transformer from lesson 0 is, on average, as uncertain as a fair choice among about 7 characters.

## 4. The best possible Bigram, without any training

For a lookup table, the probabilities that give the **lowest possible average −log p on a dataset** are the observed frequencies:

```
P[a, b] = (number of times b followed a) / (number of times anything followed a)
```

This is a standard result (the maximum-likelihood estimate; the proof is a few lines of calculus I'll skip). So you can compute the best Bigram just by counting pairs. The lab does exactly that, on the training set, including the start→first-letter and last-letter→end pairs. Result: **training loss 2.4542. No Bigram of any kind can go lower on this training set.**

Then the lab trains the `Bigram` class by gradient descent (lesson 3 explains how) and gets **2.4543**. The learned probabilities match the counted ones to within 0.001 in every one of the 729 cells.

That is the key insight of this lesson. **Training is a method for finding parameter values that minimize the loss.** For the Bigram you could skip it and just count. For every other model in this file no such shortcut exists, so you need training. But it is the same game: lower the average −log p.

> **Careful:** 102 of the 729 pairs never occur in the training set, e.g. `q→q`. Raw counts give them probability 0. If such a pair then appears in a test name, −log 0 = ∞ and the test loss is infinite. The lab adds 1 to every count ("add-one smoothing") before measuring the test set. The neural Bigram never has this problem: softmax output is never exactly 0, because `exp` of any finite number is positive.

---

## Lab 2

```bash
python course/labs/lab02_bigram.py
```

**What you should see** (real output; dataset lines omitted):

```
=== 1. softmax by hand: logits (any real numbers) -> probabilities (positive, sum to 1)
logits          : tensor([2.0000, 1.0000, 0.1000])
probs by hand   : tensor([0.6590, 0.2424, 0.0986]) sum = 0.9999999403953552
F.softmax       : tensor([0.6590, 0.2424, 0.0986])

=== 2. cross-entropy by hand: -log(probability the model gave to the correct answer)
correct answer = 0: p = 0.6590, -log p = 0.4170, F.cross_entropy = 0.4170
correct answer = 2: p = 0.0986, -log p = 2.3170, F.cross_entropy = 2.3170

=== 3. the Bigram model before any training
its only parameter: model.logits, shape (27, 27) - all zeros at the start
input X shape (32, 16) -> output logits shape (32, 16, 27)
loss at init = 3.2958   ln(27) = 3.2958   (uniform guess over 27 tokens)

=== 4. the best possible bigram, computed by counting (no neural net, no gradients)
count-based bigram, train loss: 2.4542  <- the lowest loss ANY bigram can reach on this training set
count-based bigram (add-one smoothed), test loss: 2.4497
pairs never seen in training: 102 of 729

=== 5. train the Bigram 'neural net' by gradient descent and watch it reach the counting answer
step   0 | train loss 3.2958
step 100 | train loss 2.4555
...
step 500 | train loss 2.4543
largest |learned prob - counted prob| over all 27x27 entries: 0.0010
row for <START> (which letter starts a name), counted vs learned, first 6 letters a..f:
  counted: tensor([0.1382, 0.0407, 0.0485, 0.0523, 0.0478, 0.0131])
  learned: tensor([0.1382, 0.0407, 0.0485, 0.0523, 0.0478, 0.0131])
```

**Decode it:**

| Line | Meaning |
|---|---|
| `sum = 0.9999999403953552` | Not exactly 1, because computers store decimals with limited precision (32-bit floats keep about 7 significant digits). Normal. |
| `-log p = 0.4170, F.cross_entropy = 0.4170` | PyTorch's loss is exactly the hand formula. |
| `(32, 16) -> (32, 16, 27)` | 32 names × 16 positions in; 27 scores per position out. |
| `count-based ... train loss: 2.4542` | The floor for any Bigram on this training set. |
| `test loss: 2.4497` | Slightly *lower* than the training number. Possible because the test set is a different, small sample of 1,000 names. It is not a bug. |
| `step 500 \| train loss 2.4543` | Gradient descent on the whole training set reaches the counting floor. (Section 5 uses a large learning rate, 0.1, so this happens in 500 steps. `makemore.py`'s default rate is 0.0005. Lesson 3 shows why that is much slower.) |
| `counted: 0.1382 ... learned: 0.1382` | 13.8% of names start with `a`. Both methods agree. |

---

## What counts as success

| You can... | Test |
|---|---|
| Compute a softmax and a cross-entropy by hand for 3 logits | Question 1 |
| Explain 3.2958 | Question 2 |
| Explain why the Transformer starts above 3.2958 | Question 3 |
| State what the Bigram cannot learn | Question 4 |

## Check yourself

1. Logits `[0, 0, log 2]` (log 2 ≈ 0.693), correct answer is index 2. What is the loss?
2. A model with vocabulary size 100 outputs all-zero logits. What loss does it report?
3. Lesson 0's Transformer started at 3.5123. Is it broken?
4. Can a Bigram learn that names ending in `-lyn` are common? Why or why not?
5. Why does the loss line contain `ignore_index=-1`? What would go wrong without it?

<details><summary>Answers</summary>

1. `exp` gives `[1, 1, 2]`, which sums to 4, so the probabilities are `[0.25, 0.25, 0.5]`. Loss = −log 0.5 = **0.693**.
2. `log(100) ≈ 4.605`.
3. No. Its random starting logits are unequal, which is worse than uniform until training fixes it. It drops below 3 within 10 steps.
4. No. Predicting `n` after `-ly` needs **two** previous characters. A Bigram only sees `y`, so it can only learn "how often `n` follows `y` across all names". This limit is why its best loss is 2.454 while the other models reach about 2.0.
5. Without it, the call fails: `-1` isn't a valid class index. I checked: PyTorch raises `IndexError: Target -1 is out of bounds.` (If padding targets were instead set to `0`, the model would be trained to predict "end" over and over after every name, and the loss would be dominated by these trivial positions.)

</details>

Next: [Lesson 3 — how training changes the parameters](03-training.md).
