# Lesson 7 — Bag of Words: averaging the past

**This lesson promises:** you will see how a masked softmax of zeros computes "the average of everything so far" in one matrix multiplication. That exact mechanism, with one change, is the attention of lesson 8. You will also learn the residual connection.

**Code covered:** `CausalBoW`, `BoWBlock`, `BoW` (lines 158–251).
**Lab:** `python course/labs/lab07_bow.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **matrix multiplication (`@`)** | For `A` of shape `(T, T)` and `x` of shape `(T, C)`: `(A @ x)[t] = Σₛ A[t, s] · x[s]`. Row `t` of the result is a weighted sum of the rows of `x`, with weights from row `t` of `A`. With a leading batch dimension, `@` does this separately for each batch element. |
| **lower-triangular matrix** | A square matrix whose entries above the diagonal (column > row) are 0. `torch.tril` makes one. |
| **causal mask** | A lower-triangular pattern used to stop position `t` from using positions after `t`. |
| **`masked_fill(cond, value)`** | Returns a copy of a tensor with `value` written wherever `cond` is true. |
| **position embedding (`wpe`)** | An embedding table indexed by *position* (0–15) instead of by token. It gives the model a learned vector meaning "I am at position t". |
| **residual connection** | `x = x + f(x)`: a layer's output is *added* to its input instead of replacing it. |
| **buffer (`register_buffer`)** | A tensor stored inside a module and saved in `model.pt`, but **not** a parameter: the optimizer never changes it. |

---

## 1. The idea

The MLP gave each window slot its own weights. The RNN squeezed the past into one state, step by step. The BoW model does the simplest possible thing: at position `t`, **average the vectors of positions 0..t**, then process that average.

"Bag of words" is the old name for representations that ignore order, like a bag where the words are just tossed in. Averaging ignores order, except for what position embeddings put back.

## 2. `CausalBoW.forward`: lines 174–183

```python
att = torch.zeros((B, T, T), device=x.device)
att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)
```

`self.bias` is the lower-triangular matrix of ones from line 171, stored as a buffer. **Despite its name, it is a mask, not a bias.** (The name comes from minGPT/GPT-2 code.)

Watch it with `T = 5`, as in the lab:

```
zeros                 mask with -inf above diagonal      softmax along each row
0 0 0 0 0             0  -inf -inf -inf -inf             1.000 0     0     0     0
0 0 0 0 0             0  0    -inf -inf -inf             0.500 0.500 0     0     0
0 0 0 0 0      ->     0  0    0    -inf -inf     ->      0.333 0.333 0.333 0     0
0 0 0 0 0             0  0    0    0    -inf             0.250 0.250 0.250 0.250 0
0 0 0 0 0             0  0    0    0    0                0.200 0.200 0.200 0.200 0.200
```

Why softmax produces exactly this: in row `t`, the allowed entries are all 0, so `exp(0) = 1` each, and the masked ones are `exp(−∞) = 0`. Dividing by the sum, `t+1`, gives `1/(t+1)` for each allowed position.

Then `y = att @ x`. Row `t` of `y` = `Σₛ att[t,s] · x[s]` = **the average of `x[0..t]`**. The lab checks this against a direct running mean: largest difference `1.19e-07`, i.e. rounding.

Why build an average in this roundabout way, instead of `x.cumsum(1) / counts`? Because the next lesson replaces the `zeros` with **scores computed from the data**. Everything else in these four lines stays. That is the "suspiciously similar" the docstring winks at.

## 3. `BoWBlock`: lines 185–204, and the residual connection

```python
def forward(self, x):
    x = x + self.cbow(x)     # add the average of the past to each position's own vector
    x = x + self.mlpf(x)     # add an MLP's output: Linear 64->64, tanh, Linear 64->64
    return x
```

`x = x + f(x)` is a **residual connection**. Two things it does, mechanically:

1. **Nothing is overwritten.** Position `t` keeps its own vector `x[t]` and gets the average *added* to it. Without the `x +`, the current character would be just one of `t+1` equal contributors to an average. With it, the current character always comes through at full strength.
2. **Gradients get a direct path.** The derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`. The `1` passes the gradient backwards unchanged, whatever `f` does. This is the same reason the GRU's `z ≈ 0` path helped (lesson 6). It becomes essential when you stack many blocks (lesson 8).

## 4. `BoW.forward`: lines 229–251

```python
tok_emb = self.wte(idx)       # (b, t, 64): what each character is
pos_emb = self.wpe(pos)       # (1, t, 64): where each character is; broadcast over the batch
x = tok_emb + pos_emb         # each vector now carries "what" and "where", added together
x = self.context_block(x)     # one BoWBlock
logits = self.lm_head(x)      # (b, t, 27)
```

- **Why `wpe` is needed:** an average is the same whatever the order. "ab" and "ba" average to the same vector. Adding a position vector to each token *before* averaging makes `a`-at-0 differ from `a`-at-1, so some order information survives.
- `pos_emb` has shape `(1, t, 64)` and `tok_emb` `(b, t, 64)`. PyTorch **broadcasts**: it repeats the size-1 dimension to match, so every name in the batch gets the same position vectors.

Parameters (lab section 3): `wte` 1,728 + `wpe` 1,024 + two `Linear(64,64)` 8,320 + `lm_head` 1,755 = **12,827**. `CausalBoW` itself has **zero** parameters. Its weights are fixed at `1/(t+1)` and never learned.

## 5. Why it starts at loss 4.15

The lab reports `loss at init 4.1517`, far above the uniform 3.2958. I measured why. Embedding tables start with random values of standard deviation 1. Adding token + position gives vectors with std about 1.43. The residual block adds more on top: std 2.18. Nothing rescales them before `lm_head`, so the starting logits are spread out: per-position std 1.26, versus 0.23–0.57 for the other models. Spread-out random logits mean confident random guesses, and the loss is high. Training fixes it fast: in my run the batch loss was 3.65 at step 10 and 2.99 at step 50. The Transformer avoids the problem with a LayerNorm right before `lm_head` (lesson 8).

## 6. What it achieved, and why it is limited

20,000 steps: best test loss **2.1530**. That is better than the Bigram (2.4495), worse than the MLP (2.0304) and the RNN (2.0539).

The limits are mechanical:
- **Fixed, equal weights.** At position 10, the character 10 positions back counts exactly as much in the average as the one just before. But the last 2–3 characters matter most for the next letter. The model can't choose what to look at.
- **One averaging step only.** There is one `BoWBlock`, not a stack.

Fixing the first limit means letting each position compute *its own* weights from the content. That is attention.

---

## Lab 7

```bash
python course/labs/lab07_bow.py
```

**What you should see** (real output; dataset lines omitted): the three 5 × 5 matrices shown in section 2, then:

```
=== 2. so att @ x is the running mean of x over positions 0..t
max |CausalBoW(x) - running mean| = 1.1920928955078125e-07

=== 3. the full BoW model
  wte.weight                       (27, 64)     1728
  wpe.weight                       (16, 64)     1024
  context_block.mlp.c_fc.weight    (64, 64)     4096
  context_block.mlp.c_fc.bias      (64,)          64
  context_block.mlp.c_proj.weight  (64, 64)     4096
  context_block.mlp.c_proj.bias    (64,)          64
  lm_head.weight                   (27, 64)     1728
  lm_head.bias                     (27,)          27
  total: 12827
input (32, 16) -> logits (32, 16, 27) | loss at init 4.1517
note: CausalBoW has no parameters. the averaging weights are fixed, not learned.
```

**Decode it:**

| Line | Meaning |
|---|---|
| `1.19e-07` | Matches the running mean up to float rounding. |
| `wpe.weight (16, 64)` | One learned vector per position 0–15. This is why `block_size` matters: there is no row for position 16. |
| no `cbow` line | `CausalBoW` has only a buffer (the mask), which `named_parameters` doesn't list. |
| `loss at init 4.1517` | Explained in section 5. |

---

## What counts as success

| You can... | Test |
|---|---|
| Produce the softmax matrix for any `T` by hand | Question 1 |
| Explain what `att @ x` computes, row by row | Question 2 |
| Explain both effects of a residual connection | Question 3 |
| Say exactly what attention will change | Question 4 |

## Check yourself

1. Write row 2 (third row) of `att` for `T = 4` after the softmax.
2. If instead of zeros, row 3 of the scores before masking were `[0, 0, 0, log 3]`, what would row 3 of `att` be?
3. Remove the `x +` from `x = x + self.cbow(x)`. What does position 0 receive? Position 9?
4. In one sentence: what will attention change in these four lines of `CausalBoW.forward`?
5. Why can't the BoW model (or the Transformer) process a sequence of 17 tokens?

<details><summary>Answers</summary>

1. `[1/3, 1/3, 1/3, 0]`.
2. `exp` gives `[1, 1, 1, 3]`, which sums to 6, so `[1/6, 1/6, 1/6, 1/2]`. Position 3 now attends to itself three times as much as to each earlier position. Non-equal scores give non-equal weights.
3. Position 0 receives the average of `x[0]`, which is just `x[0]`. Position 9 receives the average of 10 vectors, so its own character contributes only 1/10. The current character gets diluted.
4. The `zeros` become `q·k` scores computed from each position's content, so the averaging weights become data-dependent and learned.
5. `wpe` has 16 rows, for positions 0–15. Position 16 has no embedding. The `assert t <= self.block_size` on line 233 stops it before the lookup fails.

</details>

Next: [Lesson 8 — the Transformer](08-transformer.md).
