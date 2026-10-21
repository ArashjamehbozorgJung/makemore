# Lesson 5 — The MLP: embeddings and a window of context

**This lesson promises:** you will understand embeddings, linear layers and nonlinearities, which are the building blocks of every remaining model. You will also understand the unusual `torch.roll` loop that gives the MLP its view of the past.

**Code covered:** `MLP`, lines 347–394.
**Lab:** `python course/labs/lab05_mlp.py`
**Paper:** Bengio et al. 2003, "A Neural Probabilistic Language Model" (linked in the code). Optional: Karpathy's lecture "Building makemore Part 2: MLP" builds a similar model in a notebook. That version is not identical to this file.

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **vector** | A list of numbers, e.g. 64 numbers. |
| **embedding table (`nn.Embedding(n, d)`)** | A trainable `n × d` table. Looking up integer `i` returns row `i`, a vector of `d` numbers. Mechanically it is exactly the Bigram's `self.logits[idx]` lookup, except the rows are a vector of features, not next-token logits. |
| **embedding** | The vector looked up for a token. Training adjusts these vectors. |
| **linear layer (`nn.Linear(a, b)`)** | Holds a weight matrix `W` of shape `(b, a)` and a bias vector of length `b`. It maps a vector `x` of length `a` to `W·x + bias`, of length `b`. Each output is a weighted sum of all inputs plus a constant. Parameters: `a·b + b`. |
| **`tanh`** | The function `tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`, applied to each number separately. Output is always between −1 and 1. Near 0 it is close to `x`. Large inputs flatten out to ±1. |
| **nonlinearity / activation** | A function like `tanh` applied between linear layers. |
| **hidden layer** | The vector between the input and the output layers. Here 64 numbers (`n_embd2`). |
| **concatenate (`torch.cat`)** | Join vectors end to end. Three vectors of 64 become one of 192. |
| **context window** | The fixed number of previous tokens the model looks at. |
| **MLP (multi-layer perceptron)** | Linear layer → nonlinearity → linear layer (→ …). The simplest neural network that isn't just one linear map. |

---

## 1. Why the Bigram hits a wall

The Bigram's best possible loss is 2.454, because it sees only one previous character. To use more context with a lookup table, you would need one row per combination of previous characters: 27² = 729 rows for two characters, 27³ = 19,683 for three, more than 10²² for 16. Most of those rows would never occur in the data, so they could never be learned.

The MLP's fix: don't give each *combination* its own row. Give each *character* a learned vector, and let a small neural network combine the vectors of the last several characters.

## 2. The parts: lines 359–370

```python
self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)   # 28 x 64 table
self.mlp = nn.Sequential(
    nn.Linear(self.block_size * config.n_embd, config.n_embd2),  # 16*64 = 1024 -> 64
    nn.Tanh(),
    nn.Linear(config.n_embd2, self.vocab_size)                   # 64 -> 27 logits
)
```

- `vocab_size + 1 = 28` rows. Rows 0–26 are the normal tokens. **Row 27 is a special `<BLANK>` token** that only the MLP has. It means "there is nothing here: this window slot is before the start of the sequence".
- `block_size = 16` here, because `makemore.py` passes the dataset's `block_size` to every model. So the MLP's window is 16 characters, and since names are at most 15 letters plus start, **it always sees the entire name so far.**
- `nn.Sequential` just applies its layers in order.

Parameter count (lab section 2):

| Part | Shape | Count |
|---|---|---|
| `wte` | 28 × 64 | 1,792 |
| first `Linear` weight + bias | 64 × 1024 + 64 | 65,600 |
| second `Linear` weight + bias | 27 × 64 + 27 | 1,755 |
| **total** | | **69,147** |

95% of the parameters are in the first layer: it has a separate weight for every (window slot, embedding feature, hidden unit) combination.

## 3. The `torch.roll` loop: lines 377–386

This is the strangest code in the file. Read it slowly.

```python
embs = []
for k in range(self.block_size):
    tok_emb = self.wte(idx)              # embed the current version of idx: (b, t, 64)
    idx = torch.roll(idx, 1, 1)          # shift every row one place to the right (wrapping around)
    idx[:, 0] = self.vocab_size          # overwrite the wrapped-around first column with <BLANK>
    embs.append(tok_emb)
x = torch.cat(embs, -1)                  # (b, t, 64 * 16)
```

`torch.roll(idx, 1, 1)` shifts every row right by one along dimension 1. The last element wraps around to the front. Then column 0 is overwritten with `<BLANK>` (27), so the wrapped value is discarded.

The lab replays it with a window of 3 on `.emma` (`.` = token 0, `_` = `<BLANK>`):

```
k=0: idx = [0, 5, 13, 13, 1]   shown as '.emma'
k=1: idx = [27, 0, 5, 13, 13]  shown as '_.emm'
k=2: idx = [27, 27, 0, 5, 13]  shown as '__.em'
```

Now read **down a column**. Column `t` lists the token at `t`, the one before it, and the one before that:

```
position 0: oldest..newest = '__.'
position 1: oldest..newest = '_.e'
position 2: oldest..newest = '.em'
position 3: oldest..newest = 'emm'
position 4: oldest..newest = 'mma'
```

So after the loop, at every position `t`, `x[:, t, :]` is 16 embeddings joined end to end. In order: the token at `t`, the one at `t−1`, …, down to `t−15`, with `<BLANK>` wherever that reaches before position 0. All positions of all names are built at once, with no Python loop over positions.

> **Careful:** the comment on line 377 says "gather the word embeddings of the previous 3 words". It is stale. The window is `block_size` = 16 **characters**. Bengio's original paper used words, and the comment wasn't updated.

**Why `<BLANK>` instead of 0?** Token 0 means "start of name" and sits at position 0. The slots *before* position 0 need a different marker. Otherwise the model couldn't tell "start token, two positions back" from "nothing, two positions back".

**Why the window can't see the future:** every rolled copy only moves tokens **rightwards**, to later positions. Position `t` only ever receives tokens from positions ≤ `t`. Lab 9 confirms it numerically.

## 4. The forward pass in shapes

```
idx                 (32, 16)          integers
16 × wte lookup     16 × (32, 16, 64)
torch.cat           (32, 16, 1024)    16 embeddings side by side, per position
Linear 1024→64      (32, 16, 64)
tanh                (32, 16, 64)      every number squashed into (−1, 1)
Linear 64→27        (32, 16, 27)      logits
cross_entropy       one number        exactly as in lesson 2
```

`nn.Linear` acts on the **last** dimension only. It processes all 32 × 16 positions independently in one call.

## 5. Why the `tanh` is essential

Without it, the model would be `Linear(Linear(x))`. A linear map of a linear map is just another linear map: `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`. So two linear layers without a nonlinearity can compute nothing that one couldn't. The `tanh` between them lets the network compute "if these characters together, then that". For example, "`l` then `y` → `n` is likely" can be learned as a combination, not just as a sum of separate effects of `l` and `y`.

## 6. What it achieved

From my 20,000-step runs (lesson 9 has the full table): Bigram test loss 2.4495, **MLP 2.0304**, still improving at the end. Using the whole name instead of one character was worth about 0.42 nats per character.

The MLP's weakness: each window slot has its **own** block of weights in the first layer. What it learns about "`a` one position back" is stored separately from "`a` two positions back". It doesn't share knowledge across positions. The RNN (lesson 6) and the Transformer (lesson 8) handle this differently.

---

## Lab 5

```bash
python course/labs/lab05_mlp.py
```

**What you should see** (real output; dataset lines omitted): the `k=0..2` and `position 0..4` blocks shown above, then:

```
=== 2. the real MLP as makemore.py builds it (block_size=16, n_embd=64, n_embd2=64)
  wte.weight     (28, 64)       1792 numbers
  mlp.0.weight   (64, 1024)    65536 numbers
  mlp.0.bias     (64,)            64 numbers
  mlp.2.weight   (27, 64)       1728 numbers
  mlp.2.bias     (27,)            27 numbers
  total: 69147
input (32, 16) -> logits (32, 16, 27) | loss at init 3.3834
```

**Decode it:**

| Line | Meaning |
|---|---|
| `mlp.0`, `mlp.2` | Layers 0 and 2 of the `nn.Sequential`. Layer 1 is `tanh`, which has no parameters, so it isn't listed. |
| `mlp.0.weight (64, 1024)` | PyTorch stores a `Linear(a, b)` weight as `(b, a)` = (out, in). |
| `loss at init 3.3834` | Random starting weights: slightly worse than a uniform guess (3.2958). Normal. |

---

## What counts as success

| You can... | Test |
|---|---|
| Explain an embedding as a lookup | Question 1 |
| Run the roll loop in your head | Question 2 |
| Compute the parameter count from the settings | Question 3 |
| Explain why the nonlinearity matters | Question 4 |

## Check yourself

1. What is the difference between `Bigram.logits[idx]` and `MLP.wte(idx)`, mechanically?
2. With a window of 3, what does the MLP see at position 2 of `.ava`?
3. Run with `--type mlp --n-embd 32 --n-embd2 128`. How many parameters?
4. Replace `nn.Tanh()` with nothing. Which earlier model is the result most like, and why can't it beat the Bigram by much?
5. Why is the MLP's context window never a limitation on `names.txt`?

<details><summary>Answers</summary>

1. None in mechanism: both look up row `idx` of a trainable table. The difference is what the row means. The Bigram's row is directly 27 next-token logits. The MLP's row is 64 features that later layers turn into logits.
2. `.av`, i.e. tokens `0, 1, 22` (start, a, v). (Position 2 holds `v`. The window is `v`, `a`, and the start token.)
3. `wte`: 28 × 32 = 896. First layer: 128 × (16 × 32) + 128 = 65,664. Second layer: 27 × 128 + 27 = 3,483. Total **70,043**. (Run it: `model #params` prints the same.)
4. It becomes one big linear map from 16 concatenated embeddings to logits. Each previous character then adds its own independent contribution to the logits, and no interactions are possible. It can beat the Bigram somewhat, because it sees more characters, but it can't learn combinations.
5. `block_size` is set to the longest name + 1 = 16, so the window always covers the whole name. On a dataset of long lines, the window would equal the longest line, and the first layer would grow with it (`block_size × n_embd` inputs).

</details>

Next: [Lesson 6 — RNN and GRU](06-rnn-gru.md).
