# Lesson 6 — RNN and GRU: a memory carried forward

**This lesson promises:** you will know what a hidden state is, how the RNN and GRU update it one character at a time, and what the GRU's two gates actually compute. You will also see what the one-line bug fixed in commit `bf38625` had broken.

**Code covered:** `RNNCell`, `GRUCell`, `RNN` (lines 253–345).
**Lab:** `python course/labs/lab06_rnn.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **hidden state `h`** | A vector of 64 numbers (`n_embd2`) that the model updates after reading each character. It is the model's only memory of everything read so far. |
| **recurrence** | A rule of the form `h_t = f(x_t, h_{t−1})`: the new state is computed from the current input and the previous state. The *same* `f`, with the same parameters, is applied at every position. |
| **cell** | The module that implements `f`. Here `RNNCell` or `GRUCell`. |
| **sigmoid `σ(x)`** | `1 / (1 + e⁻ˣ)`, applied to each number. Output always between 0 and 1. |
| **gate** | A vector of numbers between 0 and 1 (made with a sigmoid), multiplied element by element with another vector. 0 blocks that element, 1 lets it through, 0.5 halves it. |
| **element-wise product (`*` in PyTorch)** | Multiply two same-shape vectors position by position: `[a, b] * [c, d] = [ac, bd]`. |
| **vanishing gradient** | When a gradient is passed backwards through many steps, it gets multiplied by a factor at each step. If those factors are mostly below 1, the gradient shrinks toward 0, and early inputs get almost no learning signal. |

---

## 1. The RNN in one sentence

Start with a learned vector `h`. Then, for each character in order: combine the character's embedding with `h` to make a new `h`. Then turn each `h` into 27 logits.

## 2. `RNN.forward`: lines 320–345

```python
emb = self.wte(idx)                      # (b, t, 64): embed every character up front
hprev = self.start.expand((b, -1))       # (b, 64): the learned starting state, one copy per name
hiddens = []
for i in range(t):                       # one position at a time, in order
    xt = emb[:, i, :]                    # (b, 64): the character at position i
    ht = self.cell(xt, hprev)            # (b, 64): new state from (character, previous state)
    hprev = ht                           # carry it forward  <- the line added by commit bf38625
    hiddens.append(ht)
hidden = torch.stack(hiddens, 1)         # (b, t, 64)
logits = self.lm_head(hidden)            # (b, t, 27)
```

- `self.start` is an `nn.Parameter` of shape `(1, 64)`. It starts at zeros and is **learned** like any other weight. `.expand((b, -1))` makes it look like `b` copies without copying memory.
- **No position embedding.** The RNN knows "how far into the name" only through what has accumulated in `h`.
- **It can't see the future** by construction: `h` at position `i` is built only from positions `0..i`.

> **Careful — the bug that commit `bf38625` fixed:** before it, the line `hprev = ht` was missing. Every position then computed `ht = cell(xt, self.start)`, always from the *starting* state. The model had no memory at all: each prediction depended only on the current character. It was a Bigram with extra steps. It still trained and produced name-ish output, which is exactly why the bug survived. **A model that runs and whose loss goes down is not necessarily the model you think you wrote.** The causality and "does changing earlier input change later output" tests in lab 9 are the kind of check that catches this.

## 3. `RNNCell`: lines 261–275

```python
self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)   # 128 -> 64

def forward(self, xt, hprev):
    xh = torch.cat([xt, hprev], dim=1)      # (b, 128): character and memory side by side
    ht = F.tanh(self.xh_to_h(xh))           # (b, 64), each number in (-1, 1)
    return ht
```

As a formula: `h_t = tanh(W · [x_t ; h_{t−1}] + b)`. One linear layer and one `tanh`, reused at all 16 positions. The same weights at every position means that what it learns about `ly→n` at position 3 applies at position 8 too. The MLP (lesson 5) couldn't do that.

Parameters: `start` 64 + `wte` 27×64 = 1,728 + cell 64×128 + 64 = 8,256 + `lm_head` 27×64 + 27 = 1,755 → **11,803**.

## 4. Why a plain RNN is hard to train

To learn that a character 10 positions back matters, the gradient must travel backwards through 10 applications of the cell. Each one multiplies it by the cell's weights and by the slope of `tanh`, which is at most 1 and much less than 1 when `tanh` is near ±1. Multiply ten such factors and the result is usually tiny: the **vanishing gradient**. The plain RNN therefore learns short-range patterns easily and long-range ones poorly.

## 5. `GRUCell`: lines 277–301

The GRU (Gated Recurrent Unit, Cho et al. 2014) adds two gates. Line by line:

```python
xh = torch.cat([xt, hprev], dim=1)
r = F.sigmoid(self.xh_to_r(xh))           # reset gate, (b, 64), each in (0, 1)
hprev_reset = r * hprev                   # zero out (part of) the memory where r is near 0
xhr = torch.cat([xt, hprev_reset], dim=1)
hbar = F.tanh(self.xh_to_hbar(xhr))       # candidate new state, computed from the reset memory
z = F.sigmoid(self.xh_to_z(xh))           # update gate, (b, 64), each in (0, 1)
ht = (1 - z) * hprev + z * hbar           # per number: keep the old value, take the new one, or a mix
```

What each gate decides, for each of the 64 numbers separately:

- **`r` (reset):** how much of the old memory to use when computing the candidate `hbar`. `r ≈ 0` means "compute the candidate as if starting fresh".
- **`z` (update):** how much to replace. `z ≈ 0` keeps `hprev` **unchanged**. `z ≈ 1` replaces it with `hbar`.

**Why this helps with vanishing gradients:** when `z ≈ 0`, `ht ≈ hprev` exactly. No `tanh`, no weight matrix. The gradient passes backwards through that step almost unchanged. The GRU can learn to hold a piece of information across many steps.

> **Careful:** the comment on line 284 says `# input, forget, output, gate`. Those are the gate names of an **LSTM**, a different recurrent cell that this file doesn't implement (lines 255–258 say why). The GRU has two gates, reset and update, plus the candidate. The comment is stale.

> **Careful (only if you read the paper):** Cho et al. write the blend as `h = z·h_prev + (1−z)·h̃`, with `z` meaning "keep". This code uses `z` to mean "replace". They are the same model with `z` relabeled. A learned gate can represent either convention.

Parameters: three `Linear(128 → 64)` = 3 × 8,256 = 24,768, plus `start`, `wte`, `lm_head` as before → **28,315**.

## 6. What they achieved

From my 20,000-step runs: RNN best test loss **2.0539**, GRU **1.9988**. The MLP got 2.0304. So the GRU beats the MLP and the plain RNN does not. The GRU was still improving at step 20,000. It beats the plain RNN with 2.4× the parameters (28,315 vs 11,803), and it is the best non-Transformer model in this file.

Both are slow per parameter. The loop over positions is sequential: position 5 can't be computed until position 4 is done. The Transformer (lesson 8) computes all positions at once.

---

## Lab 6

```bash
python course/labs/lab06_rnn.py
```

**What you should see** (real output; dataset lines omitted):

```
=== RNN parameters
  start                  (1, 64)        64
  wte.weight             (27, 64)     1728
  cell.xh_to_h.weight    (64, 128)    8192
  cell.xh_to_h.bias      (64,)          64
  lm_head.weight         (27, 64)     1728
  lm_head.bias           (27,)          27
  total: 11803
  input (32, 16) -> logits (32, 16, 27) | loss at init 3.3356

=== GRU parameters
  ...
  total: 28315
  input (32, 16) -> logits (32, 16, 27) | loss at init 3.3069

=== stepping the RNN cell by hand on '.emma': the hidden state h is the model's only memory
t=0 read '.': h[:5] = tensor([ 0.152,  0.379,  0.584, -0.426,  0.307])  |h| = 2.906
t=1 read 'e': h[:5] = tensor([ 0.529,  0.135, -0.205, -0.336, -0.093])  |h| = 3.098
t=2 read 'm': h[:5] = tensor([-0.345, -0.580, -0.270, -0.365, -0.171])  |h| = 2.699
t=3 read 'm': h[:5] = tensor([-0.375, -0.346, -0.039, -0.541, -0.456])  |h| = 2.717
t=4 read 'a': h[:5] = tensor([-0.254,  0.568,  0.527, -0.388,  0.001])  |h| = 3.216
every h value is squashed into (-1, 1) by tanh

=== GRU gates on one step: r (reset) and z (update) are sigmoids, so each entry is in (0, 1)
r[:6] = tensor([0.504, 0.311, 0.384, 0.381, 0.420, 0.585])
z[:6] = tensor([0.611, 0.689, 0.343, 0.436, 0.446, 0.470])
check the blend ht = (1-z)*hprev + z*hbar is what the cell returned:
  max difference: 0.0
```

**Decode it:**

| Line | Meaning |
|---|---|
| `start (1, 64) 64` | The learned initial memory. |
| `cell.xh_to_h.weight (64, 128)` | 128 inputs (64 character features + 64 memory), 64 outputs. |
| `t=0 ... t=4` lines | The first 5 of the 64 numbers in `h`, after each character. The state changes completely each step, because the weights are random and untrained. `\|h\|` is the vector's length, √(sum of squares). With 64 values each in (−1, 1), it can be at most 8. |
| `t=2` and `t=3` both read `'m'` but give different `h` | The same input with a different memory gives a different state. That is the point of the recurrence. |
| gates around 0.3–0.7 | Untrained gates sit near 0.5: "half keep, half replace". Training pushes them toward 0 or 1 where useful. |
| `max difference: 0.0` | The lab recomputed the GRU's last line by hand and got exactly what the module returns. |

---

## What counts as success

| You can... | Test |
|---|---|
| Write the RNN update as a formula | Question 1 |
| Say what `z ≈ 0` and `z ≈ 1` do | Question 2 |
| Explain the effect of the `bf38625` bug | Question 3 |
| Explain where the RNN gets position information | Question 4 |

## Check yourself

1. Write `h_t` for the RNN in terms of `x_t`, `h_{t−1}`, a weight matrix, a bias and `tanh`.
2. One number in `z` is 0.0 at every step of a name. What happens to that number of the hidden state?
3. Before commit `bf38625`, what was the best loss this "RNN" could possibly reach, roughly, and why?
4. The RNN has no `wpe`. How can it predict that names rarely exceed 10 letters?
5. With `--n-embd2 128` instead of 64, how many parameters does the RNN cell (`xh_to_h`) have?

<details><summary>Answers</summary>

1. `h_t = tanh(W · concat(x_t, h_{t−1}) + b)`, with `W` of shape 64 × 128 and `b` of length 64.
2. It never changes: `ht = 1·hprev + 0·hbar`. That number keeps the value from `start` all the way through the name.
3. About the Bigram's floor, 2.454. Each prediction depended only on the current character's embedding, passed through one `tanh` layer from a fixed starting state. That is a function of one character, the same information a Bigram has.
4. Position is implicit in `h`. After reading more characters, the state is different, and the model can learn to make "end" more likely as the state reflects a longer name.
5. Input 64 + 128 = 192, output 128: 192 × 128 + 128 = **24,704**.

</details>

Next: [Lesson 7 — Bag of Words](07-bow.md).
