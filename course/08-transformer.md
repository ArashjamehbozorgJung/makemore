# Lesson 8 — The Transformer

**This lesson promises:** you can trace a tensor through every line of `CausalSelfAttention`, `Block` and `Transformer`, shape by shape. You can account for all 204,544 parameters, and say what the trained model's attention actually learned to look at.

**Code covered:** lines 30–156.
**Lab:** `python course/labs/lab08_transformer.py` and, after training, `python course/labs/lab08_transformer.py names/model.pt`
**Paper:** Vaswani et al. 2017, "Attention Is All You Need". Optional: Karpathy's lecture "Let's build GPT: from scratch, in code, spelled out" builds the same architecture on Shakespeare.

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **dot product** | For two vectors of equal length: multiply position by position and sum. `[1,2]·[3,4] = 3 + 8 = 11`. Large and positive when the vectors point the same way, near 0 when unrelated, negative when opposite. |
| **attention** | At each position, compute a weighted average of vectors from positions 0..t. The weights are a softmax of **scores computed from the data**. |
| **query `q`, key `k`, value `v`** | Three vectors computed from each position's vector by three different linear maps. The score between position `t` and position `s` is `q[t]·k[s]`. The output at `t` is the weighted average of the `v[s]`. |
| **head** | One independent copy of the attention computation, with its own slice of `q`, `k`, `v`. Here 4 heads of 16 numbers each. |
| **LayerNorm** | For each position's vector of 64 numbers separately: subtract their mean, divide by their standard deviation, then multiply by a learned vector `γ` and add a learned vector `β`. |
| **GELU** | An activation function. Like `max(0, x)` but smooth: ≈ `x` for large positive `x`, ≈ 0 for large negative `x`, slightly negative in between (minimum about −0.17). |
| **residual stream** | The vector `x` at each position that flows through all blocks. Each block reads it and *adds* its output to it (lesson 7's residual connection). |
| **pre-norm** | LayerNorm is applied to the *input* of each sub-layer (`attn(ln_1(x))`), not to the output. GPT-2 does this. |
| **`view` / `transpose` / `contiguous`** | `view` reinterprets a tensor's shape without moving data. `transpose(a, b)` swaps two dimensions. `contiguous()` makes an actual copy in the new memory order, which `view` needs after a `transpose`. |

---

## 1. The whole model on one screen: lines 136–156

```python
pos = torch.arange(0, t)                       # [0, 1, ..., t-1]
tok_emb = self.transformer.wte(idx)            # (b, t, 64)
pos_emb = self.transformer.wpe(pos)            # (1, t, 64)
x = tok_emb + pos_emb                          # same start as the BoW model
for block in self.transformer.h:               # 4 blocks, one after another
    x = block(x)                               # (b, t, 64) in, (b, t, 64) out
x = self.transformer.ln_f(x)                   # final LayerNorm
logits = self.lm_head(x)                       # (b, t, 27); Linear with bias=False
```

Compared with the BoW model, three things changed:
1. The fixed average becomes **attention** (sections 2–4).
2. Every block gets **LayerNorm** before each sub-layer, plus one at the end (section 5).
3. The block is **stacked 4 times** (`--n-layer`), and its MLP is 4× wider (section 6).

## 2. From the BoW average to attention: the one real change

Lesson 7's `CausalBoW`:

```python
att = torch.zeros((B, T, T))                          # every score = 0
att = att.masked_fill(mask == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ x
```

`CausalSelfAttention` (lines 83–86), for one head:

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # score[t, s] = q[t]·k[s] / sqrt(16)
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))   # same causal mask
att = F.softmax(att, dim=-1)                                       # same softmax
y = att @ v                                                        # same weighted sum, of v instead of x
```

The mask, the softmax and the weighted sum are the same. Only the scores changed, from constant zeros to `q[t]·k[s]`. Position `t` now gives a high weight to position `s` when `t`'s query vector points the same way as `s`'s key vector. Since `q`, `k` and `v` come from learned linear maps, **training decides what each position looks for (`q`), how each position advertises itself (`k`), and what it hands over when looked at (`v`).**

`k.transpose(-2, -1)` turns `k` from `(…, T, 16)` into `(…, 16, T)`. Then `q @ kᵀ` gives `(…, T, T)`: every query dotted with every key in one operation.

### Why divide by √16 = 4

If `q` and `k` each have 16 entries of typical size 1, their dot product is a sum of 16 products. Its typical size grows like √16 = 4. Large scores make softmax nearly one-hot: one weight ≈ 1, the rest ≈ 0. Near-one-hot softmax gives near-zero gradients for the losing positions, so they stop learning. Dividing by √(head size) brings the scores back to typical size ≈ 1.

## 3. Multi-head: lines 77–87, shape by shape

With `B` names, `T` positions, `C = 64`, `nh = 4` heads, `hs = C/nh = 16`:

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # c_attn: Linear 64 -> 192; split into 3 x (B,T,64)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B,T,64) -> (B,T,4,16) -> (B,4,T,16)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # same
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # same
att = ... # (B, 4, T, T): a separate T x T attention pattern per head
y = att @ v                                          # (B, 4, T, 16)
y = y.transpose(1, 2).contiguous().view(B, T, C)     # (B, T, 4, 16) -> (B, T, 64): heads side by side
y = self.c_proj(y)                                    # Linear 64 -> 64: mix the heads' outputs
```

- One `Linear(64, 192)` computes `q`, `k` and `v` together. That is three `Linear(64, 64)`s stacked into one matrix, for speed.
- `view(B, T, 4, 16)` cuts each 64-vector into 4 slices of 16. `transpose(1, 2)` moves the head dimension forward, so the `@` treats each head like a separate batch element.
- Each head computes **its own** `T × T` pattern from its own 16-number slice. One head can look one character back while another looks two back (section 8 shows a trained model doing exactly that).
- After `att @ v`, the reverse reshape puts the 4 heads' 16 numbers side by side to form 64 again. `c_proj` then lets the heads' results combine.

The lab replays all of this by hand on `.emma` and compares the result with `blk.attn(...)`. Difference: `0.0`.

> **Careful:** `self.bias` (line 68) is the causal mask again, shape `(1, 1, 16, 16)`, registered as a **buffer**. It is saved in `model.pt` but never trained. It is not a bias. The `(1, 1, …)` lets it broadcast over batch and heads.

## 4. The fact that organises the whole architecture

**Attention is the only operation in the model that moves information between positions.** `wte`, `wpe`, every `Linear`, GELU and LayerNorm act on each position's vector **separately**. So a Transformer block is:

1. **attention:** each position gathers information from earlier positions, then
2. **MLP:** each position processes what it now holds, on its own.

Repeat 4 times. Lab 9 confirms that no position ever receives information from a later one.

## 5. The block: lines 93–112

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # normalize, attend, add back to the stream
    x = x + self.mlpf(self.ln_2(x))   # normalize, MLP, add back to the stream
    return x
```

**LayerNorm (`nn.LayerNorm(64)`)**, at each position: `out = γ · (x − mean(x)) / sqrt(var(x) + 1e-5) + β`, with `γ` and `β` learned, 64 numbers each, so **128 parameters**. It keeps the input to each sub-layer at a predictable scale however large the residual stream grows. That is the scale problem that gave the BoW its 4.15 starting loss (lesson 7, section 5). The Transformer's starting loss was 3.41–3.51.

Pre-norm means the residual stream `x` itself is never normalized, only the copy handed to each sub-layer. The `x +` path stays a clean "pass straight through", with gradient 1 (lesson 7).

## 6. The MLP: lines 101–107, and GELU: lines 45–51

```python
c_fc   = nn.Linear(64, 256)     # widen 4x
act    = NewGELU()
c_proj = nn.Linear(256, 64)     # back to 64
self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))
```

`NewGELU` computes `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. That is a standard approximation of `x · Φ(x)`, where `Φ` is the probability that a standard normal random number is below `x`. You don't need the formula. You need the shape: it passes positive values, suppresses negative ones, and is smooth. Its job is the same as `tanh` in the MLP of lesson 5: without a nonlinearity, the two `Linear`s would collapse into one.

## 7. Every parameter accounted for

| Part | Computation | Count |
|---|---|---|
| `wte` | 27 × 64 | 1,728 |
| `wpe` | 16 × 64 | 1,024 |
| per block: `ln_1` | 64 + 64 | 128 |
| per block: `attn.c_attn` | 64 × 192 + 192 | 12,480 |
| per block: `attn.c_proj` | 64 × 64 + 64 | 4,160 |
| per block: `ln_2` | 64 + 64 | 128 |
| per block: `mlp.c_fc` | 64 × 256 + 256 | 16,640 |
| per block: `mlp.c_proj` | 256 × 64 + 64 | 16,448 |
| **one block** | | **49,984** |
| 4 blocks | 4 × 49,984 | 199,936 |
| `ln_f` | 64 + 64 | 128 |
| **printed as "0.20M"** | everything above | **202,816** |
| `lm_head` (no bias) | 64 × 27 | 1,728 |
| **total (`model #params`)** | | **204,544** |

Two-thirds of each block (33,088 of 49,984) is the MLP, not the attention.

## 8. What the trained model's attention learned

I trained the default Transformer for 20,000 steps and loaded its best checkpoint into the lab. These are block 0's four heads on `.emma`. Row = the position doing the reading, column = the position being read; each row sums to 1:

```
head 3                                   head 0
      .     e     m     m     a                .     e     m     m     a
  .   1.00  0.00  0.00  0.00  0.00         .   1.00  0.00  0.00  0.00  0.00
  e   0.89  0.11  0.00  0.00  0.00         e   0.61  0.39  0.00  0.00  0.00
  m   0.01  0.98  0.01  0.00  0.00         m   0.52  0.33  0.15  0.00  0.00
  m   0.02  0.02  0.89  0.07  0.00         m   0.04  0.94  0.01  0.00  0.00
  a   0.00  0.02  0.01  0.97  0.00         a   0.03  0.06  0.75  0.15  0.00
```

**Head 3 reads the previous character.** Each row's weight sits just left of the diagonal (0.89, 0.98, 0.89, 0.97). **Head 0 mostly reads the character two back** (row `m`→`e` 0.94, row `a`→`m` 0.75).

One name is an anecdote, so I averaged over all 1,000 test names, at every real position from 2 onward:

| Layer 0 head | weight on itself | 1 back | 2 back | start token |
|---|---|---|---|---|
| head 0 | 0.10 | 0.19 | **0.44** | 0.10 |
| head 1 | 0.12 | **0.47** | 0.17 | 0.18 |
| head 2 | 0.12 | 0.31 | 0.35 | 0.11 |
| head 3 | 0.06 | **0.82** | 0.06 | 0.03 |

So the pattern holds across the test set: layer 0 dedicates a head almost entirely to "previous character" and another largely to "two back". Nobody programmed this. It came out of minimizing the loss. Layers 1–3 show no single dominant offset; their weights are spread (0.15–0.45 on any one offset). What they compute is not readable from these averages.

> **Careful:** this is one training run with one seed. Which head *number* ends up with which job is arbitrary. Another seed might put the previous-character pattern in head 1. The claim "some head in layer 0 learns previous-character attention" is what this evidence supports. "Head 3 is the previous-character head" is not a general fact.

## 9. "Exactly as in GPT-2": true for the architecture, not for training

The docstring says "Transformer Language Model, exactly as seen in GPT-2". The **block structure** matches GPT-2: pre-norm, GELU, a 4× MLP, learned position embeddings, a final LayerNorm. Differences you should know about:

| GPT-2 | This file | Source |
|---|---|---|
| Dropout | None | file docstring, lines 8–9 |
| Special weight initialization | PyTorch defaults | commit `d26d975` "remove weight init, not needed at this scale" |
| `lm_head` shares its weights with `wte` ("weight tying") | Separate `lm_head` | lines 122, 127 |
| 12 layers, 768 channels, 12 heads (smallest GPT-2) | 4 layers, 64 channels, 4 heads | defaults, lines 37–40 |
| ~50,000 sub-word tokens | 27 characters | — |

These are deliberate simplifications, documented by the author. They are not bugs. Dropout's absence does have a measurable effect here: lesson 9.

---

## Lab 8

```bash
python course/labs/lab08_transformer.py
```

**What you should see** (real output, untrained weights; dataset lines omitted):

```
=== 1. where the parameters are (the constructor prints its own count first)
number of parameters: 0.20M
  transformer.wte                                         1728
  transformer.wpe                                         1024
  transformer.h.0.ln_1                                     128
  transformer.h.0.attn.c_attn                            12480
  transformer.h.0.attn.c_proj                             4160
  transformer.h.0.ln_2                                     128
  transformer.h.0.mlp.c_fc                               16640
  transformer.h.0.mlp.c_proj                             16448
  transformer.h.1..3 (3 more blocks, same as block 0)   149952
  transformer.ln_f                                         128
  lm_head                                                 1728
  total: 204544  (the 0.20M printed above leaves out lm_head)

=== 2. replay block 0's attention by hand on '.emma'
x (B,T,C) = (1, 5, 64);  q, k, v each (1, 5, 64)
split into 4 heads of size 16: q, k, v each (B, nh, T, hs) = (1, 4, 5, 16)
scores = q @ k^T / sqrt(16): (B, nh, T, T) = (1, 4, 5, 5)
max |hand-computed - CausalSelfAttention output| = 0.0

head 0: row = position reading, column = position read from (rows sum to 1)
      .     e     m     m     a
  .   1.00  0.00  0.00  0.00  0.00
  e   0.37  0.63  0.00  0.00  0.00
  m   0.33  0.25  0.43  0.00  0.00
  m   0.24  0.37  0.19  0.20  0.00
  a   0.11  0.08  0.30  0.35  0.16
...
(untrained weights: the pattern is near-uniform noise. pass a trained model.pt to see learned patterns)
```

**Decode it:**

| Line | Meaning |
|---|---|
| `0.20M` then `204544` | The constructor's count excludes `lm_head`; section 7 reconciles them. |
| `q, k, v each (1, 5, 64)` → `(1, 4, 5, 16)` | The split into 4 heads of 16. |
| `(1, 4, 5, 5)` | One 5 × 5 score table per head. |
| `0.0` | The hand replay equals the module exactly. |
| untrained head 0 | Weights spread fairly evenly: no pattern yet. Compare section 8. |
| row `.` is always `1.00 0 0 0 0` | Position 0 can only see itself, trained or not. |

Then train (lesson 0, step 5) and run `python course/labs/lab08_transformer.py names/model.pt`. Your heads will probably differ from mine in numbering. Look for a head whose weights sit just left of the diagonal.

---

## What counts as success

| You can... | Test |
|---|---|
| Say what attention changes relative to BoW | Question 1 |
| Trace shapes through multi-head attention | Question 2 |
| Compute a block's parameter count from `n_embd` | Question 3 |
| Say which operations move information between positions | Question 4 |
| Separate "documented simplification" from "bug" | Question 5 |

## Check yourself

1. Write the single line of `CausalBoW.forward` that would need to change to make it single-head attention (ignoring that you'd also need `q`, `k`, `v`).
2. With `--n-embd 96 --n-head 6`, what is the head size, and what is the shape of `att` for a batch of 32 full-length names?
3. With `n_embd = 96`, how many parameters does one block have?
4. In a Transformer block, which operations can let position 5 use information from position 2?
5. A colleague says "this Transformer is buggy, GPT-2 uses dropout and it doesn't". Respond.
6. Why does `--n-embd 64 --n-head 5` crash, and on which line?

<details><summary>Answers</summary>

1. `att = torch.zeros((B, T, T))` → `att = (q @ k.transpose(-2, -1)) / math.sqrt(C)`, and `y = att @ x` → `y = att @ v`.
2. Head size 96 / 6 = 16. `att` has shape `(32, 6, 16, 16)`: batch, heads, T, T.
3. LayerNorms 2 × 192 = 384. `c_attn` 96 × 288 + 288 = 27,936. `c_proj` 96 × 96 + 96 = 9,312. `c_fc` 96 × 384 + 384 = 37,248. `mlp.c_proj` 384 × 96 + 96 = 36,960. Total **111,840**. (General: 12·C² + 13·C. Check: C = 64 gives 49,152 + 832 = 49,984.)
4. Only attention. Everything else in the block acts on each position independently.
5. Dropout's absence is a deliberate, documented simplification (file docstring), not a bug. It does have a measurable cost: the default model overfits after about 10,000–15,000 steps (lesson 9). "Different from GPT-2" and "broken" are different claims.
6. `assert config.n_embd % config.n_head == 0` on line 62: 64 can't be split into 5 equal heads.

</details>

Next: [Lesson 9 — all six models compared, and your own data](09-experiments.md).
