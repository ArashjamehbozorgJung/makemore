# Lesson 3 — How training changes the parameters

**This lesson promises:** you will know what each line of the training loop does to the numbers inside the model. That includes what `loss.backward()` puts where, what `optimizer.step()` adds to each parameter, and why the default learning rate makes the Bigram slow.

**Code covered:** the main block, lines 594–718; `InfiniteDataLoader`, lines 574–591; `evaluate`, lines 487–501.
**Lab:** `python course/labs/lab03_training.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **gradient** | For one parameter: how much the loss would change per unit of change in that parameter, if you nudged only that one. Positive means "increasing this parameter increases the loss". One gradient number per parameter. |
| **`.grad`** | The tensor where PyTorch stores the gradients. It has the same shape as the parameter. |
| **backpropagation / autograd** | PyTorch's way of computing all the gradients at once. During the forward pass it records every operation. `loss.backward()` then walks that record backwards, applying the chain rule of calculus. You never write derivatives yourself. |
| **gradient descent** | Move each parameter a small step in the direction opposite to its gradient, so the loss goes down. |
| **learning rate (`lr`)** | A number that sets how big each step is. Default here: `5e-4` = 0.0005. |
| **optimizer** | The object that applies the update rule to every parameter. Here it is `torch.optim.AdamW`. |
| **Adam / AdamW** | An update rule that divides each gradient by a running estimate of its own typical size. AdamW adds weight decay. |
| **weight decay** | Every step, shrink every parameter slightly toward 0: `p ← p × (1 − lr × wd)`. Here `wd = 0.01`. |
| **batch / batch size** | The group of examples used for one step. Here 32 names (`--batch-size`). |
| **step** | One forward pass, one backward pass, one parameter update. |
| **epoch** | One full pass over the training set. **This code has no epochs** (see section 4). |
| **validation set** | Data used to choose *when to stop* or *which settings to keep*. It must be different from the test set that reports the final score. Section 6 explains why this matters here. |

---

## 1. The loop: lines 668–718, stripped to the essentials

```python
while True:
    X, Y = batch_loader.next()          # 1. 32 random names: X (32,16) inputs, Y (32,16) targets
    logits, loss = model(X, Y)          # 2. forward pass: predictions and the average -log p
    model.zero_grad(set_to_none=True)   # 3. throw away the previous step's gradients
    loss.backward()                     # 4. compute d(loss)/d(parameter) for every parameter
    optimizer.step()                    # 5. move every parameter using its gradient
    # every 10 steps: print; every 500: evaluate and maybe save; every 200: print samples
    step += 1
    if args.max_steps >= 0 and step >= args.max_steps:
        break
```

That is all training is: steps 1–5, thousands of times.

## 2. `loss.backward()`: what lands in `.grad`

After step 4, every parameter tensor `p` has a `p.grad` of the same shape. For the Bigram, `model.logits.grad` is 27 × 27.

For cross-entropy over softmax there is a known formula. You don't need to derive it, but the lab checks it:

```
gradient for the logits at one counted position = (softmax(logits) − onehot(correct answer)) / (number of counted positions)
```

`onehot(k)` is a list of 27 zeros with a 1 at position `k`. The Bigram reuses one row of its table at many positions, so each row's gradient is the **sum** of this over every position that used that row.

What the signs mean, mechanically:
- The correct answer gets `p − 1`, which is negative. Gradient descent moves *against* the gradient, so that logit goes **up**.
- Every wrong answer gets `p`, which is positive, so those logits go **down**.
- The bigger the mistake, the bigger the gradient.

The lab compares PyTorch's `.grad` with this hand formula on a real batch. The largest difference is `4.66e-10`, which is just rounding. So `backward()` computed exactly this.

**Why `zero_grad` before `backward`:** `backward()` *adds* into `.grad`; it doesn't overwrite. Without step 3, gradients from every previous step would pile up. `set_to_none=True` sets `.grad` to `None` instead of filling it with zeros. The effect is the same, with slightly less work.

## 3. `optimizer.step()`: AdamW, in plain terms

Line 662:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
```

For each parameter `p` with gradient `g`, AdamW keeps two running averages:

- `m`: running average of `g`. `betas[0] = 0.9` means each step keeps 90% of the old `m` and adds 10% of the new `g`.
- `v`: running average of `g²`. `betas[1] = 0.99`: keep 99% old, add 1% new.

Then the update is:

```
p ← p × (1 − lr × weight_decay)                          # weight decay
p ← p − lr × m̂ / (√v̂ + eps)                              # the Adam step
```

`m̂` and `v̂` are `m` and `v` with a correction for starting at zero. `eps = 1e-8` only prevents division by zero.

**The consequence that matters:** `m̂ / √v̂` is roughly "the gradient divided by its own typical size". Its magnitude is around 1 or smaller, whatever the gradient's size. So **each parameter moves by at most about `lr` per step.** On the very first step `m̂ = g` and `v̂ = g²`, so the step is `lr × g/|g|`: exactly `lr` in the direction of the sign. The lab shows every parameter with a nonzero gradient moving between 0.00049997 and 0.00049999999 on step 1. Those are all ≈ `lr = 0.0005`.

**Why this makes the default Bigram slow:** the best Bigram (lesson 2) has logits spread over several units. For example, `log(0.138)` − `log(0.0001)` ≈ 7. At 0.0005 per step, moving a logit by 7 takes at least 14,000 steps. In my 20,000-step `--type bigram` run, the test loss reached 2.4544 at step 10,000 and 2.4495 at step 20,000. Lab 2 used `lr = 0.1` and got there in about 100 steps.

> **Careful:** a bigger learning rate is not "better". It is faster for the Bigram, whose loss has a single smooth valley. For the deeper models a too-large `lr` makes the loss jump around or blow up. `5e-4` is a safe default for the Transformer. If you change it, change it by factors of about 3 and watch the loss.

## 4. Where batches come from: `InfiniteDataLoader`, lines 574–591

```python
train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
```

- `RandomSampler(..., replacement=True)` picks example indices at random, **with replacement**. The same name can appear twice in one batch, and some names may not be picked for a long time.
- `num_samples=1e10` makes it effectively endless. So there are no epochs: the loop just draws 32 random names per step, forever.
- `DataLoader` calls `CharDataset.__getitem__` for each picked index and stacks 32 results into `X: (32, 16)`, `Y: (32, 16)`.
- `num_workers=4` (default `-n 4`) runs `__getitem__` in 4 background processes. For data this small, `-n 0` (do it in the main process) is just as fast.

## 5. Evaluation and saving: lines 697–709 and `evaluate`, lines 487–501

Every 500 steps:

```python
train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
...
if best_loss is None or test_loss < best_loss:
    torch.save(model.state_dict(), out_path)   # model.pt
    best_loss = test_loss
```

- `evaluate` computes the loss **without updating anything**. `@torch.inference_mode()` turns off gradient recording, so it is faster and uses less memory.
- **Train loss** here uses 1,100 random training names, not all 31,033. The loop body checks `i >= max_batches` *after* processing batch `i`, so `max_batches=10` gives 11 batches (lab 10 counts them). It is an estimate, and it wobbles by a few hundredths from one evaluation to the next.
- **Test loss** uses all 1,000 test names, because the test set runs out after 10 batches of 100.
- `model.state_dict()` is a dictionary from parameter name to tensor. `model.pt` holds only that. It does **not** hold the optimizer's `m` and `v`, the step number, or the settings you used.

## 6. Why "best test loss" is not an honest test score

The program keeps the checkpoint with the lowest test loss. So the test set is being used to **choose** the model. Pick the best of 40 noisy measurements, and the chosen number is biased low. In ML terms the test set is acting as a **validation set**, and there is no separate untouched test set left to report.

In this repo the effect is probably small. One choice out of a few dozen evaluations, on a smooth curve. But you should know it is there before quoting a "test loss" as a clean number. Lesson 10 lists it with the other weaknesses.

## 7. The timing lines: 687–690

```python
if args.device.startswith('cuda'):
    torch.cuda.synchronize()
```

On a GPU, PyTorch returns from a call *before* the GPU has finished the work. Without `synchronize()`, the measured step time would be too small. On the CPU, work finishes before the call returns, so nothing is needed.

---

## Lab 3

```bash
python course/labs/lab03_training.py
```

**What you should see** (real output; dataset lines omitted):

```
=== 1. loss.backward() fills .grad: how the loss changes if you nudge each parameter
model.logits.grad has the same shape as model.logits: (27, 27)
counted positions in this batch: 233
max |autograd grad - hand-derived grad| = 4.656612873077393e-10
grad row for <START>, columns a..f: tensor([-0.03354,  0.00509,  0.00079, -0.00779, -0.00350,  0.00509])
negative entries = 'raise this logit to lower the loss' (these letters started names in this batch)

=== 2. optimizer.step(): AdamW's very first step moves each parameter by about lr, whatever the grad size
parameters with nonzero grad: 648 of 729
their |change| ranges from 0.0004999685334041715 to 0.0004999998491257429
parameters with zero grad changed by: 0.0 (weight decay of a zero is zero)

=== 3. the makemore.py loop: batch -> forward -> zero_grad -> backward -> step, repeated
step    0 | batch loss 3.2958
step  500 | batch loss 3.1127
step 1000 | batch loss 2.9745
step 1500 | batch loss 2.8117
step 2000 | batch loss 2.7620
step 2500 | batch loss 2.6559
step 3000 | batch loss 2.5472
with lr=5e-4 the bigram is still above its best (2.454) after 3000 steps: each logit moves at most ~lr per step
```

**Decode it:**

| Line | Meaning |
|---|---|
| `counted positions in this batch: 233` | 32 names with an average of about 6.3 letters, plus 32 end tokens. Padding is not counted. |
| `max \|autograd - hand\| = 4.66e-10` | `backward()` computed exactly the formula in section 2. |
| `grad row for <START>`: `0.00509` for `b` | No name in this batch starts with `b`. Every one of the 32 start positions contributes `(1/27)/233`, and 32 × (1/27)/233 = 0.00509. Positive: lower the logit for `b`. |
| `-0.03354` for `a` | 9 names in this batch start with `a`: 0.00509 − 9/233 = −0.0335. Negative: raise the logit for `a`. |
| `648 of 729` | 81 cells got no gradient: their row (previous token) didn't occur in this batch at all. |
| `0.00049997 to 0.00049999985` | Every touched parameter moved by ≈ lr on step 1, as section 3 says. |
| loss 3.30 → 2.55 over 3,000 steps | Slow, steady progress, limited by the ~lr-per-step speed. |

---

## What counts as success

| You can... | Test |
|---|---|
| Say what each of the 5 loop lines changes | Question 1 |
| Explain why `zero_grad` is needed | Question 2 |
| Predict the effect of the learning rate on Adam's step size | Question 3 |
| Say what `model.pt` does and doesn't contain | Question 4 |

## Check yourself

1. Which of the five loop lines change the model's parameters?
2. You delete `model.zero_grad(...)`. The code still runs. What goes wrong?
3. A parameter's gradient is 1000× bigger than another's. Under AdamW, roughly how do their step sizes compare?
4. After `--resume`, is training exactly as if it never stopped? (Look at what `model.pt` holds.)
5. In lesson 0's output, the per-step loss went from 2.0253 at step 940 *up* to 2.2508 at step 960. Is the model getting worse?

<details><summary>Answers</summary>

1. Only `optimizer.step()`. `backward()` writes `.grad` but leaves parameters alone. The forward pass only reads them.
2. Gradients accumulate across steps. Each update then uses the sum of all past gradients, so the updates become wrong and the training goes off track.
3. About the same, roughly `lr` each. Adam divides each gradient by its own running size. (Plain gradient descent would make the first step 1000× bigger.)
4. No. `model.pt` holds parameters only. The optimizer's running averages `m` and `v` restart from zero, the step counter restarts at 0, and `best_loss` restarts as `None`. So the first evaluation after resuming overwrites `model.pt` even if it is worse (lesson 10 shows this).
5. Not necessarily. The per-step loss is measured on one random batch of 32 names, and some batches are harder than others. Only the trend over many steps, or the 500-step evaluations on 1,000+ names, tells you about progress.

</details>

Next: [Lesson 4 — making new names](04-sampling.md).
