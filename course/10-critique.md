# Lesson 10 — Where this code is weak, and what is intended

**This lesson promises:** a list of the real weaknesses in `makemore.py`, each with evidence you can reproduce. It also lists the "weaknesses" you must **not** present as faults, because the author documented them as deliberate.

**Lab:** `python course/labs/lab10_findings.py`

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **defect** | The code does something different from what it is evidently meant to do, or fails on reasonable input. |
| **documentation error** | The code behaves sensibly, but a comment, docstring or README says something untrue about it. |
| **deliberate simplification** | Something left out or done crudely **on purpose**, and documented as such by the author. Calling it a bug is wrong, and it costs you credibility. |
| **data leakage** | Information from the evaluation data getting into training, so that evaluation scores look better than they really are. |
| **selection bias** | Choosing the best of several noisy measurements and then reporting that best value as if it were a single measurement. The reported value is biased toward looking good. |

---

## How to use this lesson

Before you tell anyone "this code has a bug", put the claim in one of three bins:

1. **Defect, with evidence.** You can show inputs → wrong behaviour.
2. **Documentation error.** The text is wrong; the code is fine.
3. **Deliberate, documented simplification.** It *is* intended. Point to the commit or comment that says so.

This code is an **educational** file by its own description ("mostly intended for educational purposes", README). Bin 3 is large on purpose.

---

## Bin 1 — Defects, with evidence

| # | Defect | Evidence (reproduce it) | Consequence | Minimal fix |
|---|---|---|---|---|
| 1 | **Duplicate lines leak into the test set.** `names.txt` has 2,539 names twice. The split (lines 562–565) shuffles lines, not distinct names. | Lab 1 / lab 10: 153 of 1,000 test names (15.3%) are also training names. Lab 9b: Transformer test loss 1.740 on those vs 2.003 on unseen names. | Test loss is flattered, most for the models that memorize most. In lesson 9 it made the Transformer look clearly better than the GRU. On unseen names the two are nearly tied. | Deduplicate before splitting, e.g. `words = list(dict.fromkeys(words))` after line 552. (`dict.fromkeys` keeps the first copy of each name, in order.) |
| 2 | **The test set is also used to choose the checkpoint.** Lines 705–709 save whichever model has the lowest *test* loss. | Read the code. | The "best test loss" is selection-biased: optimistic, probably by a small amount here. No untouched set is left for an honest final number. | A three-way split: train / validation (for choosing) / test (report once, at the end). |
| 3 | **Files with fewer than 10 lines crash with an unhelpful error.** `int(n * 0.1)` is 0, and `rp[:-0]` is an empty list in Python, so *all* lines become test lines. | 5-line file: `split up the dataset into 0 training examples and 5 test examples`, then `RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=0`. A 10-line file works (9 train / 1 test). | A new user with a small file gets an error that names neither the cause nor the fix. | `test_set_size = max(1, min(1000, int(len(words) * 0.1)))`, plus a clear error if the file has fewer than 2 lines. |
| 4 | **`--resume` forgets everything except the weights.** `model.pt` holds only `model.state_dict()` (line 708). | My resume run: weights loaded (first loss 2.12, not 3.5). But it printed `step 0`, and at step 500 it printed `is the best so far, saving model`, because `best_loss` had restarted as `None` (line 668). | The first evaluation after resuming **overwrites `model.pt` even if it is worse** than what was saved. AdamW's running averages restart from zero. Step numbers restart, so TensorBoard curves from the two sessions overlap. | Save a dict: `{'model': ..., 'optimizer': optimizer.state_dict(), 'step': step, 'best_loss': best_loss, 'args': vars(args)}`, and restore all of it. |
| 5 | **The model's settings are not saved with it.** `--sample-only` rebuilds the model from the command-line flags (lines 635–651) and then loads weights into it. | Train `--type bigram`, sample without `--type`: `RuntimeError: Error(s) in loading state_dict for Transformer: Missing key(s) ... Unexpected key(s) in state_dict: "logits".` | Easy to hit; the error doesn't say "you forgot `--type bigram`". With mismatched sizes the error is a shape mismatch, equally unhelpful. | Same fix as #4: store `vars(args)` in the checkpoint and read the model settings from it. |
| 6 | **The input file is read with the system's default text encoding.** Line 548: `open(input_file, 'r')`, no `encoding=`. | On a system whose default is cp1252 (common on Windows), UTF-8 `'Jürgen'` decodes as `'JÃ¼rgen'`. I reproduced the decoding. I did not test on a Windows machine. | Non-ASCII data on Windows silently trains on garbage tokens. Relevant for German data. | `open(input_file, 'r', encoding='utf-8')`. Workaround without editing: `python -X utf8 makemore.py ...`. |
| 7 | **`evaluate(max_batches=10)` evaluates 11 batches** (minor). The check `i >= max_batches` (line 497) runs *after* batch `i` is used. | Lab 10: `train: 11 batches, 1100 words out of 31033`. | Harmless in effect, but the parameter doesn't mean what it says. It also averages *batch* means rather than per-character losses, a tiny bias. | `if max_batches is not None and i + 1 >= max_batches: break`. |

## Bin 2 — Documentation errors

| # | What the text says | What is true | Evidence |
|---|---|---|---|
| 1 | README: "PyTorch is the only requirement." | `tensorboard` is also required. Line 28 imports it at startup. | Lab 10 #4: `ModuleNotFoundError: No module named 'tensorboard'` from line 28. |
| 2 | Docstring, lines 10–12: "I removed weight decay and all of the complexity around what parameters are and are not weight decayed." | Weight decay is **on**: `--weight-decay` defaults to 0.01 and applies to every parameter, biases and LayerNorm included. What was removed is the *selective* decay. | Lab 10 #3: `parameter groups: 1 \| weight_decay of that group: 0.01`. |
| 3 | Line 377: "gather the word embeddings of the previous 3 words". | It gathers `block_size` = 16 **character** embeddings. | Lesson 5, lab 5. |
| 4 | Line 284: "# input, forget, output, gate". | Those are LSTM gate names. The GRU has reset and update gates. | Lesson 6. |
| 5 | README: "test logprob of ~1.92". | Not reproduced: my default run's best was 1.966 in 20,000 steps, with test loss rising. And the number is contaminated by defect #1. Also, it is a *negative* log-prob (a loss), in nats per character. | Lesson 9. |

## Bin 3 — Deliberate simplifications: do **not** present these as bugs

| What's missing or crude | Where the author says it is deliberate |
|---|---|
| No dropout anywhere | File docstring, lines 8–9 |
| PyTorch default initialization instead of GPT-2's | Commit `d26d975`: "remove weight init, not needed at this scale" |
| No gradient clipping | Commit `4e0137d`: "remove gradient clipping i dont think its needed at this small scale" |
| Weight decay applied to all parameters | File docstring, lines 10–12 (the wording is off, see bin 2 #2, but the simplification is intended) |
| No LSTM | Code comment, lines 255–258 |
| No CNN/WaveNet despite the README listing it | README marks it "(in progress...)" |
| `0.20M` printed parameter count leaves out `lm_head` | Comment on line 129 says so |
| Training runs forever by default | `--max-steps` help text: "-1 for infinite" |
| `InfiniteDataLoader` is a hack | Its own docstring: "this is really hacky and I'm not proud of it" |
| `generate` recomputes the full prefix each step (no caching) | Not stated, but irrelevant at 16 tokens, and caching would obscure the teaching code |
| `print_samples` runs with the model in training mode | No dropout or batch norm exists, so training mode and evaluation mode behave identically |

One measurable *consequence* of bin 3: without dropout, the default Transformer overfits after about 10,000–15,000 steps (lesson 9). It is fair to say: "a documented simplification has a measurable cost at the default settings". It is not fair to say: "the Transformer is broken".

## Historical: a real bug that was fixed

Commit `bf38625` (from an outside contributor, merged in commit `f61811b`) added the line `hprev = ht` to `RNN.forward`. Before it, the "RNN" had no memory (lesson 6). Its loss still went down and it still produced name-like output. Remember that pattern: **a falling loss proves the code trains something, not that it trains what you think.**

---

## Lab 10

```bash
python course/labs/lab10_findings.py
```

**What you should see** (real output; dataset lines omitted):

```
=== 1. evaluate(max_batches=10) looks at 11 batches, not 10
train: 11 batches, 1100 words out of 31033
test : 10 batches, 1000 words out of 1000 (the test set simply ran out)

=== 2. names that appear twice in names.txt can land in both train and test
153 of 1000 test words (15.3%) are also training words

=== 3. weight decay is applied to every parameter, including biases and LayerNorm
number of parameters: 0.20M
parameter groups: 1 | weight_decay of that group: 0.01

=== 4. 'PyTorch is the only requirement' (README): importing makemore without tensorboard
  File ".../makemore.py", line 28, in <module>
  File ".../torch/utils/tensorboard/__init__.py", line 1, in <module>
ModuleNotFoundError: No module named 'tensorboard'
```

**Decode it:**

| Line | Meaning |
|---|---|
| `train: 11 batches` | Defect #7. |
| `test : 10 batches` | The off-by-one has no effect on the test set: it has only 10 batches of 100. |
| `153 of 1000 ... (15.3%)` | Defect #1. |
| `parameter groups: 1`, `0.01` | One group containing all parameters, all decayed: bin 2 #2. |
| `line 28` → `tensorboard/__init__.py, line 1` | The import chain that fails without `tensorboard`: bin 2 #1. The lab simulates a missing package by hiding the installed one from a fresh Python process. The message is the same one a real missing package gives. |

---

## Exercises: fix them yourself

This is the "hero" part. Each is small. Do them on your own branch, one at a time, and re-run the matching lab to prove the fix.

1. **Defect #1:** deduplicate in `create_datasets`. Proof: lab 1 should report 0 overlapping test words. Then retrain the Transformer and the GRU. Is the ranking still clear?
2. **Defects #4 + #5:** save and restore a full checkpoint dict. Proof: resume a run, then confirm the step counter continues and a worse evaluation does **not** overwrite `model.pt`. Also, `--sample-only` without `--type` should now work.
3. **Defect #3:** make a 5-line file either train or fail with a clear message.
4. **Defect #6:** add `encoding='utf-8'` and train on a file with umlauts. Check the printed vocabulary.

When you've done one, show me the diff. I will try to break it.

---

## What counts as success

| You can... | Test |
|---|---|
| Name three real defects with evidence | Question 1 |
| Name three things that are *not* bugs and say why | Question 2 |
| Judge a claim before repeating it | Question 3 |

## Check yourself

1. Name three defects from bin 1. For each, say the cheapest way to demonstrate it.
2. Someone files an issue: "No dropout, no gradient clipping, and weight init was deleted. This Transformer is broken." Respond.
3. Someone says: "The README claims the Transformer reaches 1.92, so it's clearly the best model for names." What two problems are there with this argument?

<details><summary>Answers</summary>

1. For example: (a) the duplicate leak: run lab 1 and see 153 overlapping test names. (b) `--resume` overwriting a better checkpoint: resume and see "best so far, saving" at the first evaluation regardless. (c) A 5-line file: see "0 training examples" and the `random_` error. Also acceptable: test set used for selection (read lines 705–709), encoding (decode `'Jürgen'` as cp1252), the settings not being saved (sample a bigram run without `--type`).
2. All three are deliberate, documented simplifications: the docstring (dropout) and commits `4e0137d` (clipping) and `d26d975` (init). They are not bugs. The fair criticism is narrower: without dropout, the default Transformer overfits after about 10–15k steps, measurable as a growing train/test gap.
3. (a) I did not reproduce 1.92 with the defaults in 20,000 steps (best 1.966). (b) The test set is contaminated by duplicates. On truly unseen names, the Transformer (2.003) and GRU (2.010) are within noise of each other from one run each. "Clearly the best" isn't established.

</details>

---

You've reached the end of the course. Go back to the [course map](README.md) and check the success table. Then do one exercise from this lesson and send it to me.
