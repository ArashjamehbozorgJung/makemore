# Lesson 9 — All six models compared, and training on your own data

**This lesson promises:** real numbers for all six models under identical conditions, and how to read them, including the one result that the raw table gets wrong. You also get what every setting does, and the traps when you feed `makemore.py` your own file.

**Labs:** `python course/labs/lab09_all_models.py`, and `python course/labs/lab09_leak_split.py --type <type> <dir>/model.pt` after training

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **hyperparameter** | A setting chosen by you, not learned by training: model type, sizes, learning rate, batch size. |
| **overfitting** | The model gets better on the training data while getting no better, or worse, on new data. It is fitting details of the specific training examples that don't carry over. You see it as a growing gap between train loss and test loss. |
| **generalization** | Performance on data not used for training. That is what you actually want. |
| **capacity** | Loosely, how much a model *can* fit. More parameters usually means more capacity. |
| **seed** | The starting number for the random generator. The same seed gives the same "random" split, initialization and batches. A different seed gives a different but equally valid run. |
| **noise (in a measurement)** | Variation in a number that comes from randomness (which seed, which batches), not from a real difference between models. |

---

## 1. The experiment

Each model trained with `makemore.py`, unmodified, all defaults, seed 3407, 20,001 steps:

```bash
python makemore.py -i names.txt -o exp/<type> --type <type> --max-steps 20001 -n 0
```

The six runs ran at the same time on a 4-core cloud machine, one CPU thread each. So the wall-clock times below reflect that setup, not a fair speed comparison.

## 2. Raw results

"Train" is `makemore.py`'s own estimate at step 20,000, from 1,100 random training names. "Best test" is the lowest test loss over all 40 evaluations, i.e. the checkpoint `model.pt` kept.

| Model | Params | Test @1k | Test @5k | Test @20k | **Best test** | Train @20k | Wall time |
|---|---|---|---|---|---|---|---|
| bigram | 729 | 2.9647 | 2.5008 | 2.4495 | **2.4495** | 2.4382 | 91 s |
| bow | 12,827 | 2.2853 | 2.2159 | 2.1559 | **2.1530** | 2.1528 | 155 s |
| rnn | 11,803 | 2.2206 | 2.1138 | 2.0559 | **2.0539** | 2.0383 | 194 s |
| mlp | 69,147 | 2.1941 | 2.0864 | 2.0304 | **2.0304** | 1.9930 | 248 s |
| gru | 28,315 | 2.2017 | 2.0755 | 1.9988 | **1.9988** | 1.9993 | 293 s |
| transformer | 204,544 | 2.1338 | 2.0126 | 1.9840 | **1.9655** | 1.7089 | 563 s |

At face value: Transformer > GRU > MLP > RNN > BoW > Bigram. That is the order the README's list suggests.

## 3. Read the Transformer's curve, not just its best number

Test and train loss every 2,500 steps:

| Step | 2,500 | 5,000 | 7,500 | 10,000 | 12,500 | 15,000 | 17,500 | 20,000 |
|---|---|---|---|---|---|---|---|---|
| train (est.) | 2.039 | 1.976 | 1.906 | 1.860 | 1.837 | 1.773 | 1.759 | 1.709 |
| test | 2.063 | 2.013 | 1.983 | 1.973 | 1.973 | 1.966 | 1.976 | 1.984 |

Train keeps falling. Test flattens near 1.97 from about step 7,500, and is slightly *worse* at 20,000 than at 15,000. **This is overfitting.** The model has 204,544 parameters and the training set has 220,980 target characters: almost one parameter per thing to predict. There is no dropout (a deliberate simplification, lesson 8). So it starts to learn the training names themselves.

Compare the GRU: train 1.9993, test 1.9988, still improving. No gap, so no sign of overfitting yet. More steps would likely help it. Same for the MLP (gap 0.04, still improving).

## 4. The result the raw table gets wrong

Lesson 1 found that 153 of the 1,000 test names also appear in the training set, because `names.txt` lists 2,539 names twice. A model that memorizes training names gets those 153 nearly "for free". I split each model's test loss by whether the name is one of those 153, using the saved best checkpoint. You can do the same for any checkpoint with `python course/labs/lab09_leak_split.py --type <type> <path>/model.pt`. "Benefit from the leak" = unseen loss − all-names loss: how much the duplicates flatter the reported number.

| Model | All 1,000 test names | 153 also in training | **847 truly unseen** | Benefit from the leak |
|---|---|---|---|---|
| bigram | 2.4496 | 2.4449 | **2.4504** | 0.001 |
| bow | 2.1532 | 2.1265 | **2.1576** | 0.004 |
| rnn | 2.0537 | 2.0111 | **2.0608** | 0.007 |
| mlp | 2.0305 | 1.9527 | **2.0435** | 0.013 |
| gru | 1.9986 | 1.9288 | **2.0103** | 0.012 |
| transformer | 1.9652 | **1.7402** | **2.0028** | **0.038** |

Read it carefully:
- **The Bigram is the control.** It can't memorize whole names (it sees one character). Its leaked and unseen numbers differ by only 0.006 (2.4449 vs 2.4504). So the leaked names are only very slightly easier in themselves, and the much larger gaps in the other rows (0.26 for the Transformer) are memorization.
- **The Transformer memorizes by far the most.** On the leaked names it scores 1.740, much better than on new names.
- **On truly unseen names, the Transformer's lead over the GRU shrinks from 0.033 to 0.0075.** That is small enough that I would not claim the Transformer is better than the GRU here from one run each. A second seed could reverse it. I didn't run one; that is the cheapest test that would settle it (section 7, exercise 1).

The honest summary: **on this data, with these defaults and 20,000 steps, the GRU and the Transformer generalize about equally well. The Transformer's apparent win in the raw table is mostly memorized duplicates.** The README's "test logprob of ~1.92" comes from the same split procedure, so the same contamination applies to it. I did not reach 1.92 in 20,000 steps with the defaults. My best was 1.966, and the test loss was already rising, so more steps of the same settings look unlikely to get there. (Different PyTorch versions or hardware can shift results slightly; I can't rule that out.)

> **Careful:** don't over-correct into "Transformers are no better". At this tiny scale (204k parameters, 31k short names, no dropout) the architecture's advantages barely show. The claim the evidence supports is narrow: *in this repo, with defaults, the ranking between the top two isn't established.*

## 5. Every setting, and which models it affects

| Flag | Default | What it changes | Used by |
|---|---|---|---|
| `--type` | `transformer` | Which model class | — |
| `--n-layer` | 4 | Number of Transformer blocks | transformer only |
| `--n-head` | 4 | Attention heads per block; must divide `--n-embd` | transformer only |
| `--n-embd` | 64 | Size of token/position embeddings and the residual stream | all except bigram |
| `--n-embd2` | 64 | MLP hidden size; RNN/GRU hidden-state size; BoW MLP hidden size | mlp, rnn, gru, bow — **not** transformer |
| `--batch-size` / `-b` | 32 | Names per step | all |
| `--learning-rate` / `-l` | 5e-4 | Step size (lesson 3) | all |
| `--weight-decay` / `-w` | 0.01 | Shrink every parameter by `lr × wd` per step | all |
| `--max-steps` | −1 (forever) | When to stop | all |
| `--top-k` | −1 (off) | Sampling restriction (lesson 4) | sampling only |
| `--seed` | 3407 | Split, initialization, batch order | all |
| `--device` | `cpu` | `cpu`, `cuda`, `cuda:N`, `mps` | all |
| `--num-workers` / `-n` | 4 | Background data-loading processes | all |
| `--resume` | off | Load `model.pt` and keep training (see lesson 10 for what it forgets) | all |
| `--sample-only` | off | Load `model.pt`, print 50 samples, exit | all |

> **Careful:** passing `--n-layer 8` with `--type gru` does nothing, silently. No warning tells you a flag is ignored.

## 6. Training on your own file

The format is one item per line. That's all. Before you run it, check these. Each is a real trap I reproduced or found in the code:

| Trap | What happens | Do this |
|---|---|---|
| **Fewer than 10 lines** | `int(n × 0.1) = 0` test examples, and `rp[:-0]` is an *empty* list in Python. Output: `split up the dataset into 0 training examples and 5 test examples`, then `RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=0`. | Use at least 10 lines. Realistically hundreds or more. |
| **Duplicate lines** | Same leak as `names.txt`: test loss looks better than it is. | Deduplicate first. Linux/macOS: `sort -u in.txt > out.txt`. |
| **Upper/lower case** | `Emma` and `emma` use different tokens. The vocabulary doubles and data per token halves. | Lowercase, unless case matters to you. |
| **Non-ASCII text on Windows** (ä, ö, ü, ß, é…) | Line 548 opens the file with `open(input_file, 'r')`, with no encoding given. Python then uses the system's default encoding, which on Windows is often a legacy code page like cp1252, unless Python's UTF-8 mode is on. A UTF-8 `ü` read as cp1252 becomes the two characters `Ã¼`. No error, just wrong tokens. I reproduced the decoding: `'Jürgen'` → `'JÃ¼rgen'`. | Run `python -X utf8 makemore.py ...`, which forces UTF-8 mode. Then check the printed `vocabulary:` line shows your real characters. |
| **One very long line** | `block_size` = longest line + 1. Every Transformer attention table is `block_size²` per head. The MLP's first layer grows with `block_size × n_embd`. | Look at `max word length` in the output. Drop or cut outliers. |
| **Spaces inside lines** | Kept, as a token. Only leading and trailing spaces are stripped (line 551). | Fine for multi-word items, e.g. company names. Just know it is a token. |
| **Very large file** | The test set stays at 1,000 lines (`min(1000, …)`). `contains()` does a linear search through a Python list for every sample printed. | Fine up to a few hundred thousand lines. Beyond that, expect slow sample printing. |

**Which model to start with:** the default Transformer, plus the GRU as a comparison, both on the same seed. Watch train **and** test loss. If they separate, the model is memorizing. Use a smaller `--n-embd`, or stop earlier (the best checkpoint is already kept).

---

## Lab 9: none of the six models can see the future

Every model claims that position `t` uses only positions `0..t`. Lab 9 tests it. It randomizes all parameters, so the test can't pass by accident. Then it changes the input at position 6 and measures how much the logits at positions 0–5 change.

```bash
python course/labs/lab09_all_models.py
```

**What you should see** (real output):

```
changed the input at position 6; a model that cannot see the future must leave positions 0..5 untouched
model          params  loss@init   max change pos<6   max change pos>=6
bigram            729     3.2958           0.000000            4.380754
mlp             69147     3.2666           0.000000           20.498747
rnn             11803     3.3356           0.000000           39.319603
gru             28315     3.3069           0.000000           30.223658
bow             12827     4.1517           0.000000          268.726654
transformer    204544     3.4140           0.000000           24.287218
ln(27) = 3.2958 is the loss of a uniform guess
```

**Decode it:**

| Column | Meaning |
|---|---|
| `params` | Matches each model's `model #params` and lessons 2 and 5–8. |
| `loss@init` | Before randomizing, with the normal starting weights. Bigram exactly ln 27, because its logits start at 0. The BoW's 4.15 is explained in lesson 7. |
| `max change pos<6` = `0.000000` everywhere | Changing position 6 has **no** effect on positions 0–5 in any model. The causal property holds. |
| `max change pos>=6` > 0 | Positions from 6 onward *do* react. The test isn't passing because nothing changes at all. |

Why it matters: a leak of future information would make training loss look excellent while generation fails, since at generation time the future doesn't exist yet. This is the test that would catch an error in a mask or a `roll`.

---

## 7. Exercises (predict first, then run)

I haven't run these. Write your prediction down before you run them.

1. **Is the Transformer really better than the GRU?** Run both with `--seed 1` and `--seed 2`, 20,001 steps each. Then get the unseen-name loss with `python course/labs/lab09_leak_split.py --type gru --seed 1 <dir>/model.pt` (the `--seed` must match, because the split depends on it). Does the order hold?
2. **Can you stop the overfitting?** Run `--type transformer --n-embd 32` (about a quarter of the parameters). Does the train/test gap at 20,000 steps shrink? Is the best test loss better or worse?
3. **Add a `--temperature` flag.** Add an `argparse` line, and pass `temperature=args.temperature` in `print_samples`. Sample at 0.5 and 1.5 and describe the difference.
4. **Deduplicate and retrain.** `sort -u names.txt > names_unique.txt`, then train the Transformer on it. The test set is now truly unseen. What best test loss do you get?

---

## What counts as success

| You can... | Test |
|---|---|
| Read a train/test curve for overfitting | Question 1 |
| Explain why the raw ranking was misleading | Question 2 |
| Name the ignored-flag trap | Question 3 |
| Prepare your own dataset safely | Question 4 |

## Check yourself

1. Train loss 1.71, test loss 1.98, test loss flat for 10,000 steps. What is happening, and what are two things you could try?
2. Why is the Bigram's row in the leak table the most important row?
3. You run `--type mlp --n-head 8`. What happens?
4. A colleague's file has 2,000 German street names in UTF-8, mixed case, with duplicates, and they are on Windows. List what they should do before training.

<details><summary>Answers</summary>

1. Overfitting: the model is fitting the training names, not the general patterns. Options: fewer parameters (`--n-embd 32`), stop around where test loss bottoms out (the saved checkpoint already does this), add dropout (code change), or get more or deduplicated data.
2. It is the control. The Bigram can't memorize names, so any difference between its leaked and unseen losses measures how much easier the leaked names are in themselves. That difference is small (0.006). So the large gaps for the other models, up to 0.26, are memorization.
3. Nothing different: the MLP ignores `n_head`, and no warning is printed.
4. Deduplicate. Lowercase (unless case matters). Run with `python -X utf8` and check the printed vocabulary shows `ä ö ü ß` correctly. Check `max word length` for outliers. With 2,000 lines the test set will be 200, so it trains fine, but expect noisier test numbers than with 1,000 test lines.

</details>

Next: [Lesson 10 — where this code is weak](10-critique.md).
