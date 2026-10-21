# makemore, from zero to hero

**What this course gives you:** after it, you can explain every line of `makemore.py` and predict what it prints. You can train all six of its models, and you can say where this code is weak.

Every number in these lessons was produced by running the code. Each lesson has a lab script in `course/labs/` so you can check the claims yourself. Where your numbers will differ from mine, the lesson says so and says why.

---

## Before you start: what this course assumes

| You need | Why | If you don't have it |
|---|---|---|
| You can read basic Python: variables, lists, `for` loops, functions, classes | The whole repo is Python. This course explains PyTorch and machine learning, not Python. | Say so and I will add a Python lesson 0. Don't push through without it; nothing after this will make sense. |
| School algebra: functions, exponent `e^x`, logarithm `log` | Loss and probabilities are built from `exp` and `log`. | Lesson 2 defines each one at the point it is used. |
| No calculus | Lesson 3 explains gradients without derivations. It checks one numerically instead. | — |
| A computer where you may install Python packages | You need it to run the labs. | See the hard blocker in [lesson 0](00-setup.md). |

---

## Words used in this document

These are the words that appear in every lesson. Each lesson defines its own new words at the top.

| Word | What it actually is, mechanically |
|---|---|
| **token** | One unit of text the model reads or writes. Here it is one character, `a`–`z`, or the special token `0`. |
| **vocabulary** | The fixed list of all tokens. Here it has 27 entries: 26 letters plus token `0`. |
| **tensor** | A block of numbers with a shape, like a list of lists of lists. Shape `(32, 16)` means 32 rows of 16 numbers. PyTorch's basic data type. |
| **parameter** | A number inside the model that training is allowed to change. "The model has 204,544 parameters" means 204,544 such numbers. |
| **model** | A function that takes token integers in and puts scores for the next token out. Its behaviour is set by its parameters. |
| **logits** | The raw scores a model outputs: 27 real numbers per position, one per possible next token. Higher means "more likely". They are not probabilities yet. |
| **softmax** | The formula that turns logits into probabilities: `exp(each) / sum(exp(all))`. |
| **loss** | One number that measures how wrong the model's predictions were. Lower is better. Training is the process of making it smaller. |
| **training** | Repeating: compute the loss, compute which way each parameter should move to lower it, move each parameter a small step that way. |
| **sampling** | Generating new text: repeatedly pick a next token at random according to the model's probabilities. |
| **autoregressive** | Each prediction is fed back in as input for the next prediction, one token at a time. |
| **character-level language model** | A model that assigns probabilities to the next character given the characters before it. That is all makemore is. |

---

## What you physically have, and what each thing is for

| File | Size | What it is for |
|---|---|---|
| `makemore.py` | 719 lines | The entire program: data loading, six models, training loop, sampling. |
| `names.txt` | 32,033 lines | The training data: one lowercase name per line. |
| `README.md` | 81 lines | The author's short usage notes. |
| `course/NN-*.md` | — | These lessons. |
| `course/labs/labNN_*.py` | — | One runnable script per lesson. Each prints the thing that lesson claims. |

`makemore.py` is written by Andrej Karpathy. It is the finished, single-file version of what lectures 2–6 of his free video course "Neural Networks: Zero to Hero" build up step by step. The videos and this file are **not** the same code. The videos use notebooks, BatchNorm (a normalization layer) and a WaveNet-style model, and none of those are in this file. This course teaches the file you have. Lesson links to the videos are optional extra viewing. Syllabus: <https://github.com/karpathy/nn-zero-to-hero>.

---

## How the file is laid out, and why we read it backwards

`makemore.py` puts the most complex model at the top and the simplest at the bottom. Learning goes the other way. The lessons read it bottom-up:

| Lesson | Topic | Lines in `makemore.py` | Lab |
|---|---|---|---|
| [0](00-setup.md) | Install, first run, read the output | 594–718 (run it, don't read it yet) | — |
| [1](01-data.md) | Names become integers: the dataset | 503–572 | `lab01_data.py` |
| [2](02-bigram-and-loss.md) | The simplest model, logits, softmax, cross-entropy loss | 396–423 | `lab02_bigram.py` |
| [3](03-training.md) | How training changes parameters: gradients, AdamW, the loop | 574–591, 594–718 | `lab03_training.py` |
| [4](04-sampling.md) | Making new names: generate, temperature, top-k | 425–501 | `lab04_sampling.py` |
| [5](05-mlp.md) | MLP: embeddings and a fixed window of context | 347–394 | `lab05_mlp.py` |
| [6](06-rnn-gru.md) | RNN and GRU: a memory carried forward | 253–345 | `lab06_rnn.py` |
| [7](07-bow.md) | Bag of Words: averaging the past | 158–251 | `lab07_bow.py` |
| [8](08-transformer.md) | Transformer: attention, the full GPT-2 block | 30–156 | `lab08_transformer.py` |
| [9](09-experiments.md) | All six models compared; training on your own data | whole file | `lab09_all_models.py`, `lab09_leak_split.py` |
| [10](10-critique.md) | Where this code is weak, and what is intended | whole file | `lab10_findings.py` |

Each lesson ends with **check yourself** questions. The answers are folded underneath. Answer before you open them. If you get one wrong, the lesson has not landed yet. Go back rather than forward.

---

## How to use this course with me

1. Read one lesson.
2. Run its lab. Compare your output with the output in the lesson.
3. Answer the check-yourself questions without looking.
4. Tell me your answers, and anything that didn't match. I will push on the weak ones.

Don't read lesson 8 first because "Transformers are the interesting part". Lesson 8 is short only because lessons 1–7 already built every piece it uses.

---

## What counts as success for the whole course

| You can... | Tested in |
|---|---|
| Write down `x` and `y` for any name, by hand | Lesson 1 |
| Say why the first loss printed is about 3.3, and compute it | Lesson 2 |
| Say what `loss.backward()` and `optimizer.step()` change, and what they don't | Lesson 3 |
| Predict how temperature and top-k change the output | Lesson 4 |
| Trace the shape of a tensor through any of the six models | Lessons 5–8 |
| Explain what attention can do that the bag-of-words average can't | Lesson 8 |
| Pick a model and settings for your own data and defend the choice with numbers | Lesson 9 |
| Name three real weaknesses of this code, and say which "weaknesses" are deliberate simplifications | Lesson 10 |
