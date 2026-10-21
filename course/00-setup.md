# Lesson 0 — Install, first run, and how to read what it prints

**This lesson promises:** you will run `makemore.py` on your own machine, stop it, sample from it, and know what every line it prints means.

You don't need to understand the code yet. Lessons 1–4 explain all of it.

---

## Words used in this lesson

| Word | What it actually is |
|---|---|
| **Python** | The programming language the code is written in. The program `python` reads a `.py` file and executes it line by line. |
| **package** | A bundle of someone else's Python code that you install so your code can `import` it. |
| **pip** | The program that downloads and installs packages. |
| **virtual environment (venv)** | A folder holding its own copy of Python's package directory. Packages installed while it is active go into that folder only, not into your system Python. |
| **PyTorch (`torch`)** | The package that provides tensors, neural network layers, and automatic gradient computation. All six models are built from it. |
| **TensorBoard (`tensorboard`)** | A package that writes training curves to log files and shows them in a browser. `makemore.py` imports it on line 28, so it must be installed even if you never open it. |
| **CPU / GPU** | The CPU is your computer's main processor. A GPU is a separate processor that does many multiplications in parallel. `makemore.py` runs on the CPU by default. These models are small enough that the CPU is fine. |
| **CUDA / MPS** | CUDA is NVIDIA's interface for running code on NVIDIA GPUs. MPS is Apple's interface for the GPU in Apple Silicon Macs (M1 and later). |
| **step** | One update of the model's parameters, computed from one batch of 32 names. |
| **checkpoint (`model.pt`)** | A file holding all of the model's parameter values, so they can be loaded again later. |
| **working directory (`-o`)** | The folder where `makemore.py` writes `model.pt` and the TensorBoard logs. |

---

## What you physically need, and what each thing is for

| Thing | What it is for | Do you have it? |
|---|---|---|
| A computer running Windows, macOS or Linux | Runs everything | — |
| Python 3.10 or newer | PyTorch 2.14 (the version I tested with) requires Python ≥ 3.10 | Check in step 1 |
| About 5 GB free disk space on Linux, less on Windows/macOS | On Linux, `pip install torch` from the default package index pulled 1.2 GB of PyTorch plus 3.2 GB of NVIDIA GPU libraries when I installed it, even though no GPU was used | Check before step 3 |
| Permission to install packages and reach `pypi.org` | `pip` downloads PyTorch from there | Step 3 will tell you |
| A copy of this repository | Contains `makemore.py`, `names.txt`, the labs | You have it on GitHub |
| A GPU | **Not needed.** Optional speed-up only. | — |

---

## The hard blocker

**You must be able to install two packages, `torch` and `tensorboard`, into a Python 3.10+ environment.** No setting in the code gets around this. `makemore.py` imports both at the top (lines 23–28). It stops with `ModuleNotFoundError` before doing anything if either is missing.

The README says "PyTorch is the only requirement". That is incomplete. I reproduced it: without `tensorboard`, line 28 of `makemore.py` fails with `ModuleNotFoundError: No module named 'tensorboard'` (lab 10 shows this).

**If a company laptop blocks installing packages or blocks `pypi.org`**, use this fallback so the session isn't wasted:

- **Fallback A — Google Colab.** Colab is a free hosted Python notebook in the browser at <https://colab.research.google.com>, with a Google account. Upload `makemore.py` and `names.txt` with the file panel on the left. Then run `!python makemore.py -i names.txt -o names --max-steps 1001` in a cell. *I have not tested Colab in this session. It has historically shipped with `torch` and `tensorboard` preinstalled. If the import fails, run `!pip install tensorboard` in a cell first.* Check your company's policy before uploading anything to a Google service. `names.txt` is public data, but your own datasets may not be.
- **Fallback B — read-only.** Every lesson shows the real output I got. You can follow the course by reading, and run the labs later.

---

## Practice run (2–3 minutes, uses only what's in the repo)

### Step 1 — check Python

**Where:** a terminal. On Windows that is PowerShell, on macOS Terminal, on Linux any shell.

**Command:**

| OS | Command |
|---|---|
| macOS / Linux | `python3 --version` |
| Windows | `py --version` (if that fails: `python --version`) |

**What you should see:** `Python 3.10.x` or higher, for example `Python 3.13.16`.

**Why this step exists:** an older Python makes `pip` fail later with the confusing message "No matching distribution found for torch". That message really means "no PyTorch build exists for this Python version".

### Step 2 — get the code and go into its folder

```bash
git clone https://github.com/ArashjamehbozorgJung/makemore.git
cd makemore
```

**What you should see:** `ls` (macOS/Linux) or `dir` (Windows) lists `makemore.py`, `names.txt`, `README.md`, `course`.

**Why:** `makemore.py` opens `names.txt` by a relative path, so the terminal must be in this folder. Otherwise you get `FileNotFoundError`.

### Step 3 — make a virtual environment and install the two packages

| OS | Commands |
|---|---|
| macOS / Linux | `python3 -m venv .venv` then `source .venv/bin/activate` then `pip install torch tensorboard` |
| Windows (PowerShell) | `py -m venv .venv` then `.venv\Scripts\Activate.ps1` then `pip install torch tensorboard` |

**What you should see:** your prompt now starts with `(.venv)`. `pip` ends with `Successfully installed ... tensorboard-... torch-...`.

**Why a virtual environment:** it keeps these large packages out of your system Python. You can delete the whole install by deleting the `.venv` folder.

> **Careful (Linux):** plain `pip install torch` on Linux downloads the GPU build, about 4.4 GB installed, even if you have no GPU. If disk space is tight, pytorch.org's "Get Started" page generates a CPU-only install command for your exact OS. I could not test that CPU-only command here, because its download server was blocked in my environment.

> **Careful (Windows):** if PowerShell refuses to run `Activate.ps1` with a message about "execution policy", that is a Windows security setting, not a Python problem. Ask your IT team, or use `cmd.exe` and run `.venv\Scripts\activate.bat` instead.

**Check it worked:**

```bash
python -c "import torch, tensorboard; print(torch.__version__)"
```

You should see a version number such as `2.14.1`. Any `ModuleNotFoundError` means step 3 didn't finish. See the table further down.

### Step 4 — the practice run: 1,001 training steps

```bash
python makemore.py -i names.txt -o practice --max-steps 1001
```

**Why `--max-steps 1001` and not 1000:** the program evaluates and saves the model when the step number is a multiple of 500 and greater than 0. Steps are numbered from 0. With 1001 steps the last one is step 1000, so you get an evaluation and a saved `model.pt` at the end. With `--max-steps 1000` the last step is 999 and the final evaluation never happens.

**What you should see:** it took 25 seconds on my 4-core cloud machine. Your time will differ with your hardware. The output looks like this (my real output, shortened):

```
{'input_file': 'names.txt', 'work_dir': 'practice', 'resume': False, ..., 'type': 'transformer', ..., 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 32033
max word length: 15
number of unique characters in the vocabulary: 26
vocabulary:
abcdefghijklmnopqrstuvwxyz
split up the dataset into 31033 training examples and 1000 test examples
dataset determined that: vocab_size=27, block_size=16
number of parameters: 0.20M
model #params: 204544
.../dataloader.py:1102: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
step 0 | loss 3.5123 | step time 69.41ms
step 10 | loss 2.8783 | step time 18.30ms
step 20 | loss 2.7156 | step time 18.29ms
...
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
...
step 500 train loss: 2.2132... test loss: 2.1971...
test loss 2.1971... is the best so far, saving model to practice/model.pt
...
step 1000 train loss: 2.144432544708252 test loss: 2.1338284015655518
test loss 2.1338284015655518 is the best so far, saving model to practice/model.pt
--------------------------------------------------------------------------------
1 samples that are in train:
ariona
0 samples that are in test:
9 samples that are new:
killea
lomira
noren
braqan
brsen
daskima
koon
michayn
carge
--------------------------------------------------------------------------------
```

### Decode the output, line by line

| Line | What it means | Where in the code |
|---|---|---|
| `{'input_file': ... }` | Every command-line setting, including the defaults you didn't type. `'type': 'transformer'` means the default model is the Transformer. | `print(vars(args))`, line 620 |
| `number of examples in the dataset: 32033` | `names.txt` has 32,033 non-empty lines. | line 555 |
| `max word length: 15` | The longest name has 15 letters (`muhammadibrahim`, `muhammadmustafa`). | line 556 |
| `number of unique characters ...: 26` and `abcdefghijklmnopqrstuvwxyz` | Every character that appears anywhere in the file, sorted. | lines 557–559 |
| `split up ... 31033 training examples and 1000 test examples` | 1,000 random names are held back to measure the model on names it never trained on. | lines 562–566 |
| `vocab_size=27, block_size=16` | 27 = 26 letters + 1 special token. 16 = longest name + 1 slot for the start token. Lesson 1 explains both. | line 632 |
| `number of parameters: 0.20M` | Printed inside the Transformer's constructor. It counts 202,816 numbers and **leaves out** the final output layer. | line 131 |
| `model #params: 204544` | The true total, all parameters included. | line 653 |
| `UserWarning: 'pin_memory' ...` | **Harmless on a CPU.** The data loader is asked to use a GPU-transfer optimisation (`pin_memory=True`, line 665) and there is no GPU. Ignore it. | line 665 |
| `step 0 \| loss 3.5123 \| step time 69.41ms` | Step number, the loss on this step's batch of 32 names, and how long the step took. Printed every 10 steps. The first step is slow because of one-time setup. | line 694 |
| loss going 3.51 → 2.88 → 2.72 ... | The model is learning. Lesson 2 explains why a "know nothing" model scores about 3.30 and why this one started a bit worse, at 3.51. | — |
| The `----` blocks with samples | Every 200 steps, 10 newly generated names. "in train" means the generated name exists in the training list. "in test" means it exists only in the test list. "new" means it's in neither. | `print_samples`, lines 459–485 |
| `step 500 train loss: ... test loss: ...` | Every 500 steps, a more careful loss measurement on 1,100 training names and on all 1,000 test names. | lines 697–703 |
| `... is the best so far, saving model to practice/model.pt` | The test loss is the lowest seen so far, so the parameters are written to disk. A worse result is not saved. | lines 705–709 |

Notice the samples at step 1000: `lomira`, `noren`, `daskima`. They sound like names after 25 seconds. That is a 204,544-number function that has seen about 32,000 names (1,000 steps × 32).

---

## The real thing

### Step 5 — train without a step limit

```bash
python makemore.py -i names.txt -o names
```

**This never stops by itself.** The default `--max-steps` is `-1`, which means "forever" (line 717). Watch the `test loss` lines. When they stop improving for a few thousand steps, press **Ctrl+C**. That is safe. The best model was already saved to `names/model.pt` at the last improvement. You only lose the steps since the last multiple of 500.

What to expect, from my run: the test loss was 2.13 at step 1,000, 2.01 at step 5,000, 1.97 at step 10,000, and its best value was 1.966 at step 15,000. After that it got slightly worse while the training loss kept falling. Lesson 9 explains why.

### Step 6 — sample from the saved model

```bash
python makemore.py -i names.txt -o names --sample-only
```

**What you should see:** the same setup lines, then `resuming from existing model in the workdir`, then 50 names sorted into "in train", "in test" and "new". It exits without training. My 20,000-step model gave, for example: `maddoxxz habab farrena ellizavder zanyam adanson ziyaan galirah tallis khaizagr jessin lashela`.

> **Careful:** `--sample-only` rebuilds the model from the command-line flags and then loads `model.pt` into it. If you trained with `--type bigram` or `--n-layer 6`, you must pass exactly the same flags again when sampling. Nothing records them for you. If you forget, you get a wall of `Missing key(s) in state_dict` (see the table below).

### Step 7 (optional) — look at the training curves

**What it buys you:** a graph of train and test loss over time instead of reading numbers. It is the fastest way to see overfitting (lesson 9).

```bash
tensorboard --logdir names
```

Open <http://localhost:6006> in a browser. You will see two curves, `Loss/train` and `Loss/test`, with one point every 500 steps. Stop TensorBoard with Ctrl+C.

---

## If it does not work

| What you see | Most likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | The virtual environment isn't active in this terminal, or step 3 failed | Run the `activate` command from step 3 again, then `pip install torch tensorboard` |
| `ModuleNotFoundError: No module named 'tensorboard'` | Only PyTorch was installed (the README suggests that is enough; it isn't) | `pip install tensorboard` |
| `ERROR: No matching distribution found for torch` | Your Python is too old (needs 3.10+), or `pypi.org` is blocked | Check step 1. If Python is fine, it is a network block: use Fallback A. |
| `FileNotFoundError: [Errno 2] No such file or directory: 'name.txt'` | Typo in `-i`, or the terminal isn't in the repo folder | `cd` into the folder that contains `names.txt` |
| `RuntimeError: Error(s) in loading state_dict for Transformer: Missing key(s) in state_dict: "transformer.wte.weight", ... Unexpected key(s) in state_dict: "logits".` | You trained one model type (here `bigram`, whose only parameter is called `logits`) and sampled with another (default `transformer`) | Repeat the exact `--type` and size flags you trained with |
| `RuntimeError: Found no NVIDIA driver on your system` | `--device cuda` on a machine without an NVIDIA GPU | Remove `--device cuda`. The CPU default works. |
| `AssertionError: Torch not compiled with CUDA enabled` | `--device cuda` with a CPU-only PyTorch build | Same: remove `--device cuda`, or install a CUDA build from pytorch.org |
| `RuntimeError: PyTorch is not linked with support for mps devices` | `--device mps` on anything other than an Apple Silicon Mac | Remove `--device mps` |
| It never stops | By design: `--max-steps` defaults to `-1`, meaning forever | Ctrl+C, or pass `--max-steps N` |
| An error mentioning DataLoader workers or multiprocessing | The 4 background data-loading processes (`--num-workers 4` default) failed on your system | Add `-n 0` to load data in the main process. These datasets are tiny, so it costs almost no speed. |
| The `pin_memory` UserWarning | Not an error | Ignore |

---

## What counts as success

| You have... | How you know |
|---|---|
| A working install | `python -c "import torch, tensorboard"` prints nothing and returns |
| Completed the practice run | `practice/model.pt` exists (about 0.8 MB) and the last test loss is roughly 2.1–2.2 (your exact number will differ slightly) |
| Read the output | You can say, without looking, what `vocab_size=27`, `block_size=16`, `0.20M` vs `204544`, and "best so far, saving" mean |
| Sampled from a saved model | `--sample-only` printed 50 names and exited |

---

## Check yourself

1. You run `python makemore.py -i names.txt -o run1 --max-steps 500`. Will `run1/model.pt` exist afterwards?
2. Why does the Transformer print two different parameter counts?
3. You trained with `--type gru -o g`. What exact command samples from it?

<details><summary>Answers</summary>

1. **No.** Saving happens only at evaluation, when `step > 0 and step % 500 == 0`. With `--max-steps 500` the steps are 0…499, so step 500 never runs and nothing is saved. You need `--max-steps 501`.
2. `0.20M` is printed by the Transformer's constructor. It counts only `self.transformer.parameters()`, which leaves out `lm_head` (64 × 27 = 1,728 numbers). `model #params: 204544` counts everything. 204,544 − 1,728 = 202,816 ≈ 0.20M.
3. `python makemore.py -i names.txt -o g --type gru --sample-only`. Without `--type gru` it builds a Transformer and the load fails.

</details>

Next: [Lesson 1 — the data](01-data.md).
