"""
shared setup for the labs: makes `import makemore` work no matter which
directory you run a lab from, and loads the same train/test split that
makemore.py itself would produce with its default seed.
"""

import os
import sys
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch
import makemore

NAMES = os.path.join(REPO, 'names.txt')

def load_data(seed=3407):
    # makemore.py calls torch.manual_seed(args.seed) right before create_datasets,
    # and 3407 is its default seed, so this reproduces its exact train/test split
    torch.manual_seed(seed)
    return makemore.create_datasets(NAMES)

def show(ix):
    # display helper: token 0 (<START>/<STOP>) is printed as '.', everything else decoded
    return ''.join('.' if i == 0 else makemore_itos[i] for i in ix)

def set_args(**kwargs):
    # evaluate() and print_samples() in makemore.py read a global called `args`;
    # when we import makemore instead of running it, that global does not exist yet
    makemore.args = argparse.Namespace(**kwargs)

makemore_itos = {i+1: ch for i, ch in enumerate('abcdefghijklmnopqrstuvwxyz')}
