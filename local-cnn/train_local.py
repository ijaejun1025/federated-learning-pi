#!/usr/bin/env python
# coding: utf-8
"""Local baseline: the federated CNN trained on one client's data only, as
that Pi would train alone, with no federated learning.

It runs ``centralized-cnn/train_central.py --client PiN``, so the model,
features, batch, epochs, learning rate and evaluation are the centralized
baseline's (and the FL run's) by construction; only the training data
differ. Output: ``local-cnn/results/PiN/`` (another seed needs ``--out``).

usage: python train_local.py --client Pi3 [--seed 0] [--epochs 15] [--out DIR]
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CENTRAL = os.path.join(os.path.dirname(HERE), "centralized-cnn", "train_central.py")

if "--client" not in sys.argv:
    sys.exit("usage: python train_local.py --client Pi3 [--seed 0] [--epochs 15] [--out DIR]")
sys.argv = [TRAIN_CENTRAL] + sys.argv[1:]
runpy.run_path(TRAIN_CENTRAL, run_name="__main__")
