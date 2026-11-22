#!/usr/bin/env python
# coding: utf-8
"""Centralized baseline: the federated CNN trained on all six clients' data at once.

Everything except the training itself is the federated setup's own code, so
the two can be compared row for row:

- model, features and scaling: ``federated-cnn/Server`` (``model_cav``,
  ``common_cav``), imported rather than copied;
- data: each Pi's ``split_cache/<dataset>_v1.npz`` (``federated-cnn/PiN``),
  with each client's own local train / validation split
  (``get_client_data``); the six train parts are pooled and shuffled;
- training: batch ``BATCH_SIZE`` (256), one epoch per FL round
  (``NUM_ROUNDS`` = 15), so every row is seen as often as in FL, and the
  learning rate falls 1e-3 -> 1e-4 over the epochs as it does over the rounds
  (``model_cav.learning_rate``);
- evaluation: the server's ``Evaluator``. After every epoch it scores the six
  test sets at threshold 0.5. After the last epoch it computes the four final
  results at 0.5 and at the median of the six datasets' proposed thresholds,
  each from that dataset's validation normals as a Pi computes it
  (``THRESHOLD_TARGET_FPR``). The Traverse sets also get a threshold
  calibrated on that vehicle.

The output CSVs have the server's names and columns
(round_metrics, dataset_metrics, final_results, final_dataset_metrics,
generalization_files); "round" is the epoch. They go to
``centralized/results/seed<N>/`` with progress_log.txt, run.json and the
weights after every epoch (``weights/round_NN.weights.h5``, as the server saves them).

usage: python train_central.py [--seed 0] [--epochs 15] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(os.path.dirname(HERE), "federated-cnn", "Server")
sys.path.insert(0, SERVER_DIR)  # Server/ first: its common_cav / model_cav; server_cav adds the Pi folders

import server_cav as srv  # noqa: E402
from model_cav import (BATCH_SIZE, THRESHOLD_TARGET_FPR, BatchSequence, build_model,  # noqa: E402
                       format_duration, learning_rate, set_learning_rate)
from tensorflow import keras  # noqa: E402

LOCAL_VAL_SIZE = 0.2  # as on every Pi (PiN/*_client_cav.py)
OUTPUT_FILES = ("ROUND_METRICS_CSV", "DATASET_METRICS_CSV", "FINAL_RESULTS_CSV",
                "FINAL_DATASETS_CSV", "GENERALIZATION_FILES_CSV", "PROGRESS_LOG", "WEIGHTS_DIR")

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--seed", type=int, default=0, help="weight init, dropout and shuffling")
parser.add_argument("--epochs", type=int, default=srv.NUM_ROUNDS, help="default: the FL round count")
parser.add_argument("--out", default=None, help="default: centralized/results/seed<N>")
args = parser.parse_args()

out_dir = args.out or os.path.join(HERE, "results", f"seed{args.seed}")
if os.path.isdir(out_dir) and os.listdir(out_dir):
    sys.exit(f"{out_dir} is not empty (the CSVs are appended to): remove it or pass --out")
os.makedirs(out_dir, exist_ok=True)
# The Evaluator writes to these module paths (and the weights after every
# epoch to weights/round_NN.weights.h5); point them into out_dir.
for name in OUTPUT_FILES:
    setattr(srv, name, os.path.join(out_dir, os.path.basename(getattr(srv, name))))
log = srv.log_progress
keras.utils.set_random_seed(args.seed)

# ---- training data: each client's local train / validation split, as on its Pi
x_parts, y_parts, val_sets = [], [], []
for dataset_name, module in srv.DATASETS:
    x_tr, x_val, y_tr, y_val = module.get_client_data(srv.TEST_SIZE, srv.RANDOM_STATE, LOCAL_VAL_SIZE)
    x_parts.append(x_tr); y_parts.append(y_tr)
    val_sets.append((dataset_name, srv.common_cav.reshape_for_cnn(x_val), y_val))
    log(f"[DATA] {dataset_name}: train {len(y_tr)} rows ({y_tr.mean():.3f} attack), validation {len(y_val)}")
x_train = srv.common_cav.reshape_for_cnn(np.concatenate(x_parts))
y_train = np.concatenate(y_parts)
del x_parts, y_parts, x_tr, x_val, y_tr, y_val

# ---- test data: exactly what the server loads
x_parts, y_parts, w_parts, dataset_names, dataset_rows = [], [], [], [], []
for dataset_name, module in srv.DATASETS:
    x_part, y_part, w_part = srv.load_test_split(module)
    x_parts.append(x_part); y_parts.append(y_part); w_parts.append(w_part)
    dataset_names.append(dataset_name); dataset_rows.append(len(y_part))
x_test = srv.common_cav.reshape_for_cnn(np.concatenate(x_parts))
y_test = np.concatenate(y_parts).astype(np.int8)
w_test = np.concatenate(w_parts).astype(np.float32)
del x_parts, y_parts, w_parts, x_part, y_part, w_part

gen_paths = srv.cantt_utils_cav.generalization_cache_paths(srv.TEST_SIZE, srv.RANDOM_STATE)
run_generalization = all(gen_paths.values())
if not run_generalization:
    missing = [srv.cantt_utils_cav.GEN_DATASET_KEYS[n] for n, p in gen_paths.items() if p is None]
    log(f"[CACHE] results 2-4 will be SKIPPED: Pi4/split_cache lacks {', '.join(missing)}")

model = build_model((x_train.shape[1], x_train.shape[2]))
evaluator = srv.Evaluator(model, x_test, y_test, w_test, dataset_names, dataset_rows, run_generalization)
evaluator.evaluate_round(0, model.get_weights(), {})  # checks the evaluation code before training

train_batches = BatchSequence(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
train_batches.rng = np.random.default_rng(args.seed)  # reproducible shuffling
train_batches.order = train_batches.rng.permutation(len(x_train))

log(f"central training: {len(y_train)} rows ({y_train.mean():.3f} attack), batch {BATCH_SIZE}, "
    f"{len(train_batches)} steps per epoch, {args.epochs} epochs, seed {args.seed}")
history = []
for epoch in range(1, args.epochs + 1):
    rate = learning_rate(epoch, args.epochs)
    set_learning_rate(model, rate)
    t_start = time.time()
    fit = model.fit(train_batches, epochs=1, verbose=2)
    fit_seconds = time.time() - t_start
    _, metrics = evaluator.evaluate_round(epoch, model.get_weights(), {})
    history.append({"epoch": epoch, "learning_rate": rate, "train_loss": fit.history["loss"][0],
                    "fit_seconds": fit_seconds, **metrics})
    log(f"epoch {epoch}/{args.epochs}: learning rate {rate:.2e}, train loss {fit.history['loss'][0]:.5f}, "
        f"fit {format_duration(fit_seconds)}; macro detection {metrics['macro_detection_rate']:.4f}, "
        f"fpr {metrics['macro_fpr']:.5f}, PR-AUC {metrics['macro_pr_auc']:.4f} (threshold 0.5)")

# ---- the final results, with the thresholds the six clients would propose
thresholds = {}
for dataset_name, x_val, y_val in val_sets:
    p_val = srv._predict_attack(model, x_val)
    thresholds[dataset_name] = float(np.quantile(p_val[y_val == 0], 1 - THRESHOLD_TARGET_FPR))
evaluator.evaluate_final(args.epochs, thresholds)

with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
    json.dump({"seed": args.seed, "epochs": args.epochs, "batch_size": BATCH_SIZE, "train_rows": int(len(y_train)),
               "thresholds": thresholds, "history": history}, f, indent=1)
log(f"done; results in {out_dir}")
