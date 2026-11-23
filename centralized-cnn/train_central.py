#!/usr/bin/env python
# coding: utf-8
"""Centralized and local baselines: the federated CNN trained without FL.

- centralized (default): on all six clients' local train splits at once;
- local (``--client Pi3``, run through ``local-cnn/train_local.py``): on one
  client's local train split only, as that Pi would train alone. Its final
  threshold is that client's own proposed one (the "median" of one), and it
  is scored on all six test sets, so the six local runs give a 6x6
  train-client x test-set matrix.

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
  calibrated on that vehicle;
- hardware: ``hardware_cav``, as on the Pis and the FL server. Run on the
  Jetson (the FL server), after the FL run, so its numbers sit next to the
  FL server's: one ``fit`` and one ``evaluate`` row per epoch, and an
  ``inference`` row at the end (batch 256 and one frame at a time).

The output CSVs have the server's names and columns
(round_metrics, dataset_metrics, final_results, final_dataset_metrics,
generalization_files, hardware_metrics); "round" is the epoch. They go to
``centralized-cnn/results/`` (local: ``local-cnn/results/PiN/``; the seed is in
run.json, and another seed needs ``--out``, e.g. ``--out results/seed1``)
with progress_log.txt, run.json and the weights after every epoch
(``weights/round_NN.weights.h5``, as the server saves them).

usage: python train_central.py [--client PiN] [--seed 0] [--epochs 15] [--out DIR]
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

import hardware_cav  # noqa: E402
import server_cav as srv  # noqa: E402
from model_cav import (BATCH_SIZE, THRESHOLD_TARGET_FPR, BatchSequence, build_model,  # noqa: E402
                       format_duration, learning_rate, set_learning_rate)
from tensorflow import keras  # noqa: E402

LOCAL_VAL_SIZE = 0.2  # as on every Pi (PiN/*_client_cav.py)
OUTPUT_FILES = ("ROUND_METRICS_CSV", "DATASET_METRICS_CSV", "FINAL_RESULTS_CSV",
                "FINAL_DATASETS_CSV", "GENERALIZATION_FILES_CSV", "PROGRESS_LOG", "WEIGHTS_DIR",
                "HARDWARE_METRICS_CSV")

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--client", default=None, help="local baseline: train on this client only (Pi1 ... Pi6)")
parser.add_argument("--seed", type=int, default=0, help="weight init, dropout and shuffling")
parser.add_argument("--epochs", type=int, default=srv.NUM_ROUNDS, help="default: the FL round count")
parser.add_argument("--out", default=None, help="default: centralized-cnn/results, local: local-cnn/results/PiN")
args = parser.parse_args()

if args.client is None:
    train_datasets = srv.DATASETS
    run_dir = os.path.join(HERE, "results")
else:
    train_datasets = [(name, module) for name, module in srv.DATASETS
                      if name.split("-")[0].lower() == args.client.lower() or name.lower() == args.client.lower()]
    if len(train_datasets) != 1:
        sys.exit(f"--client {args.client}: expected one of {', '.join(n.split('-')[0] for n, _ in srv.DATASETS)}")
    run_dir = os.path.join(os.path.dirname(HERE), "local-cnn", "results", train_datasets[0][0].split("-")[0])
out_dir = args.out or run_dir
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
# (local baseline: that one client's)
x_parts, y_parts, val_sets = [], [], []
for dataset_name, module in train_datasets:
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

mode = "central" if args.client is None else f"local ({train_datasets[0][0]} only)"
node = "central" if args.client is None else f"local-{train_datasets[0][0].split('-')[0]}"
device_info = hardware_cav.device_info()
log(f"{mode} training: {len(y_train)} rows ({y_train.mean():.3f} attack), batch {BATCH_SIZE}, "
    f"{len(train_batches)} steps per epoch, {args.epochs} epochs, seed {args.seed}")
history = []
for epoch in range(1, args.epochs + 1):
    rate = learning_rate(epoch, args.epochs)
    set_learning_rate(model, rate)
    t_start = time.time()
    meter = hardware_cav.HardwareMeter()
    fit = model.fit(train_batches, epochs=1, verbose=2)
    fit_seconds = time.time() - t_start
    hw_fit = meter.stop(rows=len(x_train))
    hw_fit.update(device_info)
    srv._record_hardware(epoch, node, "fit", hw_fit)
    meter = hardware_cav.HardwareMeter()
    _, metrics = evaluator.evaluate_round(epoch, model.get_weights(), {})
    srv._record_hardware(epoch, node, "evaluate", meter.stop(rows=len(x_test)))
    history.append({"epoch": epoch, "learning_rate": rate, "train_loss": fit.history["loss"][0],
                    "fit_seconds": fit_seconds, **metrics})
    log(f"epoch {epoch}/{args.epochs}: learning rate {rate:.2e}, train loss {fit.history['loss'][0]:.5f}, "
        f"fit {format_duration(fit_seconds)}; macro detection {metrics['macro_detection_rate']:.4f}, "
        f"fpr {metrics['macro_fpr']:.5f}, PR-AUC {metrics['macro_pr_auc']:.4f} (threshold 0.5)"
        f"{srv._hardware_summary(hw_fit)}")

# ---- the final results, with the thresholds the six clients would propose
# (local baseline: only that client's own, so the "median" is that threshold)
thresholds = {}
for dataset_name, x_val, y_val in val_sets:
    p_val = srv._predict_attack(model, x_val)
    thresholds[dataset_name] = float(np.quantile(p_val[y_val == 0], 1 - THRESHOLD_TARGET_FPR))
evaluator.evaluate_final(args.epochs, thresholds)

# ---- inference on this machine, as each Pi measures it in its evaluate:
# batch 256 over the (first) training client's validation set, then one
# frame at a time
x_val0 = val_sets[0][1]
t_start = time.time()
model.predict(BatchSequence(x_val0), verbose=0)
hw_infer = {"hw_rows": len(x_val0), "hw_infer_batch_ms_per_row": 1000.0 * (time.time() - t_start) / len(x_val0)}
hw_infer.update(hardware_cav.frame_latency_ms(model, x_val0))
srv._record_hardware(args.epochs, node, "inference", hw_infer)
log(f"inference: {hw_infer['hw_infer_batch_ms_per_row']:.4f} ms per row at batch {BATCH_SIZE}"
    f"{srv._hardware_summary(hw_infer)}")

with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
    json.dump({"mode": mode, "seed": args.seed, "epochs": args.epochs, "batch_size": BATCH_SIZE, "train_rows": int(len(y_train)),
               "thresholds": thresholds, "history": history}, f, indent=1)
log(f"done; results in {out_dir}")
