#!/usr/bin/env python
# coding: utf-8

import datetime
import glob
import time
import flwr as fl
import numpy as np
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LR_IDS_DIR = os.path.dirname(BASE_DIR)

# Each dataset's utils module lives inside its own PiN/ folder (self-contained
# so that folder alone can be copied onto that Raspberry Pi), a sibling of
# this Server/ folder. The server runs centrally on the Jetson and needs all
# 6, so append every PiN/ folder to the import path. Appending (not inserting
# at the front) keeps this directory's own common_cav.py/model_cav.py as the
# ones actually used below, even though every PiN/ folder also carries its
# own copy for standalone Pi use.
for _pi_folder in ("Pi1", "Pi2", "Pi3", "Pi4", "Pi5", "Pi6"):
    sys.path.append(os.path.join(LR_IDS_DIR, _pi_folder))

import canmirgu_utils_cav
import cantt_utils_cav
import carhacking_utils_cav
import common_cav
import otids_utils_cav
import road_utils_cav
import survival_utils_cav
from model_cav import build_model, format_duration

NUM_ROUNDS = 3
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
DATASET_METRICS_CSV = os.path.join(BASE_DIR, "dataset_metrics.csv")
# Generalization sets (Pi4: Forester unseen attacks, Traverse = a vehicle no
# client trains on). Evaluated once, after the last round, and kept out of
# round_metrics.csv / dataset_metrics.csv.
GENERALIZATION_CSV = os.path.join(BASE_DIR, "generalization_metrics.csv")
GENERALIZATION_FILES_CSV = os.path.join(BASE_DIR, "generalization_files.csv")
# For a new vehicle, thresholds are also set from its own attack-free
# recording (no labels): the score quantile that would give these FPRs there.
CALIBRATED_FPR_TARGETS = (0.01, 0.001)
CLIENT_METRICS_CSV = os.path.join(BASE_DIR, "client_metrics.csv")
PROGRESS_LOG = os.path.join(BASE_DIR, "progress_log.txt")
TEST_SIZE = 0.33
RANDOM_STATE = 41
# Longest observed client round (Pi2, ~6.4h fit) plus margin, so a dropped
# client ends its round instead of hanging the server indefinitely.
ROUND_TIMEOUT_SEC = 10 * 3600
# Rows per Keras call during server evaluation. Keras/TF need several hundred
# bytes per row of the chunk on top of the data, so this sets the evaluation's
# extra memory. A multiple of Keras' default batch size (32), so the batches
# are the same as in one unchunked call.
EVAL_CHUNK_ROWS = 262_144
# Keras clips probabilities to [eps, 1 - eps] before taking the log in its
# cross-entropy; the per-dataset loss below does the same.
KERAS_EPSILON = 1e-7


def log_progress(message: str) -> None:
    """Timestamped line to both stdout and progress_log.txt, so a round's
    start/end and each client's response survive even if the tmux pane
    scrollback is lost or the session dies."""
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def _record_client_metrics(server_round, client, eval_res) -> None:
    """Append one row per Pi's own local-validation result for this round, so
    all 6 clients' final numbers end up in one file on the server instead of
    only printed on each Pi's own terminal."""
    row = {
        "round": server_round,
        "client_name": eval_res.metrics.get("client_name", client.cid),
        "num_examples": eval_res.num_examples,
        "loss": eval_res.loss,
    }
    row.update({k: v for k, v in eval_res.metrics.items() if k != "client_name"})
    pd.DataFrame([row]).to_csv(
        CLIENT_METRICS_CSV,
        mode="a",
        header=not os.path.exists(CLIENT_METRICS_CSV),
        index=False,
    )


def _stop_if_any_failure(phase: str, server_round: int, failures) -> None:
    """A client that doesn't answer within ROUND_TIMEOUT_SEC shows up here as
    a failure. All 6 Pis are required (min_*_clients=6), so rather than limp
    on with a partial/undefined aggregation, log which client dropped and
    kill the whole experiment immediately so it doesn't hang unattended."""
    if not failures:
        return
    for failure in failures:
        if isinstance(failure, BaseException):
            log_progress(f"[ROUND {server_round}] {phase} FAILURE: {failure!r}")
        else:
            client, _ = failure
            log_progress(f"[ROUND {server_round}] {phase} FAILURE, cid={client.cid}")
    log_progress(
        f"[ROUND {server_round}] {len(failures)} client(s) did not respond within "
        f"{ROUND_TIMEOUT_SEC}s during {phase} -- stopping the experiment now."
    )
    os._exit(1)


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with round start/end and per-client response logging."""

    def configure_fit(self, server_round, parameters, client_manager):
        log_progress(f"[ROUND {server_round}] fit round starting")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        for client, fit_res in results:
            name = fit_res.metrics.get("client_name", client.cid)
            fit_seconds = fit_res.metrics.get("fit_seconds")
            fit_time = "" if fit_seconds is None else f", fit_time={format_duration(fit_seconds)}"
            log_progress(
                f"[ROUND {server_round}] fit OK from {name} "
                f"(num_examples={fit_res.num_examples}{fit_time})"
            )
        _stop_if_any_failure("fit", server_round, failures)
        log_progress(f"[ROUND {server_round}] fit round done: {len(results)} ok")
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round, parameters, client_manager):
        log_progress(f"[ROUND {server_round}] evaluate round starting")
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        for client, eval_res in results:
            name = eval_res.metrics.get("client_name", client.cid)
            log_progress(
                f"[ROUND {server_round}] evaluate OK from {name} "
                f"(num_examples={eval_res.num_examples})"
            )
            _record_client_metrics(server_round, client, eval_res)
        _stop_if_any_failure("evaluate", server_round, failures)
        log_progress(f"[ROUND {server_round}] evaluate round done: {len(results)} ok")
        return super().aggregate_evaluate(server_round, results, failures)


# One module per Raspberry Pi client; the server pools each dataset's
# held-out test split into one combined evaluation set every round. The
# names match each client's CLIENT_NAME.
DATASETS = [
    ("Pi1-carhacking", carhacking_utils_cav),  # HCRL Car-Hacking Dataset
    ("Pi2-otids", otids_utils_cav),            # HCRL CAN-intrusion-dataset (OTIDS)
    ("Pi3-survival", survival_utils_cav),      # HCRL Survival Analysis Dataset
    ("Pi4-cantt", cantt_utils_cav),            # can-train-and-test, set_04
    ("Pi5-road", road_utils_cav),              # ROAD Dataset (ORNL)
    ("Pi6-canmirgu", canmirgu_utils_cav),      # CAN-MIRGU
]
DATASET_MODULES = [module for _, module in DATASETS]


def _weighted_prf(cm):
    """Weighted precision/recall/F1 from the 2x2 confusion matrix, with the
    same definitions as sklearn's average="weighted", zero_division=0."""
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    true = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred, out=np.zeros(2), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros(2), where=true > 0)
    f1 = np.divide(2 * tp, true + pred, out=np.zeros(2), where=(true + pred) > 0)
    return (
        np.average(precision, weights=true),
        np.average(recall, weights=true),
        np.average(f1, weights=true),
    )


def _mcc(cm):
    """sklearn's matthews_corrcoef formula, applied to the confusion matrix."""
    t_sum = cm.sum(axis=1, dtype=np.float64)
    p_sum = cm.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(cm, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(cov_ypyp * cov_ytyt))


def _roc_auc(y_true, scores):
    """ROC AUC as the Mann-Whitney statistic (a tied attack/normal pair counts
    1/2). Equals sklearn's roc_auc_score without its full-size copies. NaN
    when only one class is present, where sklearn raises instead."""
    pos = scores[y_true]
    neg = scores[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    neg.sort()
    below = int(np.searchsorted(neg, pos, side="left").sum())
    ties = int(np.searchsorted(neg, pos, side="right").sum()) - below
    return (2 * below + ties) / (2 * len(pos) * len(neg))


def _binary_metrics(y_true, y_pred, p_attack):
    """Metrics of one set of rows. Binary convention is fixed by
    common_cav.LABEL_TO_INT: 0 = Normal, 1 = Attack."""
    n = len(y_true)
    tp = np.count_nonzero(y_true & y_pred)
    fn = np.count_nonzero(y_true) - tp
    fp = np.count_nonzero(y_pred) - tp
    tn = n - tp - fn - fp
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    precision, recall, f1 = _weighted_prf(cm)
    metrics = {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc": _roc_auc(y_true, p_attack),
        "mcc": _mcc(cm),
    }
    return metrics, cm


def _record_dataset_metrics(server_round, dataset_names, dataset_rows, y_true, y_pred, p_attack, loss_sums):
    """Global model's metrics on each dataset's own test rows, plus their
    unweighted mean (macro_avg). The pooled numbers in round_metrics.csv are
    dominated by the largest test sets (ROAD, Car-Hacking); these show each
    dataset on its own."""
    rows = []
    start = 0
    for name, count, loss_sum in zip(dataset_names, dataset_rows, loss_sums):
        end = start + count
        metrics, cm = _binary_metrics(y_true[start:end], y_pred[start:end], p_attack[start:end])
        (tn, fp), (fn, tp) = cm.tolist()
        rows.append({
            "round": server_round,
            "dataset": name,
            "num_examples": count,
            "attack_examples": fn + tp,
            "loss": loss_sum / count,
            **metrics,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
        start = end

    table = pd.DataFrame(rows)
    count_columns = ["num_examples", "attack_examples", "tn", "fp", "fn", "tp"]
    # Rates are the unweighted mean over datasets; counts are the totals.
    macro = {"round": server_round, "dataset": "macro_avg"}
    macro.update(table[count_columns].sum().to_dict())
    macro.update(table[["loss"] + list(metrics)].mean().to_dict())
    table = pd.concat([table, pd.DataFrame([macro])], ignore_index=True)

    print(f"Round {server_round} per-dataset metrics:")
    print(table[["dataset", "num_examples", "accuracy", "f1_score", "detection_rate", "fpr", "roc_auc"]].to_string(index=False))
    table.to_csv(
        DATASET_METRICS_CSV,
        mode="a",
        header=not os.path.exists(DATASET_METRICS_CSV),
        index=False,
    )


def _predict_attack(model, x):
    """P(attack) for every row, in chunks (see EVAL_CHUNK_ROWS)."""
    p = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), EVAL_CHUNK_ROWS):
        end = min(start + EVAL_CHUNK_ROWS, len(x))
        p[start:end] = model.predict(common_cav.reshape_for_cnn(x[start:end]), verbose=0)[:, 1]
    return p


def evaluate_generalization(model, server_round):
    """Final model on each generalization set, one set in memory at a time.

    Every set is scored at the default threshold 0.5 (the same decision as
    the main evaluation). A new vehicle's set is also scored at thresholds
    taken from that vehicle's attack-free calibration recording, which needs
    no labels. Per-capture rows go to GENERALIZATION_FILES_CSV, so each attack
    type can be reported on its own.
    """
    log_progress(f"generalization evaluation of round {server_round} model started")
    t_start = time.time()
    set_rows, file_rows = [], []
    # Row weights (capped test normals) are not applied to these metrics yet;
    # detection rate, FPR and AUC do not need them.
    for name, x, y, _weights, file_index, file_names, x_cal in cantt_utils_cav.iter_generalization_sets(TEST_SIZE, RANDOM_STATE):
        y_true = y.astype(bool)
        p = _predict_attack(model, x)
        rules = [("default", 0.5, None)]
        if x_cal is not None:
            p_cal = _predict_attack(model, x_cal)
            for target in CALIBRATED_FPR_TARGETS:
                threshold = float(np.quantile(p_cal, 1 - target))
                rules.append((f"calibrated_fpr_{target:g}", threshold, float((p_cal > threshold).mean())))
        for rule, threshold, cal_fpr in rules:
            y_pred = p > threshold
            metrics, cm = _binary_metrics(y_true, y_pred, p)
            (tn, fp), (fn, tp) = cm.tolist()
            set_rows.append({"round": server_round, "set": name, "threshold_rule": rule, "threshold": threshold,
                             "calibration_fpr": cal_fpr, "num_examples": len(y), "attack_examples": fn + tp,
                             **metrics, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
            for i, file_name in enumerate(file_names):
                m = file_index == i
                attacks = int(y_true[m].sum())
                normals = int(m.sum()) - attacks
                file_rows.append({"round": server_round, "set": name, "file": file_name, "threshold_rule": rule,
                                  "num_examples": int(m.sum()), "attack_examples": attacks,
                                  "detection_rate": float(y_pred[m & y_true].mean()) if attacks else float("nan"),
                                  "fpr": float(y_pred[m & ~y_true].mean()) if normals else float("nan")})
        log_progress(f"generalization set {name}: {len(y)} rows done")
        del x, y, p, file_index, x_cal

    table = pd.DataFrame(set_rows)
    print(table[["set", "threshold_rule", "threshold", "detection_rate", "fpr", "f1_score", "roc_auc"]].to_string(index=False))
    table.to_csv(GENERALIZATION_CSV, mode="a", header=not os.path.exists(GENERALIZATION_CSV), index=False)
    files = pd.DataFrame(file_rows)
    files.to_csv(GENERALIZATION_FILES_CSV, mode="a", header=not os.path.exists(GENERALIZATION_FILES_CSV), index=False)
    log_progress(f"generalization evaluation done in {format_duration(time.time() - t_start)}")


def get_evaluate_fn(model, x_test, y_test, dataset_names, dataset_rows, run_generalization):
    n = len(y_test)
    y_true = y_test.astype(bool)
    # Which dataset each test row came from (the test sets are concatenated
    # in DATASETS order).
    dataset_id = np.repeat(np.arange(len(dataset_rows), dtype=np.int8), dataset_rows)

    def evaluate(server_round, parameters, config):
        t_start = time.time()
        model.set_weights(parameters)

        # Chunked: one evaluate()/predict() call on the whole ~17.5M-row test
        # set makes TensorFlow copy all of it into a single tensor, which ran
        # the Jetson out of memory.
        loss_sum = 0.0
        dataset_loss_sums = np.zeros(len(dataset_rows))
        p_attack = np.empty(n, dtype=np.float32)
        y_pred = np.empty(n, dtype=bool)
        for start in range(0, n, EVAL_CHUNK_ROWS):
            end = min(start + EVAL_CHUNK_ROWS, n)
            x_chunk = x_test[start:end]
            y_chunk = y_test[start:end]
            chunk_loss, _ = model.evaluate(x_chunk, y_chunk.astype(np.int64), verbose=0)
            loss_sum += chunk_loss * (end - start)
            proba = model.predict(x_chunk, verbose=0)
            p_attack[start:end] = proba[:, 1]
            y_pred[start:end] = proba[:, 1] > proba[:, 0]  # same as argmax(axis=1)
            # Per-row cross-entropy, to split the loss by dataset.
            p_true = np.where(y_chunk == 1, proba[:, 1], proba[:, 0])
            row_loss = -np.log(np.clip(p_true, KERAS_EPSILON, 1 - KERAS_EPSILON))
            dataset_loss_sums += np.bincount(
                dataset_id[start:end], weights=row_loss, minlength=len(dataset_rows),
            )
        loss = loss_sum / n

        global_metrics, cm = _binary_metrics(y_true, y_pred, p_attack)
        _record_dataset_metrics(
            server_round, dataset_names, dataset_rows, y_true, y_pred, p_attack, dataset_loss_sums,
        )

        elapsed = time.time() - t_start

        cm_df = pd.DataFrame(cm, index=["true_normal", "true_attack"], columns=["pred_normal", "pred_attack"])
        cm_xlsx = os.path.join(BASE_DIR, f"confusion_matrix_round_{server_round}.xlsx")
        cm_df.to_excel(cm_xlsx, index=True)
        print(f"Round {server_round} confusion matrix:")
        print(cm_df.to_string())

        # Same column order as before, so rows still line up with an
        # existing round_metrics.csv.
        metrics_row = pd.DataFrame([{"round": server_round, "loss": loss, **global_metrics, "elapsed_sec": elapsed}])
        metrics_row.to_csv(
            ROUND_METRICS_CSV,
            mode="a",
            header=not os.path.exists(ROUND_METRICS_CSV),
            index=False,
        )

        if run_generalization and server_round == NUM_ROUNDS:
            evaluate_generalization(model, server_round)

        return loss, {**global_metrics, "elapsed_sec": elapsed}

    return evaluate


def _cache_matches(dataset_module):
    pattern = os.path.join(
        dataset_module.BASE_DIR, "split_cache",
        f"{dataset_module.DATASET_KEY}_ts{TEST_SIZE}_rs{RANDOM_STATE}_src*.npz",
    )
    return sorted(glob.glob(pattern))


def report_cache_status(dataset_module):
    if _cache_matches(dataset_module):
        print(f"[CACHE] {dataset_module.DATASET_KEY}: found split_cache, no raw data download needed", flush=True)
    else:
        print(f"[CACHE] {dataset_module.DATASET_KEY}: NO split_cache found, raw data will be downloaded/processed", flush=True)


def load_test_split(dataset_module):
    """Only x_test/y_test from the cache: an .npz reads members on access, so
    the much larger x_train is never loaded on the server."""
    matches = _cache_matches(dataset_module)
    if not matches:
        return dataset_module.get_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    with np.load(matches[0]) as data:
        return data["x_test"], data["y_test"]


x_test_parts = []
y_test_parts = []
dataset_names = []
dataset_rows = []
for dataset_name, dataset_module in DATASETS:
    report_cache_status(dataset_module)
    x_part, y_part = load_test_split(dataset_module)
    x_test_parts.append(x_part)
    y_test_parts.append(y_part)
    dataset_names.append(dataset_name)
    dataset_rows.append(len(y_part))
    print(f"[DATA] {dataset_name}: {len(y_part)} test rows", flush=True)

x_test = np.concatenate(x_test_parts)
y_test = np.concatenate(y_test_parts).astype(np.int8)
# Drop the per-dataset copies; otherwise the test set sits in memory twice.
del x_test_parts, y_test_parts, x_part, y_part
x_test = common_cav.reshape_for_cnn(x_test)

def round_config(server_round):
    """Sent to every Pi with each fit/evaluate request, so its progress lines
    can say which round they belong to."""
    return {"server_round": server_round, "num_rounds": NUM_ROUNDS}


# Checked now rather than after the last round: building the generalization
# cache needs ~1.3 GB of raw data that the Jetson does not have.
run_generalization = common_cav.cached_split_path(
    cantt_utils_cav.GEN_DATASET_KEY, cantt_utils_cav.BASE_DIR, TEST_SIZE, RANDOM_STATE) is not None
if run_generalization:
    print(f"[CACHE] {cantt_utils_cav.GEN_DATASET_KEY}: found, generalization sets will be evaluated after round {NUM_ROUNDS}", flush=True)
else:
    print(f"[CACHE] {cantt_utils_cav.GEN_DATASET_KEY}: NOT found, generalization evaluation will be SKIPPED "
          f"(copy Pi4/split_cache/{cantt_utils_cav.GEN_DATASET_KEY}_*.npz to enable it)", flush=True)

model = build_model((x_test.shape[1], x_test.shape[2]))
strategy = LoggingFedAvg(
    evaluate_fn=get_evaluate_fn(model, x_test, y_test, dataset_names, dataset_rows, run_generalization),
    on_fit_config_fn=round_config,
    on_evaluate_config_fn=round_config,
    min_fit_clients=len(DATASET_MODULES),
    min_evaluate_clients=len(DATASET_MODULES),
    min_available_clients=len(DATASET_MODULES),
)

fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT_SEC),
    strategy=strategy,
)