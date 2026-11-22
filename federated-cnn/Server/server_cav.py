#!/usr/bin/env python
# coding: utf-8

import datetime
import time
import flwr as fl
import numpy as np
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Each dataset's utils module lives inside its own PiN/ folder (self-contained
# so that folder alone can be copied onto that Raspberry Pi), a sibling of
# this Server/ folder. The server runs centrally on the Jetson and needs all
# 6, so append every PiN/ folder to the import path. Appending (not inserting
# at the front) keeps this directory's own common_cav.py/model_cav.py as the
# ones actually used below, even though every PiN/ folder also carries its
# own copy for standalone Pi use.
for _pi_folder in ("Pi1", "Pi2", "Pi3", "Pi4", "Pi5", "Pi6"):
    sys.path.append(os.path.join(PROJECT_DIR, _pi_folder))

import canmirgu_utils_cav
import cantt_utils_cav
import carhacking_utils_cav
import common_cav
import otids_utils_cav
import road_utils_cav
import survival_utils_cav
from model_cav import THRESHOLD_TARGET_FPR, build_model, format_duration, learning_rate

# One local epoch per round (model_cav.LOCAL_EPOCHS), so more, shorter rounds:
# in the PC simulation the macro recall flattened out after rounds 6-9.
NUM_ROUNDS = 15
# Every round (threshold 0.5, progress monitoring): one row per dataset in
# DATASET_METRICS_CSV, and the macro average / weighted pool in
# ROUND_METRICS_CSV.
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
DATASET_METRICS_CSV = os.path.join(BASE_DIR, "dataset_metrics.csv")
# After the last round: the four final results (see evaluate_final), and the
# per-capture breakdown of the generalization sets (one capture = one attack
# type).
FINAL_RESULTS_CSV = os.path.join(BASE_DIR, "final_results.csv")
FINAL_DATASETS_CSV = os.path.join(BASE_DIR, "final_dataset_metrics.csv")
GENERALIZATION_FILES_CSV = os.path.join(BASE_DIR, "generalization_files.csv")
# Result labels of Pi4's generalization sets (cantt_utils_cav.GEN_FILES; one
# cache each, cantt_utils_cav.GEN_DATASET_KEYS).
GENERALIZATION_RESULTS = {
    "forester_unknown_attack": "2_trained_vehicle_new_attacks",
    "traverse_known_attack": "3_new_vehicle_known_attacks",
    "traverse_unknown_attack": "4_new_vehicle_new_attacks",
}
# Round 0 evaluates the untrained initial model; only a quick check that the
# evaluation code runs is done then, on this many rows per dataset.
SANITY_ROWS_PER_DATASET = 50_000
# The global model after every round (round_01.weights.h5, ...; ~1.2 MB each),
# so other metrics or thresholds can be computed later without rerunning FL:
# build_model(...).load_weights(path).
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
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
    all 6 clients' numbers end up in one file on the server instead of only
    printed on each Pi's own terminal. These ``local_val_*`` numbers come from
    the clients' training-like data (3-20% attacks) and are not comparable
    with the server's test metrics."""
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
    """FedAvg with round start/end and per-client response logging; after
    the last round's federated evaluation it runs the final evaluation with
    the decision thresholds the clients proposed."""

    def __init__(self, evaluator=None, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator

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
        thresholds = {}
        for client, eval_res in results:
            name = eval_res.metrics.get("client_name", client.cid)
            log_progress(
                f"[ROUND {server_round}] evaluate OK from {name} "
                f"(num_examples={eval_res.num_examples})"
            )
            _record_client_metrics(server_round, client, eval_res)
            if "threshold_at_target_fpr" in eval_res.metrics:
                thresholds[name] = float(eval_res.metrics["threshold_at_target_fpr"])
        _stop_if_any_failure("evaluate", server_round, failures)
        log_progress(f"[ROUND {server_round}] evaluate round done: {len(results)} ok")
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        # Flower evaluates centrally (evaluate_fn) before the clients, so the
        # thresholds the clients propose for this round's model arrive here.
        if self.evaluator is not None and server_round == NUM_ROUNDS:
            self.evaluator.evaluate_final(server_round, thresholds)
        return aggregated


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


# ------------------------------------------------------------------ metrics
# Test sets keep every attack row but only a uniform sample of normal rows
# (common_cav.Sampling); each row's weight says how many rows it stands for.
# All counts and rates below are weighted, so they estimate the full test set.
# Binary convention (common_cav.LABEL_TO_INT): 0 = Normal, 1 = Attack, and
# precision / F1 are those of the attack class.

def _confusion(y_true, y_pred, w):
    """[[tn, fp], [fn, tp]] as weighted counts."""
    return np.array([
        [w[~y_true & ~y_pred].sum(), w[~y_true & y_pred].sum()],
        [w[y_true & ~y_pred].sum(), w[y_true & y_pred].sum()],
    ], dtype=np.float64)


def _f1_weighted_avg(cm):
    """F1 averaged over both classes, weighted by class size (sklearn's
    average="weighted"). Kept for comparison with earlier runs only: with
    >99% normal rows it stays near 1 even when attacks are missed."""
    tp = np.diag(cm)
    pred, true = cm.sum(axis=0), cm.sum(axis=1)
    f1 = np.divide(2 * tp, true + pred, out=np.zeros(2), where=(true + pred) > 0)
    return float(np.average(f1, weights=true)) if true.sum() else 0.0


def _mcc(cm):
    """sklearn's matthews_corrcoef formula, on the (weighted) confusion matrix."""
    t_sum, p_sum = cm.sum(axis=1), cm.sum(axis=0)
    n_correct, n_samples = np.trace(cm), p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(cov_ypyp * cov_ytyt))


def _roc_auc(y_true, scores, w):
    """Weighted ROC AUC (Mann-Whitney: P(attack score > normal score), a tie
    counts 1/2). NaN when one class is missing."""
    pos, neg = scores[y_true], scores[~y_true]
    w_pos, w_neg = w[y_true], w[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(neg, kind="mergesort")
    neg, w_neg = neg[order], w_neg[order]
    cum = np.r_[0.0, np.cumsum(w_neg, dtype=np.float64)]
    below = cum[np.searchsorted(neg, pos, side="left")]
    upto = cum[np.searchsorted(neg, pos, side="right")]
    return float(np.sum(w_pos * (below + 0.5 * (upto - below))) / (w_pos.sum() * cum[-1]))


def _pr_auc(y_true, scores, w):
    """Weighted average precision (sklearn's definition: sum over thresholds of
    (R_n - R_{n-1}) * P_n). Depends on the attack share, hence the weights."""
    if not y_true.any():
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, yt, ww = scores[order], y_true[order], w[order]
    tp_cum = np.cumsum(ww * yt, dtype=np.float64)
    fp_cum = np.cumsum(ww * ~yt, dtype=np.float64)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last row of each distinct score
    tp, fp = tp_cum[last], fp_cum[last]
    precision = tp / (tp + fp)
    recall = tp / tp_cum[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _binary_metrics(y_true, p_attack, w, threshold):
    """Weighted metrics of one set of rows at one decision threshold."""
    y_pred = p_attack > threshold
    cm = _confusion(y_true, y_pred, w)
    (tn, fp), (fn, tp) = cm
    n = cm.sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    detection = tp / (tp + fn) if tp + fn > 0 else 0.0
    metrics = {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "detection_rate": detection,
        "f1_score": 2 * precision * detection / (precision + detection) if precision + detection > 0 else 0.0,
        "fpr": fp / (fp + tn) if fp + tn > 0 else 0.0,
        "fnr": fn / (fn + tp) if fn + tp > 0 else 0.0,
        "roc_auc": _roc_auc(y_true, p_attack, w),
        "pr_auc": _pr_auc(y_true, p_attack, w),
        "mcc": _mcc(cm),
        "f1_weighted_avg": _f1_weighted_avg(cm),
    }
    counts = {"num_examples": n, "attack_examples": tp + fn, "sampled_rows": int(len(y_true)),
              "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return metrics, counts


def _weighted_loss(y_true, p_attack, w):
    """Weighted mean cross-entropy (probabilities clipped as Keras does)."""
    p_true = np.where(y_true, p_attack, 1.0 - p_attack)
    row_loss = -np.log(np.clip(p_true, KERAS_EPSILON, 1 - KERAS_EPSILON))
    return float(np.sum(w * row_loss) / np.sum(w))


def _predict_attack(model, x):
    """P(attack) for every row, in chunks (see EVAL_CHUNK_ROWS). One predict()
    on the whole test set makes TensorFlow copy all of it into one tensor,
    which ran the Jetson out of memory."""
    p = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), EVAL_CHUNK_ROWS):
        end = min(start + EVAL_CHUNK_ROWS, len(x))
        p[start:end] = model.predict(common_cav.reshape_for_cnn(x[start:end]), verbose=0)[:, 1]
    return p


def _append_csv(path, table):
    table.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_round_weights(model, server_round):
    """WEIGHTS_DIR/round_NN.weights.h5. A failed save is logged, not raised,
    so it cannot stop a day-long run (the CSVs are still written)."""
    path = os.path.join(WEIGHTS_DIR, f"round_{server_round:02d}.weights.h5")
    try:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model.save_weights(path)
    except Exception as exc:  # noqa: BLE001
        log_progress(f"[ROUND {server_round}] could not save the global weights to {path}: {exc!r}")


class Evaluator:
    """Server-side evaluation of the global model.

    - Round 0 (untrained model): a quick run of the whole evaluation code on
      SANITY_ROWS_PER_DATASET rows per dataset, nothing written.
    - Rounds 1..NUM_ROUNDS: the six clients' test sets at threshold 0.5,
      per dataset, macro average (every dataset counts the same) and
      weighted pool (every row counts the same, so CAN-MIRGU's large normal
      share dominates).
    - After the last round's federated evaluation (evaluate_final): the four
      final results, each at threshold 0.5 and at the median of the
      thresholds the clients propose from their own validation normals; the
      new vehicle's sets also at a threshold set on its attack-free
      recording.
    """

    def __init__(self, model, x_test, y_test, w_test, dataset_names, dataset_rows, run_generalization):
        self.model = model
        self.x_test = x_test
        self.y_true = y_test.astype(bool)
        self.w = w_test.astype(np.float64)
        self.dataset_names = dataset_names
        self.bounds = np.r_[0, np.cumsum(dataset_rows)]
        self.run_generalization = run_generalization
        self.last_round = None
        self.last_p = None

    # ---- tables
    def _dataset_table(self, p_attack, threshold, rule, rows=None):
        """Per-dataset rows plus macro_avg and pooled_weighted rows."""
        records = []
        for i, name in enumerate(self.dataset_names):
            lo, hi = self.bounds[i], self.bounds[i + 1]
            if rows is not None:
                hi = min(hi, lo + rows)
            y, p, w = self.y_true[lo:hi], p_attack[lo:hi], self.w[lo:hi]
            metrics, counts = _binary_metrics(y, p, w, threshold)
            records.append({"dataset": name, "threshold_rule": rule, "threshold": threshold,
                            "loss": _weighted_loss(y, p, w), **metrics, **counts})
        table = pd.DataFrame(records)
        metric_cols = ["loss"] + [c for c in records[0] if c in metrics]
        count_cols = ["num_examples", "attack_examples", "sampled_rows", "tn", "fp", "fn", "tp"]
        macro = {"dataset": "macro_avg", "threshold_rule": rule, "threshold": threshold}
        macro.update(table[metric_cols].mean().to_dict())
        macro.update(table[count_cols].sum().to_dict())
        if rows is None:
            metrics, counts = _binary_metrics(self.y_true, p_attack, self.w, threshold)
            pooled = {"dataset": "pooled_weighted", "threshold_rule": rule, "threshold": threshold,
                      "loss": _weighted_loss(self.y_true, p_attack, self.w), **metrics, **counts}
            return pd.concat([table, pd.DataFrame([macro, pooled])], ignore_index=True)
        return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)

    # ---- per round (Flower's evaluate_fn)
    def evaluate_round(self, server_round, parameters, config):
        self.model.set_weights(parameters)
        if server_round == 0:
            self._sanity_check()
            return None
        save_round_weights(self.model, server_round)
        t_start = time.time()
        p_attack = _predict_attack(self.model, self.x_test)
        table = self._dataset_table(p_attack, 0.5, "default_0.5")
        table.insert(0, "round", server_round)
        elapsed = time.time() - t_start
        print(f"Round {server_round} per-dataset metrics (threshold 0.5, weighted):")
        print(table[["dataset", "detection_rate", "fpr", "precision", "f1_score", "roc_auc", "pr_auc"]].to_string(index=False))
        _append_csv(DATASET_METRICS_CSV, table[~table.dataset.isin(["macro_avg", "pooled_weighted"])])
        summary = table[table.dataset.isin(["macro_avg", "pooled_weighted"])].rename(columns={"dataset": "aggregate"})
        summary["elapsed_sec"] = elapsed
        _append_csv(ROUND_METRICS_CSV, summary)
        self.last_round, self.last_p = server_round, p_attack
        pooled = table[table.dataset == "pooled_weighted"].iloc[0]
        macro = table[table.dataset == "macro_avg"].iloc[0]
        return float(pooled["loss"]), {f"macro_{k}": float(macro[k]) for k in
                                       ("detection_rate", "fpr", "precision", "f1_score", "roc_auc", "pr_auc")}

    def _sanity_check(self):
        """Run every metric on a small slice with the initial model, so a bug
        in the evaluation shows up before hours of training, not after."""
        t_start = time.time()
        idx = np.concatenate([np.arange(lo, min(hi, lo + SANITY_ROWS_PER_DATASET))
                              for lo, hi in zip(self.bounds[:-1], self.bounds[1:])])
        p_small = np.zeros(len(self.y_true), dtype=np.float32)
        p_small[idx] = _predict_attack(self.model, self.x_test[idx])
        self._dataset_table(p_small, 0.5, "sanity", rows=SANITY_ROWS_PER_DATASET)
        if self.run_generalization:
            for name, path in cantt_utils_cav.generalization_cache_paths(TEST_SIZE, RANDOM_STATE).items():
                needed = ["x_test", "y_test", "w_test", "test_file", "test_file_names"]
                if cantt_utils_cav.GEN_SET_VEHICLE[name] == "traverse":
                    needed.append("x_calibration")
                with np.load(path) as data:
                    missing = [k for k in needed if k not in data.files]
                if missing:
                    raise RuntimeError(f"generalization cache {path} lacks {missing}")
        log_progress(f"round 0: evaluation code checked on {len(idx)} rows in {format_duration(time.time() - t_start)} "
                     f"(full evaluation starts after round 1)")

    # ---- after the last round (called from aggregate_evaluate)
    def evaluate_final(self, server_round, client_thresholds):
        log_progress(f"final evaluation of round {server_round} model started")
        t_start = time.time()
        if self.last_round != server_round:
            raise RuntimeError(f"no round {server_round} predictions (last evaluated round: {self.last_round})")
        rules = [("default_0.5", 0.5)]
        if client_thresholds:
            median = float(np.median(list(client_thresholds.values())))
            rules.append((f"client_median_fpr_{THRESHOLD_TARGET_FPR:g}", median))
            log_progress("client thresholds: " + ", ".join(f"{k}={v:.5f}" for k, v in client_thresholds.items())
                         + f" -> median {median:.5f}")
        else:
            log_progress("no client thresholds reported; final results at threshold 0.5 only")

        final_rows, dataset_tables = [], []
        for rule, threshold in rules:
            table = self._dataset_table(self.last_p, threshold, rule)
            table.insert(0, "round", server_round)
            dataset_tables.append(table)
            for aggregate in ("macro_avg", "pooled_weighted"):
                row = table[table.dataset == aggregate].iloc[0].to_dict()
                row.pop("dataset")
                final_rows.append({"result": "1_all_client_test_sets", "set": "six client test sets",
                                   "aggregate": aggregate, **row})
        _append_csv(FINAL_DATASETS_CSV, pd.concat(dataset_tables, ignore_index=True))

        if self.run_generalization:
            final_rows += self._evaluate_generalization(server_round, rules)
        final = pd.DataFrame(final_rows)
        if "round" not in final:
            final.insert(0, "round", server_round)
        _append_csv(FINAL_RESULTS_CSV, final)
        print("Final results (weighted):")
        print(final[["result", "aggregate", "threshold_rule", "threshold", "detection_rate", "fpr", "precision",
                     "f1_score", "roc_auc", "pr_auc"]].to_string(index=False))
        log_progress(f"final evaluation done in {format_duration(time.time() - t_start)}")

    def _evaluate_generalization(self, server_round, rules):
        """Pi4's generalization sets, one in memory at a time."""
        final_rows, file_rows = [], []
        for name, x, y, w, file_index, file_names, x_cal in cantt_utils_cav.iter_generalization_sets(TEST_SIZE, RANDOM_STATE):
            y_true, w = y.astype(bool), w.astype(np.float64)
            p = _predict_attack(self.model, x)
            set_rules = [(rule, threshold, None) for rule, threshold in rules]
            if x_cal is not None:
                p_cal = _predict_attack(self.model, x_cal)
                threshold = float(np.quantile(p_cal, 1 - THRESHOLD_TARGET_FPR))
                set_rules.append((f"vehicle_calibrated_fpr_{THRESHOLD_TARGET_FPR:g}", threshold, float((p_cal > threshold).mean())))
            for rule, threshold, cal_fpr in set_rules:
                metrics, counts = _binary_metrics(y_true, p, w, threshold)
                final_rows.append({"round": server_round, "result": GENERALIZATION_RESULTS.get(name, name), "set": name,
                                   "aggregate": "set", "threshold_rule": rule, "threshold": threshold,
                                   "calibration_fpr": cal_fpr, "loss": _weighted_loss(y_true, p, w), **metrics, **counts})
                for i, file_name in enumerate(file_names):
                    m = file_index == i
                    y_pred = p > threshold
                    attacks = float(w[m & y_true].sum())
                    normals = float(w[m & ~y_true].sum())
                    file_rows.append({
                        "round": server_round, "result": GENERALIZATION_RESULTS.get(name, name), "set": name,
                        "file": file_name, "threshold_rule": rule, "threshold": threshold,
                        "num_examples": attacks + normals, "attack_examples": attacks, "sampled_rows": int(m.sum()),
                        "detection_rate": float(w[m & y_true & y_pred].sum() / attacks) if attacks else float("nan"),
                        "fpr": float(w[m & ~y_true & y_pred].sum() / normals) if normals else float("nan"),
                    })
            log_progress(f"generalization set {name}: {len(y)} rows done")
            del x, y, w, p, file_index, x_cal
        _append_csv(GENERALIZATION_FILES_CSV, pd.DataFrame(file_rows))
        return final_rows


def _cache_path(dataset_module):
    return common_cav.cached_split_path(dataset_module.DATASET_KEY, dataset_module.BASE_DIR, TEST_SIZE, RANDOM_STATE)


def load_test_split(dataset_module):
    """x_test / y_test / w_test from the cache (the train rows are never
    read on the server); builds the cache from raw data if there is none."""
    if _cache_path(dataset_module) is None:
        print(f"[CACHE] {dataset_module.DATASET_KEY}: NO split_cache found, raw data will be downloaded/processed", flush=True)
        dataset_module.get_global_train_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        print(f"[CACHE] {dataset_module.DATASET_KEY}: found split_cache", flush=True)
    return common_cav.load_cached(dataset_module.DATASET_KEY, dataset_module.BASE_DIR,
                                  ("x_test", "y_test", "w_test"), TEST_SIZE, RANDOM_STATE)


def round_config(server_round):
    """Sent to every Pi with each fit/evaluate request: the round (for its
    progress lines) and this round's client learning rate (model_cav.learning_rate)."""
    return {"server_round": server_round, "num_rounds": NUM_ROUNDS,
            "learning_rate": learning_rate(server_round, NUM_ROUNDS)}


if __name__ == "__main__":
    x_parts, y_parts, w_parts, dataset_names, dataset_rows = [], [], [], [], []
    for dataset_name, dataset_module in DATASETS:
        x_part, y_part, w_part = load_test_split(dataset_module)
        x_parts.append(x_part); y_parts.append(y_part); w_parts.append(w_part)
        dataset_names.append(dataset_name)
        dataset_rows.append(len(y_part))
        print(f"[DATA] {dataset_name}: {len(y_part)} test rows ({int(y_part.sum())} attack), "
              f"standing for {w_part.sum():.0f} rows", flush=True)
    x_test = common_cav.reshape_for_cnn(np.concatenate(x_parts))
    y_test = np.concatenate(y_parts).astype(np.int8)
    w_test = np.concatenate(w_parts).astype(np.float32)
    # Drop the per-dataset copies; otherwise the test set sits in memory twice.
    del x_parts, y_parts, w_parts, x_part, y_part, w_part

    # Checked now rather than after the last round: building the
    # generalization cache needs ~1.3 GB of raw data the Jetson does not have.
    gen_paths = cantt_utils_cav.generalization_cache_paths(TEST_SIZE, RANDOM_STATE)
    for name, key in cantt_utils_cav.GEN_DATASET_KEYS.items():
        print(f"[CACHE] {key}: {'found' if gen_paths[name] else 'NOT found'}", flush=True)
    run_generalization = all(gen_paths.values())
    if run_generalization:
        print(f"[CACHE] results 2-4 will be evaluated after round {NUM_ROUNDS}", flush=True)
    else:
        missing = [f"Pi4/split_cache/{cantt_utils_cav.GEN_DATASET_KEYS[n]}.npz" for n, path in gen_paths.items() if path is None]
        print(f"[CACHE] results 2-4 will be SKIPPED (copy {', '.join(missing)} to enable them)", flush=True)

    model = build_model((x_test.shape[1], x_test.shape[2]))
    evaluator = Evaluator(model, x_test, y_test, w_test, dataset_names, dataset_rows, run_generalization)
    strategy = LoggingFedAvg(
        evaluator=evaluator,
        evaluate_fn=evaluator.evaluate_round,
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
