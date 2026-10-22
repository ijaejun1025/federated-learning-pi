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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
)

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
import carhacking_utils_cav
import ciciov_utils_cav
import common_cav
import otids_utils_cav
import road_utils_cav
import survival_utils_cav
from model_cav import build_model

ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
CLIENT_METRICS_CSV = os.path.join(BASE_DIR, "client_metrics.csv")
PROGRESS_LOG = os.path.join(BASE_DIR, "progress_log.txt")
TEST_SIZE = 0.33
RANDOM_STATE = 41
# Longest observed client round (Pi2, ~6.4h fit) plus margin, so a dropped
# client ends its round instead of hanging the server indefinitely.
ROUND_TIMEOUT_SEC = 10 * 3600


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
            log_progress(
                f"[ROUND {server_round}] fit OK from {name} "
                f"(num_examples={fit_res.num_examples})"
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
# held-out test split into one combined evaluation set every round.
DATASET_MODULES = [
    carhacking_utils_cav,  # Pi1: HCRL Car-Hacking Dataset
    otids_utils_cav,  # Pi2: HCRL CAN-intrusion-dataset (OTIDS)
    survival_utils_cav,  # Pi3: HCRL Survival Analysis Dataset
    ciciov_utils_cav,    # Pi4: CIC IoV Dataset 2024
    road_utils_cav,      # Pi5: ROAD Dataset (ORNL)
    canmirgu_utils_cav,  # Pi6: CAN-MIRGU
]


def get_evaluate_fn(model, x_test, y_test):
    labels = [0, 1]

    def evaluate(server_round, parameters, config):
        t_start = time.time()
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=0)

        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        detection_rate = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Binary convention is fixed by common_cav.LABEL_TO_INT: 0 = Normal, 1 = Attack.
        tn, fp = cm[0, 0], cm[0, 1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fn, tp = cm[1, 0], cm[1, 1]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        elapsed = time.time() - t_start

        cm_df = pd.DataFrame(cm, index=["true_normal", "true_attack"], columns=["pred_normal", "pred_attack"])
        cm_xlsx = os.path.join(BASE_DIR, f"confusion_matrix_round_{server_round}.xlsx")
        cm_df.to_excel(cm_xlsx, index=True)
        print(f"Round {server_round} confusion matrix:")
        print(cm_df.to_string())

        metrics_row = pd.DataFrame(
            [
                {
                    "round": server_round,
                    "loss": loss,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "detection_rate": detection_rate,
                    "fpr": fpr,
                    "fnr": fnr,
                    "roc_auc": roc_auc,
                    "mcc": mcc,
                    "elapsed_sec": elapsed,
                }
            ]
        )
        metrics_row.to_csv(
            ROUND_METRICS_CSV,
            mode="a",
            header=not os.path.exists(ROUND_METRICS_CSV),
            index=False,
        )

        return loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detection_rate": detection_rate,
            "fpr": fpr,
            "fnr": fnr,
            "roc_auc": roc_auc,
            "mcc": mcc,
            "elapsed_sec": elapsed,
        }

    return evaluate


def report_cache_status(dataset_module):
    pattern = os.path.join(
        dataset_module.BASE_DIR, "split_cache",
        f"{dataset_module.DATASET_KEY}_ts{TEST_SIZE}_rs{RANDOM_STATE}_src*.npz",
    )
    if glob.glob(pattern):
        print(f"[CACHE] {dataset_module.DATASET_KEY}: found split_cache, no raw data download needed", flush=True)
    else:
        print(f"[CACHE] {dataset_module.DATASET_KEY}: NO split_cache found, raw data will be downloaded/processed", flush=True)


x_test_parts = []
y_test_parts = []
for dataset_module in DATASET_MODULES:
    report_cache_status(dataset_module)
    x_part, y_part = dataset_module.get_test_split(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    x_test_parts.append(x_part)
    y_test_parts.append(y_part)

x_test = np.concatenate(x_test_parts)
y_test = np.concatenate(y_test_parts)
x_test = common_cav.reshape_for_cnn(x_test)

model = build_model((x_test.shape[1], x_test.shape[2]))
strategy = LoggingFedAvg(
    evaluate_fn=get_evaluate_fn(model, x_test, y_test),
    min_fit_clients=len(DATASET_MODULES),
    min_evaluate_clients=len(DATASET_MODULES),
    min_available_clients=len(DATASET_MODULES),
)

fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=3, round_timeout=ROUND_TIMEOUT_SEC),
    strategy=strategy,
)