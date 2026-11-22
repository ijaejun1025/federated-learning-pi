#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import flwr as fl
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

import road_utils_cav as dataset
from model_cav import (BATCH_SIZE, LEARNING_RATE_START, LOCAL_EPOCHS, THRESHOLD_TARGET_FPR, BatchSequence,
                       ProgressLog, build_model, class_weights, format_duration, progress_callbacks,
                       set_learning_rate)

TEST_SIZE = 0.33
RANDOM_STATE = 41
LOCAL_VAL_SIZE = 0.2
PROGRESS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress_log.txt")
CLIENT_NAME = "Pi5-road"


parser = argparse.ArgumentParser()
parser.add_argument("--client-id", type=int, default=4, help="Log label only (ROAD client = Pi5)")
parser.add_argument("--server-address", type=str, default="192.168.137.250:3040")
args = parser.parse_args()

x_train, x_val, y_train, y_val = dataset.get_client_data(
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    local_val_size=LOCAL_VAL_SIZE,
)

x_train = dataset.reshape_for_cnn(x_train)
x_val = dataset.reshape_for_cnn(x_val)

model = build_model((x_train.shape[1], x_train.shape[2]))
model.summary()

# class_weights() is None unless model_cav.USE_CLASS_WEIGHTS (sampling already caps the attack share).
train_batches = BatchSequence(x_train, y_train, shuffle=True, class_weight=class_weights(y_train))
val_batches = BatchSequence(x_val, y_val)
progress = ProgressLog(CLIENT_NAME, PROGRESS_LOG)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # The server lowers the learning rate every round (model_cav.learning_rate).
        rate = float(config.get("learning_rate", LEARNING_RATE_START))
        progress.write(config, f"fit started ({len(x_train)} rows, batch {BATCH_SIZE}, {LOCAL_EPOCHS} epoch(s), "
                               f"learning rate {rate:.2e})")
        fit_start = time.time()
        model.set_weights(parameters)
        set_learning_rate(model, rate)
        history = model.fit(
            train_batches,
            epochs=LOCAL_EPOCHS,
            validation_data=val_batches,
            verbose=1,
            callbacks=progress_callbacks(progress, config),
        )
        print("Fit history:", history.history)
        fit_seconds = time.time() - fit_start
        progress.write(config, f"fit done in {format_duration(fit_seconds)}; sending weights, then waiting for the other Pis")
        return model.get_weights(), len(x_train), {"client_name": CLIENT_NAME, "fit_seconds": fit_seconds}

    def evaluate(self, parameters, config):
        progress.write(config, "local evaluate started")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(val_batches, verbose=0)

        y_pred_proba = model.predict(BatchSequence(x_val), verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)
        p_attack = y_pred_proba[:, 1]

        # Local validation keeps the training mix (3-20% attacks), so these
        # are progress checks only, not comparable with the server's metrics.
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        map_score = average_precision_score(y_val, p_attack)
        # This client's proposed decision threshold (see THRESHOLD_TARGET_FPR).
        threshold = float(np.quantile(p_attack[y_val == 0], 1 - THRESHOLD_TARGET_FPR))

        print(
            f"Client {args.client_id} local val accuracy: {accuracy:.4f}, "
            f"attack F1: {f1:.4f}, PR-AUC: {map_score:.4f}, "
            f"attack precision: {precision:.4f}, detection rate: {recall:.4f}, "
            f"threshold at FPR {THRESHOLD_TARGET_FPR:g}: {threshold:.5f}"
        )

        progress.write(config, f"round finished (local val accuracy={accuracy:.4f}, attack F1={f1:.4f}, "
                               f"threshold={threshold:.5f})")

        return loss, len(x_val), {
            "client_name": CLIENT_NAME,
            "local_val_accuracy": float(accuracy),
            "local_val_attack_f1": float(f1),
            "local_val_pr_auc": float(map_score),
            "local_val_attack_precision": float(precision),
            "local_val_detection_rate": float(recall),
            "local_val_attack_share": float(np.mean(y_val)),
            "threshold_at_target_fpr": threshold,
        }


fl.client.start_numpy_client(
    server_address=args.server_address,
    client=FlowerClient(),
)
