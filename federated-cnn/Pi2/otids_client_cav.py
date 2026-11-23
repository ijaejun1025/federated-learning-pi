#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import flwr as fl
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

import otids_utils_cav as dataset
import hardware_cav
from model_cav import (BATCH_SIZE, LEARNING_RATE_START, LOCAL_EPOCHS, THRESHOLD_TARGET_FPR, BatchSequence,
                       ProgressLog, build_model, class_weights, format_duration, progress_callbacks,
                       set_learning_rate)

TEST_SIZE = 0.33
RANDOM_STATE = 41
LOCAL_VAL_SIZE = 0.2
PROGRESS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress_log.txt")
CLIENT_NAME = "Pi2-otids"


parser = argparse.ArgumentParser()
parser.add_argument("--client-id", type=int, default=1, help="Log label only (OTIDS client = Pi2)")
parser.add_argument("--server-address", type=str, required=True, help="the Jetson server, e.g. <JETSON_IP>:3040")
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
DEVICE_INFO = hardware_cav.device_info()


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # The server lowers the learning rate every round (model_cav.learning_rate).
        rate = float(config.get("learning_rate", LEARNING_RATE_START))
        progress.write(config, f"fit started ({len(x_train)} rows, batch {BATCH_SIZE}, {LOCAL_EPOCHS} epoch(s), "
                               f"learning rate {rate:.2e})")
        fit_start = time.time()
        meter = hardware_cav.HardwareMeter()
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
        weights = model.get_weights()
        # Hardware use of this round's local training (hardware_cav; the server
        # writes it to hardware_metrics.csv).
        hardware = meter.stop(rows=len(x_train) * LOCAL_EPOCHS, received_bytes=hardware_cav.array_bytes(parameters),
                              sent_bytes=hardware_cav.array_bytes(weights))
        hardware.update(DEVICE_INFO)
        progress.write(config, f"fit done in {format_duration(fit_seconds)}; sending weights, then waiting for the other Pis")
        return weights, len(x_train), {"client_name": CLIENT_NAME, "fit_seconds": fit_seconds, **hardware}

    def evaluate(self, parameters, config):
        progress.write(config, "local evaluate started")
        meter = hardware_cav.HardwareMeter()
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(val_batches, verbose=0)

        predict_start = time.time()
        y_pred_proba = model.predict(BatchSequence(x_val), verbose=0)
        predict_seconds = time.time() - predict_start
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
        # Hardware use of this evaluation, then (not counted in it) the
        # latency of one frame at a time.
        hardware = meter.stop(rows=len(x_val), received_bytes=hardware_cav.array_bytes(parameters),
                              infer_batch_ms_per_row=1000.0 * predict_seconds / len(x_val))
        hardware.update(hardware_cav.frame_latency_ms(model, x_val))

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
            **hardware,
        }


fl.client.start_numpy_client(
    server_address=args.server_address,
    client=FlowerClient(),
)
