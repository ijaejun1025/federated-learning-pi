#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import flwr as fl
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

import road_utils_cav as dataset
from model_cav import BatchSequence, ProgressLog, build_model, format_duration, progress_callbacks

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

# Compute per-class weights to counter class imbalance
_counts = np.bincount(y_train)
class_weight = {i: len(y_train) / (len(_counts) * c) for i, c in enumerate(_counts)}

model = build_model((x_train.shape[1], x_train.shape[2]))
model.summary()

train_batches = BatchSequence(x_train, y_train, shuffle=True, class_weight=class_weight)
val_batches = BatchSequence(x_val, y_val)
progress = ProgressLog(CLIENT_NAME, PROGRESS_LOG)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        progress.write(config, f"fit started ({len(x_train)} rows)")
        fit_start = time.time()
        model.set_weights(parameters)
        history = model.fit(
            train_batches,
            epochs=3,
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

        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)

        map_score = average_precision_score(y_val, y_pred_proba[:, 1])

        print(
            f"Client {args.client_id} eval accuracy: {accuracy:.4f}, "
            f"F1: {f1:.4f}, MAP: {map_score:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        progress.write(config, f"round finished (local val accuracy={accuracy:.4f}, F1={f1:.4f})")

        return loss, len(x_val), {
            "client_name": CLIENT_NAME,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "map": float(map_score),
            "precision": float(precision),
            "recall": float(recall),
        }


fl.client.start_numpy_client(
    server_address=args.server_address,
    client=FlowerClient(),
)
