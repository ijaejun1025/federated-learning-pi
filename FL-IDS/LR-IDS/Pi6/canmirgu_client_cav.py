#!/usr/bin/env python
# coding: utf-8

import argparse

import flwr as fl
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

import canmirgu_utils_cav as dataset
from model_cav import build_model

TEST_SIZE = 0.33
RANDOM_STATE = 41
LOCAL_VAL_SIZE = 0.2


parser = argparse.ArgumentParser()
parser.add_argument("--client-id", type=int, default=5, help="Log label only (CAN-MIRGU client = Pi6)")
parser.add_argument("--server-address", type=str, default="192.168.137.68:3040")
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


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(
            x_train,
            y_train,
            epochs=3,
            validation_data=(x_val, y_val),
            class_weight=class_weight,
            verbose=1,
        )
        print("Fit history:", history.history)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)

        y_pred_proba = model.predict(x_val, verbose=0)
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

        return loss, len(x_val), {
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
