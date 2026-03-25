#!/usr/bin/env python
# coding: utf-8

import flwr as fl
import numpy as np
import os
import pandas as pd
import utils_cav
import tensorflow as tf
from flwr.common import parameters_to_ndarrays
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
TEST_SIZE = 0.33
RANDOM_STATE = 41


def build_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(96, (4, 4), input_shape=input_shape, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(32, (2, 2), activation="relu", padding="same"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    return model


def get_evaluate_fn(model, x_test, y_test):
    labels = np.unique(y_test)

    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=0)

        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
        cm_csv = os.path.join(BASE_DIR, f"confusion_matrix_round_{server_round}.csv")
        cm_df.to_csv(cm_csv, index=True)
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
        }

    return evaluate


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, evaluate_fn):
        super().__init__(evaluate_fn=evaluate_fn)
        self.weights_dir = "weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        self.best_loss = float("inf")

    def evaluate(self, server_round, parameters):
        evaluate_result = super().evaluate(server_round, parameters)
        if evaluate_result is None:
            return None

        loss, metrics = evaluate_result
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"Saving best aggregated_weights at round {server_round} (loss={loss:.6f})...")
            save_path = os.path.join(self.weights_dir, "best-weights.npz")
            np.savez(save_path, *parameters_to_ndarrays(parameters))
            print(f"Saved to: {save_path}")

        return loss, metrics

_, x_test, _, y_test = utils_cav.get_global_train_test_split(
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
x_test = utils_cav.reshape_for_cnn(x_test)

model = build_model((x_test.shape[1], x_test.shape[2], x_test.shape[3]))
strategy = SaveModelStrategy(evaluate_fn=get_evaluate_fn(model, x_test, y_test))

fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)