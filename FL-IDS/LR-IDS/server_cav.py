#!/usr/bin/env python
# coding: utf-8

import flwr as fl
import numpy as np
import os
import pandas as pd
import utils_cav
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
TEST_SIZE = 0.33


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
    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=0)

        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)

        prediction_results = pd.DataFrame(y_pred_proba, columns=["prob_class_0", "prob_class_1"])
        prediction_results["y_true"] = y_test
        prediction_results["y_pred"] = y_pred
        prediction_results_csv = os.path.join(BASE_DIR, f"prediction_results_round_{server_round}.csv")
        prediction_results.to_csv(prediction_results_csv, index=False)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

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
        round_metrics_csv = os.path.join(BASE_DIR, f"round_metrics_round_{server_round}.csv")
        metrics_row.to_csv(
            round_metrics_csv,
            mode="w",
            header=True,
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
    
    def aggregate_fit(self,rnd,results,failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            save_path = os.path.join(self.weights_dir, f"round-{rnd}-weights.npz")
            np.savez(save_path, *aggregated_weights)
            print(f"Saved to: {save_path}")
        return aggregated_weights

# Load data for server-side evaluation
x, y = utils_cav.load_cav()
_, x_test, _, y_test = train_test_split(
    x,
    y,
    test_size=TEST_SIZE,
    random_state=41,
    shuffle=True,
    stratify=y,
)
if x_test.ndim == 2:
    x_test = x_test[:, :, np.newaxis, np.newaxis]

model = build_model((x_test.shape[1], x_test.shape[2], x_test.shape[3]))

# Create strategy and run server
strategy = SaveModelStrategy(evaluate_fn=get_evaluate_fn(model, x_test, y_test))

# Start Flower server for three rounds of federated learning with 1Gb of data
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)