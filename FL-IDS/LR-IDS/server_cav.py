#!/usr/bin/env python
# coding: utf-8

import time
import flwr as fl
import numpy as np
import os
import pandas as pd
import utils_cav
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
TEST_SIZE = 0.33
RANDOM_STATE = 41


def build_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(96, 4, activation="relu", padding="same"),
            tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv1D(32, 2, activation="relu", padding="same"),
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
        t_start = time.time()
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=0)

        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = y_pred_proba.argmax(axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        detection_rate = recall  # TP / (TP + FN), identical to recall in binary IDS context
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # FPR: FP / (FP + TN) — extracted from confusion matrix (binary: row 0 = Normal)
        tn, fp = cm[0, 0], cm[0, 1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # FNR: FN / (FN + TP) — extracted from confusion matrix (binary: row 1 = Attack)
        fn, tp = cm[1, 0], cm[1, 1]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        elapsed = time.time() - t_start

        cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
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


_, x_test, _, y_test = utils_cav.get_global_train_test_split(
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
x_test = utils_cav.reshape_for_cnn(x_test)

model = build_model((x_test.shape[1], x_test.shape[2]))
strategy = fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn(model, x_test, y_test))

fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)