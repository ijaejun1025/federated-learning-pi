#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flwr as fl
import utils_cav_ml
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
import os
from flwr.common import Config, NDArrays, Scalar


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND_METRICS_CSV = os.path.join(BASE_DIR, "round_metrics.csv")
PREDICTION_RESULTS_CSV = os.path.join(BASE_DIR, "prediction_results.csv")

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils_cav_ml.load_cav()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils_cav_ml.set_model_params(model, parameters)
        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        loss = log_loss(y_test, y_prob, labels=[0,1])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        res = pd.DataFrame(y_prob)
        res.index = pd.DataFrame(X_test).index # it's important for comparison
        res.columns = ["prob_class_0", "prob_class_1"]
        res["y_true"] = y_test
        res["y_pred"] = y_pred
        res.to_csv(PREDICTION_RESULTS_CSV, index=False)

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


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    utils_cav_ml.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=5,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:5040",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )

