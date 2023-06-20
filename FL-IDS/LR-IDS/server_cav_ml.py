#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from flwr.common import Config, NDArrays, Scalar

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils_cav.load_cav()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test), labels=[0,1])
        accuracy = model.score(X_test, y_test)
        res = pd.DataFrame(model.predict_proba(X_test))
        res.index = pd.DataFrame(X_test).index # it's important for comparison
        res.columns = ["prediction", 'real']
        res.to_csv("prediction_results.csv")
        return {"Aggregated Results: loss ":loss}, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    utils.set_initial_params(model)
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

