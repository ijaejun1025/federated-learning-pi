#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import flwr as fl
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils_cav_ml

if __name__ == "__main__":
    # Load CAV dataset
    (X_train, y_train), (X_test, y_test) = utils_cav_ml.load_cav()

    # Split train set into 5 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    (X_train, y_train) = utils_cav_ml.partition(X_train, y_train, 5)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils_cav_ml.set_initial_params(model)

    # Define Flower client
    class CAVClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils_cav_ml.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils_cav_ml.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils_cav_ml.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils_cav_ml.set_model_params(model, parameters)
            preds = model.predict_proba(X_test)
            all_classes = {0,1}
            loss = log_loss(y_test, preds, labels=[0,1])
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="192.168.137.68", client=CAVClient())

