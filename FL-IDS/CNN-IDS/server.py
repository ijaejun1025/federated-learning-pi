#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import keras
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
    (X_train,y_train), (X_test, y_test) = utils.load_mnist()

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
    Model = keras.Sequential([
        

        keras.layers.Conv2D(96,(4,4),input_shape=(100778, 93, 1),activation='relu',padding='same'),
        keras.layers.Conv2D(64,(3,3),activation="relu",padding='same'),
        keras.layers.Conv2D(32,(2,2),activation="relu",padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512,activation="relu"),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(32,activation="relu"),
        keras.layers.Dense(2,activation="softmax"),
    
    
    ])

    Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
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


# In[ ]:





# In[ ]:





# In[ ]:




