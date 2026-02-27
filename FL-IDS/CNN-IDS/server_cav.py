#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self,rnd,results,failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy(
    min_fit_clients=6,        # Minimum number of clients for training
    min_evaluate_clients=6,   # Minimum number of clients for evaluation
    min_available_clients=6,  # Minimum number of clients that must connect
)

# Start Flower server for three rounds of federated learning with 1Gb of data
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:3040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)


# In[ ]:




