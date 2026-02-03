#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flwr as fl
import sys
import numpy as np
import os

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
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

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning with 1Gb of data
# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:3040",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)


# In[ ]:




