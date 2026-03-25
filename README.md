# Final_Defense

Federated learning experiments for intrusion detection, including CNN- and LR-based IDS variants.

## LR-IDS CAV Data Split

For the CAV-based LR-IDS flow:

- [FL-IDS/LR-IDS/utils_cav.py](FL-IDS/LR-IDS/utils_cav.py) preprocesses the full CAV dataset and returns the full feature and label arrays.
- [FL-IDS/LR-IDS/client_cav.py](FL-IDS/LR-IDS/client_cav.py) performs the train/test split on each client.
- Each client uses the same test ratio, but because the split is randomized per client run, different Raspberry Pi clients can evaluate on different test subsets.
- [FL-IDS/LR-IDS/server_cav.py](FL-IDS/LR-IDS/server_cav.py) keeps a separate server-side evaluation split for global model assessment.