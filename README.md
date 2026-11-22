# Final_Defense

Federated learning for CAN-bus intrusion detection: a Conv1D CNN trained with
Flower FedAvg on six Raspberry Pi clients, each holding a different real CAN
dataset (non-IID), with a Jetson Xavier as the server.

## Layout

- [federated-cnn/](federated-cnn/) — the federated CNN IDS
  - `Server/` — Flower server (`server_cav.py`), runs on the Jetson
  - `Pi1/` … `Pi6/` — one self-contained client folder per Raspberry Pi
    (client script, dataset loader, shared `common_cav.py` / `model_cav.py`)
  - [DEPLOYMENT.md](federated-cnn/DEPLOYMENT.md) — datasets, features,
    training settings, caches and how to run it
