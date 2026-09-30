# FL-IDS Deployment Guide (Jetson Xavier server + 6 Raspberry Pi clients)

## Topology

- **Server**: NVIDIA Jetson Xavier, runs `Server/server_cav.py`, listens on port `3040`.
- **Clients**: 6 separate Raspberry Pi boards, each runs exactly one `PiN/` folder. Copy only that one folder onto that specific Pi — every `PiN/` folder is self-contained (own `common_cav.py`/`model_cav.py` copies, own dataset).

Every client trains a Conv1D CNN on the same 10-feature schema (`CAN_ID, DLC, DATA0..DATA7`) but on a **different real dataset** — this is a non-IID (heterogeneous-by-source) federated learning setup, not 6 shards of one dataset.

## How the server evaluates the model when every client's data is different

This is worth explaining since it's not obvious at first glance:

1. Every round, each of the 6 clients trains locally on its own dataset and sends back its updated model weights plus how many samples it trained on.
2. The server (`FedAvg` strategy) combines those 6 weight sets into one **global model**, weighted by each client's sample count.
3. The server then evaluates *that global model itself* — clients are never asked to self-report evaluation metrics used for the official round record. At startup, the server built one **combined test set** by concatenating each of the 6 datasets' own held-out test split (33% of each dataset, reserved before any client ever sees it for training — see `get_test_split()` in each `PiN/*_utils_cav.py`).
4. So `round_metrics.csv` (written in `Server/`) reports how well the *single shared global model* performs across **all 6 vehicles/attack types at once**, not per-dataset. A model that overfits to just one client's data would score poorly here even if that one client's local training loss looked great — that's the actual point of this design: it measures whether federated averaging produced a model that generalizes across genuinely different CAN-bus sources, not just memorized one.
5. `min_fit_clients = min_evaluate_clients = min_available_clients = 6` is set explicitly, so a round only proceeds once **all 6** clients are connected — losing one client (e.g. it crashes or the dataset isn't downloaded) stalls the round rather than silently degrading to a partial average.

## Per-client dataset reference

| Pi | Dataset | Vehicle | Attack types | Source | Auto-download |
|---|---|---|---|---|---|
| **Pi1** | HCRL Car-Hacking Dataset | Hyundai YF Sonata | DoS, Fuzzy, Gear-Spoofing, RPM-Spoofing | [ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) | ✅ Dropbox |
| **Pi2** | HCRL CAN-intrusion-dataset (OTIDS) | Kia Soul | DoS, Fuzzy, Impersonation | [ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset](https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset) | ✅ Dropbox |
| **Pi3** | HCRL Survival Analysis Dataset | Chevrolet Spark, Hyundai Sonata, Kia Soul | Flooding, Fuzzy, Malfunction | [ocslab.hksecurity.net/Datasets/survival-ids](https://ocslab.hksecurity.net/Datasets/survival-ids) | ✅ Dropbox (password auto-applied) |
| **Pi4** | CIC IoV Dataset 2024 (CICIoV2024) | 2019 Ford | DoS, Spoofing (GAS/RPM/SPEED/STEERING_WHEEL) | [unb.ca/cic/datasets/iov-dataset-2024.html](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) | ⚠️ Google Drive mirror (original requires a one-time personal-info form) |
| **Pi5** | ROAD Dataset (Real ORNL Automotive Dynamometer) | Real test vehicle (dynamometer + road driving) | Real (non-simulated) fuzzing, fabrication, masquerade | [zenodo.org/records/10462796](https://zenodo.org/records/10462796) | ✅ Zenodo API |
| **Pi6** | CAN-MIRGU | Modern AV-capable production vehicle | 36 real driving-scenario attacks (replay, spoofing, suspension, masquerade, etc.) | [archive.ics.uci.edu/dataset/1035/can-mirgu](https://archive.ics.uci.edu/dataset/1035/can-mirgu) | ⚠️ UCI's own file is corrupted server-side; falls back to project's Google Drive mirror |

All datasets already have their raw files downloaded and cached in this repo's working copy (`PiN/<dataset folder>/` and `PiN/split_cache/`). On a fresh machine with no cache, each client will attempt its auto-download on first run (see the "Auto-download" column); Pi4/Pi6 will print a clear error with manual-download instructions if that fails.

## Commands to run

Replace `<JETSON_IP>` with the Jetson Xavier's actual LAN IP address. The client scripts default to `192.168.137.68:3040` (a common USB-tether/ICS subnet address) — if that's not the Jetson's real address on your network, pass `--server-address` explicitly as shown.

**On the Jetson Xavier (start this first — it waits for all 6 clients before training begins):**
```bash
cd FL-IDS/LR-IDS/Server
python server_cav.py
```

**On each Raspberry Pi** (one command per board, run after the server is up):

```bash
# Pi1 — Car-Hacking Dataset
cd FL-IDS/LR-IDS/Pi1
python carhacking_client_cav.py --server-address <JETSON_IP>:3040

# Pi2 — OTIDS
cd FL-IDS/LR-IDS/Pi2
python otids_client_cav.py --server-address <JETSON_IP>:3040

# Pi3 — Survival Analysis Dataset
cd FL-IDS/LR-IDS/Pi3
python survival_client_cav.py --server-address <JETSON_IP>:3040

# Pi4 — CIC IoV Dataset 2024
cd FL-IDS/LR-IDS/Pi4
python ciciov_client_cav.py --server-address <JETSON_IP>:3040

# Pi5 — ROAD Dataset
cd FL-IDS/LR-IDS/Pi5
python road_client_cav.py --server-address <JETSON_IP>:3040

# Pi6 — CAN-MIRGU
cd FL-IDS/LR-IDS/Pi6
python canmirgu_client_cav.py --server-address <JETSON_IP>:3040
```

Each Pi also accepts `--client-id <n>` purely as a log label (defaults are already set per Pi: 0-5); it has no effect on which dataset is used or on training.

## Output

- `Server/round_metrics.csv` — one row per round: accuracy, precision, recall, F1, detection_rate, FPR, FNR, ROC-AUC, MCC, elapsed_sec.
- `Server/confusion_matrix_round_<N>.xlsx` — confusion matrix for that round's global model.
- 3 rounds run by default (`num_rounds=3` in `server_cav.py`).

## Requirements

Each machine (Jetson + every Pi) needs: `flwr`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `openpyxl`. There is no `requirements.txt` in the repo yet — install these manually, or ask to have one generated.

## Known caveats before relying on this in production

- **Raspberry Pi RAM**: Pi5 (ROAD) and Pi6 (CAN-MIRGU) build multi-million-row training arrays (ROAD: ~15M rows, CAN-MIRGU: ~3.4M rows after down-sampling). This was validated on a 16GB dev machine; a Raspberry Pi 3 (1GB RAM) is very likely to run out of memory on these two specifically, even after the memory-efficiency fixes applied to the loaders. Worth checking actual available RAM per board before relying on Pi5/Pi6 completing training unattended.
- **`flwr.server.start_server()` is deprecated** in the installed Flower version (1.39) in favor of the `flower-superlink` CLI. It still works today but may be removed in a future Flower release.
- Pi2 (OTIDS Fuzzy/Impersonation) and Pi6 (CAN-MIRGU) use documented heuristic/whole-file labeling rather than verified per-row ground truth — see the caveats in each `*_utils_cav.py` module docstring.
