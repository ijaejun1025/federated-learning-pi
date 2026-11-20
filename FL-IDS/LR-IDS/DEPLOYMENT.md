# FL-IDS Deployment Guide (Jetson Xavier server + 6 Raspberry Pi clients)

## Topology

- **Server**: NVIDIA Jetson Xavier, runs `Server/server_cav.py`, listens on port `3040`.
- **Clients**: 6 separate Raspberry Pi boards, each runs exactly one `PiN/` folder. Copy only that one folder onto that specific Pi — every `PiN/` folder is self-contained (own `common_cav.py`/`model_cav.py` copies, own dataset).

Every client trains a Conv1D CNN on the same 12-feature schema but on a **different real dataset** — this is a non-IID (heterogeneous-by-source) federated learning setup, not 6 shards of one dataset. Features per CAN frame (see `common_cav.py`): `DLC, DATA0..DATA7`, log time since the previous frame with the same CAN ID, that time divided by the ID's usual period (from the client's attack-free train captures), and the number of data bits changed since that previous frame. The CAN ID itself is not a feature.

## How the server evaluates the model when every client's data is different

This is worth explaining since it's not obvious at first glance:

1. Every round, each of the 6 clients trains locally on its own dataset and sends back its updated model weights plus how many samples it trained on.
2. The server (`FedAvg` strategy) combines those 6 weight sets into one **global model**, weighted by each client's sample count.
3. The server then evaluates *that global model itself* — clients are never asked to self-report evaluation metrics used for the official round record. At startup, the server built one **combined test set** by concatenating each of the 6 datasets' own held-out test split (time blocks 2/5/8 of each capture, whole held-out files, or the dataset's own test set — see each `PiN/*_utils_cav.py` docstring and `get_test_split()`).
4. So `round_metrics.csv` (written in `Server/`) reports how well the *single shared global model* performs across **all 6 vehicles/attack types at once**; `dataset_metrics.csv` adds the same metrics per dataset. A model that overfits to just one client's data would score poorly here even if that one client's local training loss looked great — that's the actual point of this design: it measures whether federated averaging produced a model that generalizes across genuinely different CAN-bus sources, not just memorized one.
5. `min_fit_clients = min_evaluate_clients = min_available_clients = 6` is set explicitly, so a round only proceeds once **all 6** clients are connected — losing one client (e.g. it crashes or the dataset isn't downloaded) stalls the round rather than silently degrading to a partial average.

## Per-client dataset reference

| Pi | Dataset | Vehicle | Attack types | Source | Auto-download |
|---|---|---|---|---|---|
| **Pi1** | HCRL Car-Hacking Dataset | Hyundai YF Sonata | DoS, Fuzzy, Gear-Spoofing, RPM-Spoofing | [ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) | ✅ Dropbox |
| **Pi2** | HCRL CAN-intrusion-dataset (OTIDS) | Kia Soul | DoS (Fuzzy and Impersonation excluded: no per-frame ground truth) | [ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset](https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset) | ✅ Dropbox |
| **Pi3** | HCRL Survival Analysis Dataset | Chevrolet Spark, Hyundai Sonata, Kia Soul | Flooding, Fuzzy, Malfunction | [ocslab.hksecurity.net/Datasets/survival-ids](https://ocslab.hksecurity.net/Datasets/survival-ids) | ✅ Dropbox (password auto-applied) |
| **Pi4** | can-train-and-test, set_04 (`train_01` / `test_01_known_vehicle_known_attack`) | 2017 Subaru Forester | Interval, RPM and speed spoofing, standstill, systematic | [bitbucket.org/brooke-lampe/can-train-and-test](https://bitbucket.org/brooke-lampe/can-train-and-test) | ✅ Bitbucket API |
| **Pi5** | ROAD Dataset (Real ORNL Automotive Dynamometer) | Real test vehicle (dynamometer + road driving) | Real (non-simulated) fuzzing and fabrication (masquerade copies excluded) | [zenodo.org/records/10462796](https://zenodo.org/records/10462796) | ✅ Zenodo API |
| **Pi6** | CAN-MIRGU | Modern AV-capable production vehicle | Real-attack captures (injection, replay, fuzzing, DoS); masquerade and suspension captures excluded | [archive.ics.uci.edu/dataset/1035/can-mirgu](https://archive.ics.uci.edu/dataset/1035/can-mirgu) | ⚠️ UCI's own file is corrupted server-side; falls back to project's Google Drive mirror |

Every dataset is labeled per frame (Pi1/Pi3: the dataset's R/T flag; Pi2: injected ID 0x000; Pi4/Pi6: the dataset's 0/1 column; Pi5: interval + ID + data pattern from the capture metadata). Pi1 also needs `normal_run_data.txt`, which ships zipped as `normal_run_data.7z` and must be extracted by hand (it only feeds the ID period table).

Build the caches once on a PC with the raw data (`PiN/split_cache/*.npz`) and copy them with the code; a client or the server only falls back to the raw files (and their auto-download) when no cache with the current `DATASET_KEY` exists.

## Commands to run

Replace `<JETSON_IP>` with the Jetson Xavier's actual LAN IP address. The client scripts default to `192.168.137.250:3040` (a common USB-tether/ICS subnet address) — if that's not the Jetson's real address on your network, pass `--server-address` explicitly as shown.

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

# Pi4 — can-train-and-test (set_04)
cd FL-IDS/LR-IDS/Pi4
python cantt_client_cav.py --server-address <JETSON_IP>:3040

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
- `Server/generalization_metrics.csv` / `generalization_files.csv` — written once, after the last round: the final model on sets no client trains on (Pi4's can-train-and-test: Forester unseen attacks `test_03` without `double-1`; the 2011 Chevrolet Traverse `test_02` known attacks and `test_04` unseen attacks). Each set at threshold 0.5; the Traverse sets also at thresholds set from the Traverse's own attack-free recording for 1% / 0.1% FPR (no labels). Per capture (= attack type) in the `_files` CSV. Needs `Pi4/split_cache/cantt04_gen_h12_*.npz` on the Jetson; without it the server says so at startup and skips this step. These numbers are kept out of `round_metrics.csv` and `dataset_metrics.csv`.
- 3 rounds run by default (`num_rounds=3` in `server_cav.py`).

## Requirements

Each machine (Jetson + every Pi) needs: `flwr`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `openpyxl`. There is no `requirements.txt` in the repo yet — install these manually, or ask to have one generated.

## Known caveats before relying on this in production

- **Raspberry Pi RAM**: building a split from raw data needs far more than 1GB (CAN-MIRGU reads ~118M frames). Build the caches on a PC; a Pi only streams its `CLIENT_TRAIN_FRACTION` of the cached training pool.
- **`flwr.server.start_server()` is deprecated** in the installed Flower version (1.39) in favor of the `flower-superlink` CLI. It still works today but may be removed in a future Flower release.
