# FL-IDS Deployment Guide (Jetson Xavier server + 6 Raspberry Pi clients)

## Topology

- **Server**: NVIDIA Jetson Xavier, runs `Server/server_cav.py`, listens on port `3040`.
- **Clients**: 6 separate Raspberry Pi boards, each runs exactly one `PiN/` folder. Copy only that one folder onto that specific Pi — every `PiN/` folder is self-contained (own `common_cav.py`/`model_cav.py` copies, own dataset).

Every client trains a Conv1D CNN on the same 12-feature schema but on a **different real dataset** — this is a non-IID (heterogeneous-by-source) federated learning setup, not 6 shards of one dataset. Features per CAN frame (see `common_cav.py`): `DLC, DATA0..DATA7`, log time since the previous frame with the same CAN ID, that time divided by the ID's usual period (from the client's attack-free train captures), and the number of data bits changed since that previous frame relative to the ID's usual change (`bits_rel` = log2((bits + 1) / (mean bits + 1)), the mean also from the attack-free train captures). The raw bit count meant opposite things on different datasets (an unchanged payload is mostly an attack in Car-Hacking, OTIDS, Survival and can-train-and-test, but mostly normal in ROAD, where many IDs never change); relative to each ID it means the same everywhere, and in the PC simulation it raised the macro recall at FPR 1e-3 over the six test sets from 0.90 to 0.99 (ROAD from 0.61 to 0.99). The CAN ID itself is not a feature. Every client, the server and the new-vehicle sets use the same fixed scaling (`common_cav.FEATURE_CENTER` / `FEATURE_SCALE`), not a scaler fit on each client's own data, so one frame becomes the same model input everywhere.

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

Build the caches once on a PC with the raw data and copy them with the code; a client or the server only falls back to the raw files (and their auto-download) when no cache with the current `DATASET_KEY` exists. A cache is `PiN/split_cache/<DATASET_KEY>.npz`, the key being the dataset name and a version:

| Pi | Cache |
|---|---|
| Pi1 | `split_cache/car_hacking_v1.npz` |
| Pi2 | `split_cache/otids_v1.npz` |
| Pi3 | `split_cache/survival_v1.npz` |
| Pi4 | `split_cache/can_train_test_set04_v1.npz` (training rows, and the Forester test set of result 1) |
| Pi4, Jetson only | `split_cache/can_train_test_subaru_new_attacks_v1.npz` (result 2), `split_cache/can_train_test_traverse_known_attacks_v1.npz` (result 3), `split_cache/can_train_test_traverse_new_attacks_v1.npz` (result 4); each Traverse file also holds the Traverse's attack-free calibration recording |
| Pi5 | `split_cache/road_v1.npz` |
| Pi6 | `split_cache/can_mirgu_v1.npz` |

Each Pi needs only its own cache (Pi4: `can_train_test_set04_v1.npz` only); the Jetson needs all nine, in the same `PiN/split_cache/` folders. Caches are found by key alone, so the version is bumped (v1 -> v2) whenever the rows, labels, features or sampling change, and a cache with an old key is never read again: delete old `split_cache/*.npz` files from the Pis and the Jetson when the keys change. The test size, random state and raw-file signature a cache was built with are stored inside it; a cache built with another test size or random state stops the run with an error.

Class balance (`common_cav.Sampling`): every client gets the same number of training rows (`TRAIN_ROWS_PER_CLIENT`, interim 600,000, including its local validation split), with all of its attack frames up to 20% of them and normal frames filling the rest; FedAvg weights clients by row count, so equal sizes keep any one dataset from dominating. Test sets keep every attack frame and at most 2,000,000 normal frames per dataset, uniformly sampled; each kept normal row carries a weight (`w_test`) so that weighted metrics estimate the full test set. Rows are sampled after the features are computed, so the time features always see every frame.

Local training (`model_cav.py`): each round, every client trains `LOCAL_EPOCHS = 1` epoch at `BATCH_SIZE = 256` (about 1,875 steps on its 480,000 local training rows), without class weights (`USE_CLASS_WEIGHTS = False`; sampling already caps the attack share), and the server runs `NUM_ROUNDS = 15` rounds. The learning rate falls every round from 1e-3 (round 1) to 1e-4 (round 15) by the same factor (`model_cav.learning_rate`); the server sends it with each round (`learning_rate` in the fit config) and each Pi prints it in its "fit started" line. With a fixed rate the global model kept moving between rounds, since each client pulls it toward its own data; with the falling rate the last rounds barely change, so the final round is not a matter of luck. Since every client's data comes from a different vehicle, long local training pulls the six models apart before they are averaged; in a PC simulation of this setup, batch 32 x 3 epochs per round collapsed by round 3, while this setting kept improving and levelled off after rounds 6-9.

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

All server metrics are weighted by the test rows' `w_test` (so they estimate the full test sets), and precision / F1 are those of the attack class; `f1_weighted_avg` (sklearn's average="weighted", near 1 whenever normals dominate) is kept for comparison only. Columns: accuracy, precision, detection_rate, f1_score, fpr, fnr, roc_auc, pr_auc, mcc, f1_weighted_avg, loss, weighted counts (tn/fp/fn/tp, num_examples, attack_examples) and `sampled_rows`.

Every round (threshold 0.5, for monitoring):
- `Server/dataset_metrics.csv` — one row per client test set.
- `Server/round_metrics.csv` — `macro_avg` (mean over the six datasets, each counts the same) and `pooled_weighted` (all rows pooled; CAN-MIRGU's large normal share dominates).
- Round 0 (the untrained model) only checks that the evaluation code runs, on 50,000 rows per dataset.

After the last round, once the clients have reported their thresholds:
- `Server/final_results.csv` — the four results: `1_all_client_test_sets` (macro_avg and pooled_weighted), `2_trained_vehicle_new_attacks` (Forester `test_03` without `double-1`), `3_new_vehicle_known_attacks` (2011 Chevrolet Traverse `test_02`), `4_new_vehicle_new_attacks` (Traverse `test_04`). Each at threshold 0.5 and at the median of the clients' proposed thresholds (`client_median_fpr_0.001`: the score each client's local validation normals exceed with probability 0.1%; only that number leaves a Pi). The Traverse sets also at `vehicle_calibrated_fpr_0.001`, set on the Traverse's own attack-free recording (no labels).
- `Server/final_dataset_metrics.csv` — result 1 per dataset at both thresholds.
- `Server/generalization_files.csv` — results 2-4 per capture (= attack type).
- Results 2-4 need the three generalization caches in `Pi4/split_cache/` on the Jetson (`can_train_test_subaru_new_attacks_v1.npz`, `can_train_test_traverse_known_attacks_v1.npz`, `can_train_test_traverse_new_attacks_v1.npz`); the server prints `found` / `NOT found` for each at startup and skips results 2-4 unless all three are there.
- `Server/client_metrics.csv` — each Pi's `local_val_*` numbers (training-like mix of 3-20% attacks, not comparable with the test metrics) and its proposed threshold.
- 15 rounds run by default (`NUM_ROUNDS` in `server_cav.py`).

## Requirements

Each machine (Jetson + every Pi) needs: `flwr`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `openpyxl`. There is no `requirements.txt` in the repo yet — install these manually, or ask to have one generated.

## Known caveats before relying on this in production

- **Raspberry Pi RAM**: building a split from raw data needs far more than 1GB (CAN-MIRGU reads ~118M frames). Build the caches on a PC; a Pi only reads the training rows of its cache (600,000 rows by default), never the test rows.
- **`flwr.server.start_server()` is deprecated** in the installed Flower version (1.39) in favor of the `flower-superlink` CLI. It still works today but may be removed in a future Flower release.
