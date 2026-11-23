# Federated Learning for CAN Intrusion Detection (Final_Defense)

## Overview

An intrusion detection system (IDS) reads each message on a car's CAN bus
and says *normal* or *attack*. This project trains one IDS with
**federated learning (FL)**:

- **6 clients = 6 different cars.** Each **Raspberry Pi 3** holds a different
  public CAN attack dataset. The data stays on the Pi; only model weights are
  sent. Since every Pi's data looks different, the setting is **non-IID**.
- **Server:** an **NVIDIA Jetson Xavier** averages the six models each round
  (FedAvg), using the **[Flower](https://flower.ai/)** framework.
- **Model:** a small **1D CNN** on 12 numbers per CAN message.

The same model, data, settings and evaluation are also trained without FL,
and the three are compared:

| Setting | Trains on | Role |
|---|---|---|
| **Federated** (`federated-cnn/`) | each Pi its own data, weights averaged | the proposed method |
| **Centralized** (`centralized-cnn/`) | all six datasets pooled | upper reference (if data could be shared) |
| **Local** (`local-cnn/`) | one Pi's data only (six models) | lower reference (one car alone) |

## Datasets

| Pi | Dataset | Vehicle | Attacks used | Train : test (raw frames) |
|---|---|---|---|---|
| 1 | Car-Hacking (HCRL) | Hyundai YF Sonata | DoS, fuzzy, gear spoofing, RPM spoofing | 9.4M : 4.5M (68 : 32) |
| 2 | OTIDS (HCRL) | Kia Soul | DoS | 1.9M : 0.9M (68 : 32) |
| 3 | Survival Analysis (HCRL) | Chevrolet Spark, Hyundai Sonata, Kia Soul | flooding, fuzzy, malfunction | 1.1M : 0.5M (68 : 32) |
| 4 | can-train-and-test, set_04 | 2017 Subaru Forester | interval, rpm, speed (± accessory), standstill, systematic | 9.5M : 6.9M (58 : 42) |
| 5 | ROAD (ORNL) | one car (make not published) | fuzzing, correlated signal, max coolant temp, max speedometer, reverse light on/off | 15.8M : 7.6M (68 : 32) |
| 6 | CAN-MIRGU | one car, real driving | 22 types (DoS, fuzzing, replay, warning-light, speedometer, steering, …) | 75.3M : 45.5M (62 : 38) |

How the data is used:

- **Train and test never share a frame.** They are split by time (10 blocks
  per recording, blocks 2, 5, 8 = test) or by file / recording / day, so test
  frames come from a different part of the drive.
- **Training: 600,000 rows per Pi**, at most 20% attacks (if there are fewer,
  all of them), the rest random normal frames. Each Pi uses 480,000 to train
  and 120,000 to validate. Equal sizes give every Pi the same weight in
  FedAvg.
- **Testing:** every attack frame and up to 2,000,000 random normal frames,
  weighted so the metrics equal those of the full test set.
- **Generalization (results 2–4):** Pi4's dataset also has data no client
  trains on: new attacks on the Forester (DoS, double, triple,
  force-neutral, fuzzing), and a **new car**, a 2011 Chevrolet Traverse,
  with known and with new attacks.
- **Features (12):** DLC, the 8 payload bytes, time since the last frame
  with the same ID, that time over the ID's usual period, and how many
  payload bits changed compared with what is usual for the ID. The CAN ID
  itself is not used. Scaling is fixed, the same on every device.
- The data is prepared once on a PC into
  `federated-cnn/PiN/split_cache/*_v1.npz` (9 files); the devices only read
  these.

## Model and settings

**1D CNN** (`model_cav.py`, 290,082 parameters, 1.16 MB):
input 12 × 1 → Conv1D 96 (kernel 4) → Conv1D 64 (kernel 3) → Conv1D 32
(kernel 2) → Dropout 0.5 → Flatten → Dense 512 → 128 → 32 → Dense 2
(softmax). ReLU everywhere, "same" padding.

| Setting | Value |
|---|---|
| Optimizer / loss | Adam / sparse categorical cross-entropy |
| Batch size | 256 |
| Learning rate | 1e-3 in round 1, down to 1e-4 in round 15 (same factor each round) |
| Class weights | none (attacks already capped at 20%) |
| Federated | 15 rounds × 1 local epoch, all 6 Pis every round, FedAvg |
| Centralized | 15 epochs on the 2,880,000 pooled rows |
| Local | 15 epochs on one Pi's 480,000 rows |

All three see each training row 15 times with the same schedule.

**Threshold.** The model gives an attack score from 0 to 1. Results are
reported at 0.5, and at a threshold each Pi proposes from its own normal
validation frames (the score only 0.1% of them exceed); the server uses the
median of the six. For the new car there is also a threshold set on its own
attack-free recording.

## How to run

**Setup:** `pip install flwr tensorflow numpy pandas scikit-learn` on the
Jetson and every Pi. The Jetson gets the whole repository (code only) with
all 9 cache files; each Pi gets only its own `federated-cnn/PiN/` folder and
cache. Run the three settings one after another, not at the same time.

**1. Federated** — start the server, then one client on each Pi:

```
# Jetson
cd federated-cnn/Server && python server_cav.py

# Pi1 … Pi6 (one line per Pi)
cd federated-cnn/Pi1 && python carhacking_client_cav.py --server-address <JETSON_IP>:3040
cd federated-cnn/Pi2 && python otids_client_cav.py      --server-address <JETSON_IP>:3040
cd federated-cnn/Pi3 && python survival_client_cav.py   --server-address <JETSON_IP>:3040
cd federated-cnn/Pi4 && python cantt_client_cav.py      --server-address <JETSON_IP>:3040
cd federated-cnn/Pi5 && python road_client_cav.py       --server-address <JETSON_IP>:3040
cd federated-cnn/Pi6 && python canmirgu_client_cav.py   --server-address <JETSON_IP>:3040
```

Training starts when all six are connected.

**2. Centralized** (Jetson, after FL):

```
cd centralized-cnn && python train_central.py --seed 0
```

**3. Local** (Jetson, one Pi at a time):

```
cd local-cnn && python train_local.py --client Pi1 --seed 0   # … up to Pi6
```

More details: [federated-cnn/DEPLOYMENT.md](federated-cnn/DEPLOYMENT.md).

## Results

| Setting | Folder |
|---|---|
| Federated | `federated-cnn/Server/results/` |
| Centralized | `centralized-cnn/results/` |
| Local | `local-cnn/results/Pi1/` … `Pi6/` |

All three write the same files ("round" = epoch for centralized / local).
Move the folder away before running again.

**Final results** (after the last round):

| Result | Test data | Question |
|---|---|---|
| 1 | the six Pis' test sets | does one model work on all six cars? |
| 2 | Forester, new attacks | does it catch attacks it never saw? |
| 3 | Traverse, known attacks | does it work on a new car? |
| 4 | Traverse, new attacks | new car and new attacks (hardest) |

**Files:**

| File | Contents | Use |
|---|---|---|
| `round_metrics.csv` | detection rate, FPR, precision, F1, ROC-AUC, PR-AUC every round (six test sets, threshold 0.5) | learning curve |
| `dataset_metrics.csv` | the same per dataset | which car learns slowly (non-IID) |
| `final_results.csv` | results 1–4 for every threshold rule | main numbers for the paper |
| `final_dataset_metrics.csv` | result 1 per dataset | per-car result; the six local runs give a 6 × 6 table |
| `generalization_files.csv` | results 2–4 per attack type | which new attacks are missed |
| `hardware_metrics.csv` | time, CPU, memory, temperature, clock, throttling, bytes sent, inference time | can it run on edge devices? |
| `client_metrics.csv` (FL) | each Pi's validation numbers and proposed threshold | where the final threshold comes from |
| `run.json` (centralized / local) | settings, thresholds, loss per epoch | record of the run |
| `weights/round_NN.weights.h5` | the model after each round | compute other metrics later without retraining |
| `progress_log.txt` | progress lines | follow or debug a run |

For result 1, `macro_avg` (each dataset counts the same) is fairer for
non-IID data than `pooled_weighted` (all frames together).

**Note.** Repeated attacks such as DoS send the same message again and
again, so many test attacks look exactly like training attacks (different
frames, same values). Result 1 is therefore partly "known attacks"; results
2–4 show real generalization.

## Layout

```
federated-cnn/      FL: Server/ (Jetson), Pi1/ … Pi6/ (one folder per Pi), DEPLOYMENT.md
centralized-cnn/    centralized baseline (train_central.py holds the training code)
local-cnn/          local baseline (train_local.py runs train_central.py --client PiN)
```
