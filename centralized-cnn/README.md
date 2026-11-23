# Centralized baseline

`train_central.py` trains the federated CNN on all six clients' data at once,
with no federated learning, for comparison with the FL run (the upper
reference point; `../local-cnn` is the lower one). It imports the model,
features and scaling from `federated-cnn/Server`, so both sides always use
the same code.

The local baseline (`../local-cnn/train_local.py`) is this same script with
`--client PiN`: it trains on that one client's data and writes to
`local-cnn/results/PiN/`.

| | Federated (`federated-cnn`) | Centralized (`train_central.py`) |
|---|---|---|
| Training data | each Pi's own local train split | the six local train splits pooled and shuffled |
| Passes over the data | 15 rounds × 1 local epoch | 15 epochs |
| Batch size | 256 | 256 |
| Learning rate | 1e-3 → 1e-4 over the rounds | 1e-3 → 1e-4 over the epochs |
| Evaluation | server `Evaluator` after every round | the same `Evaluator` after every epoch |
| Final threshold | median of the six Pis' proposed thresholds | median of the same six thresholds, from each dataset's validation normals |

## Run

Run it on the Jetson (the FL server), from this folder, **after** the FL run:
sharing the Jetson with the FL server would skew both runs' times and
hardware numbers. It reads the same nine `split_cache/*_v1.npz` caches in
`federated-cnn/PiN/` that the server needs (each also holds that client's
train split). Copy only the code: if `results/` from another machine comes
along, the run refuses to start.

```
python train_central.py --seed 0
```

Another seed needs its own output folder, e.g.
`python train_central.py --seed 1 --out results/seed1`.

The run writes to `results/` (the seed is in `run.json`):

- the server's CSVs: `round_metrics.csv`, `dataset_metrics.csv`,
  `final_results.csv`, `final_dataset_metrics.csv`, `generalization_files.csv`
  ("round" is the epoch)
- `progress_log.txt`
- `run.json`: per-epoch training loss, time and macro metrics, plus the six
  thresholds
- `weights/round_NN.weights.h5`: the model after every epoch
- `hardware_metrics.csv`: the FL server's columns (`hardware_cav`). One `fit`
  and one `evaluate` row per epoch (node `central`, or `local-PiN`), and an
  `inference` row at the end: `infer_batch_ms_per_row` at batch 256 and
  `infer_frame_ms_median` / `_p95` one frame at a time, as each Pi measures it

A run refuses to start if its output folder is not empty, because the CSVs
are appended to. `results/` is not tracked by git.
