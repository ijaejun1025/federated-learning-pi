# Centralized baseline

`train_central.py` trains the federated CNN on all six clients' data at once,
with no federated learning, for comparison with the FL run. It imports the
model, features and scaling from `federated-cnn/Server`, so both sides always
use the same code.

| | Federated (`federated-cnn`) | Centralized (`train_central.py`) |
|---|---|---|
| Training data | each Pi's own local train split | the six local train splits pooled and shuffled |
| Passes over the data | 15 rounds × 1 local epoch | 15 epochs |
| Batch size | 256 | 256 |
| Learning rate | 1e-3 → 1e-4 over the rounds | 1e-3 → 1e-4 over the epochs |
| Evaluation | server `Evaluator` after every round | the same `Evaluator` after every epoch |
| Final threshold | median of the six Pis' proposed thresholds | median of the same six thresholds, from each dataset's validation normals |

## Run

Run it on the PC from this folder, with the `split_cache/*_v1.npz` caches in
`federated-cnn/PiN/` (all nine, so that results 2–4 are included):

```
python train_central.py --seed 0
python train_central.py --seed 1
python train_central.py --seed 2
```

Each run writes to `results/seed<N>/`:

- the server's CSVs: `round_metrics.csv`, `dataset_metrics.csv`,
  `final_results.csv`, `final_dataset_metrics.csv`, `generalization_files.csv`
  ("round" is the epoch)
- `progress_log.txt`
- `run.json`: per-epoch training loss, time and macro metrics, plus the six
  thresholds
- `weights/round_NN.weights.h5`: the model after every epoch

A run refuses to start if its output folder is not empty, because the CSVs
are appended to. `results/` is not tracked by git.
