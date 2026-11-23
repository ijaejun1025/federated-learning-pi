# Local baseline

`train_local.py` trains the federated CNN on one client's data only, as that
Pi would train alone, with no federated learning. It is the lower reference
point: FL should beat it on the other clients' data, and the centralized
baseline (`../centralized-cnn`) is the upper one.

It runs `../centralized-cnn/train_central.py --client PiN`, so the model,
features, batch size (256), 15 epochs, learning rate (1e-3 → 1e-4) and
evaluation (the server's `Evaluator`) are the same as in the other two
settings; only the training data differ.

| | Federated | Centralized | Local |
|---|---|---|---|
| Training data | each Pi's own split, weights averaged | the six splits pooled | one Pi's split |
| Final threshold | median of the six Pis' thresholds | median of the six | that Pi's own |
| Test sets | all six + results 2–4 | the same | the same |

Every local model is scored on all six test sets, so the six runs together
give a 6×6 train-client × test-set matrix (`final_dataset_metrics.csv`).

## Run

On the Jetson from this folder, after the FL run and not at the same time as
another run (see `../centralized-cnn/README.md`), with the nine
`split_cache/*_v1.npz` caches in `federated-cnn/PiN/`:

```
python train_local.py --client Pi1 --seed 0
...
python train_local.py --client Pi6 --seed 0
```

Each run takes about 1/6 of the centralized training time, but the
evaluation after every epoch takes as long as there.

Each run writes to `results/PiN/` the same files as the centralized
baseline (see `../centralized-cnn/README.md`), including `hardware_metrics.csv` with
node `local-PiN`. A run refuses to start if its
output folder is not empty; the seed is in `run.json`, and another seed
needs `--out`, e.g. `--out results/Pi3_seed1`. `results/` is not tracked by git.
