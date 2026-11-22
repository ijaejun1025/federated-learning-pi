#!/usr/bin/env python
# coding: utf-8

import datetime
import math
import time

import numpy as np
from tensorflow import keras

# Each client proposes a decision threshold: the attack score that only this
# share of its local validation normal frames exceed. The server uses the
# median of the six (only this number leaves a Pi, never its data).
THRESHOLD_TARGET_FPR = 0.001

# Local training per round. Each client's data is a different vehicle (non-IID),
# so long local training pulls the six models apart before they are averaged
# ("client drift"): batch 32 x 3 epochs (~45,000 steps per round) collapsed in
# the PC simulation of this setup, while one epoch at batch 256 (~1,875 steps)
# with more rounds kept improving. The attack share is already capped at 20%
# by sampling, so rows are not reweighted by class (balanced weights gave
# more false alarms at the low thresholds that matter here).
BATCH_SIZE = 256
LOCAL_EPOCHS = 1
USE_CLASS_WEIGHTS = False

# Client learning rate per round: Adam's 1e-3 in round 1, falling by the same
# factor every round to 1e-4 in the last round. With a fixed rate the global
# model kept moving between rounds (in the PC simulation the macro recall at
# FPR 1e-3 still changed by 0.01-0.05 per round after round 7, ROAD by up to
# 0.28), since each client keeps pulling it toward its own data; a falling
# rate lets the average settle. The server sends the rate with every round
# (``learning_rate`` in the fit config).
LEARNING_RATE_START = 1e-3
LEARNING_RATE_END = 1e-4


def learning_rate(server_round: int, num_rounds: int) -> float:
    """Geometric decay from ``LEARNING_RATE_START`` (round 1) to ``LEARNING_RATE_END`` (last round)."""
    if num_rounds <= 1:
        return LEARNING_RATE_START
    step = (server_round - 1) / (num_rounds - 1)
    return float(LEARNING_RATE_START * (LEARNING_RATE_END / LEARNING_RATE_START) ** step)


def set_learning_rate(model, rate: float) -> None:
    """Set the compiled optimizer's learning rate (Adam's moment estimates are kept)."""
    model.optimizer.learning_rate = rate


def class_weights(y):
    """Balanced class weights if ``USE_CLASS_WEIGHTS``, else None (every row counts the same)."""
    if not USE_CLASS_WEIGHTS:
        return None
    counts = np.bincount(y)
    return {i: len(y) / (len(counts) * c) for i, c in enumerate(counts)}


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(96, 4, activation="relu", padding="same"),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(32, 2, activation="relu", padding="same"),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(LEARNING_RATE_START), "sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    return model


class BatchSequence(keras.utils.PyDataset):
    """Feeds Keras one batch at a time instead of whole arrays.

    Given whole NumPy arrays, fit/evaluate/predict keep several internal
    copies of the full array -- too much for a 1GB Pi. ``class_weight``
    becomes per-row sample weights, which is what Keras itself turns it into
    for array input.
    """

    def __init__(self, x, y=None, batch_size=BATCH_SIZE, shuffle=False, class_weight=None):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.row_weight = None
        if class_weight is not None:
            self.row_weight = np.ones(int(y.max()) + 1, dtype=np.float32)
            for cls, weight in class_weight.items():
                self.row_weight[cls] = weight
        self.rng = np.random.default_rng()
        self.order = self.rng.permutation(len(x)) if shuffle else None

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, i):
        rows = slice(i * self.batch_size, (i + 1) * self.batch_size)
        if self.shuffle:
            rows = self.order[rows]
        if self.y is None:
            return self.x[rows]
        y = self.y[rows]
        if self.row_weight is None:
            return self.x[rows], y
        return self.x[rows], y, self.row_weight[y]

    def on_epoch_end(self):
        if self.shuffle:
            self.order = self.rng.permutation(len(self.x))


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


class ProgressLog:
    """Timestamped ``[PROGRESS]`` lines with the client's name and the round,
    printed and appended to a file, so a Pi's progress can still be read back
    after the Keras progress bar has scrolled it out of the tmux pane.

    The round comes from the ``server_round``/``num_rounds`` entries the server
    puts in each fit/evaluate config.
    """

    def __init__(self, client_name, path):
        self.client_name = client_name
        self.path = path

    def write(self, config, message):
        round_label = f"round {config.get('server_round', '?')}/{config.get('num_rounds', '?')}"
        line = (
            f"{datetime.datetime.now().isoformat(timespec='seconds')} [PROGRESS] "
            f"{self.client_name} {round_label} {message}"
        )
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


class EpochProgress(keras.callbacks.Callback):
    """Writes one ProgressLog line per finished epoch."""

    def __init__(self, progress_log, config):
        super().__init__()
        self.progress_log = progress_log
        self.config = config
        self.epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        stats = ", ".join(
            f"{key}={logs[key]:.4f}"
            for key in ("loss", "val_loss", "val_sparse_categorical_accuracy")
            if key in logs
        )
        self.progress_log.write(
            self.config,
            f"epoch {epoch + 1}/{self.params['epochs']} done in "
            f"{format_duration(time.time() - self.epoch_start)} ({stats})",
        )


def progress_callbacks(progress_log, config):
    """Callbacks for model.fit. Keras appends its own progress bar after the
    callbacks it is given, which would print each epoch's line in the middle
    of the bar's unfinished last line; passing the bar first makes Keras use
    it and finish it before ours prints."""
    return [keras.callbacks.ProgbarLogger(), EpochProgress(progress_log, config)]
