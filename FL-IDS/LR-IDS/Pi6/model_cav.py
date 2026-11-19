#!/usr/bin/env python
# coding: utf-8

import datetime
import math
import time

import numpy as np
from tensorflow import keras


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
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    return model


class BatchSequence(keras.utils.PyDataset):
    """Feeds Keras one batch at a time instead of whole arrays.

    Given whole NumPy arrays, fit/evaluate/predict keep several internal
    copies of the full array -- too much for a 1GB Pi. ``class_weight``
    becomes per-row sample weights, which is what Keras itself turns it into
    for array input.
    """

    def __init__(self, x, y=None, batch_size=32, shuffle=False, class_weight=None):
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
