#!/usr/bin/env python
# coding: utf-8

"""ROAD (Real ORNL Automotive Dynamometer) Dataset loader — Pi5 (real,
non-simulated fuzzing/fabrication/masquerade attacks).

Raw data is auto-downloaded from Zenodo record 10462796 via Zenodo's public
REST API (https://zenodo.org/api/records/10462796), which is queried at
runtime for the record's real file list rather than guessing a zip URL. If
the automatic download fails (e.g. no network on the Pi), download manually
from https://zenodo.org/records/10462796 and place the extracted files
anywhere under ``ROAD Dataset/`` next to this file.

File classification: ROAD's published capture files follow the dataset's
own naming convention -- ambient (attack-free) captures are named
``ambient_*.log``. Every ``.log`` found recursively under ``ROAD Dataset/``
that isn't an ambient capture is looked up in the dataset's own
``attacks/capture_metadata.json`` (confirmed against the real downloaded
archive; every ``capture_metadata.json`` found anywhere under the folder is
merged into one lookup, keyed by capture name = log filename without
extension).

Row-level labeling (confirmed against the real archive and its own
readme.md): each attack capture's metadata entry carries an
``injection_interval: [start, end]`` in seconds elapsed since that
capture's own first frame -- rows inside that interval are labeled Attack,
everything else Normal. ROAD's readme documents that the "accelerator
attack" captures have ``injection_interval: null`` because no messages were
actually injected (the capture just records the vehicle's anomalous
behavior with no CAN-level ground truth to flag), so those are labeled
entirely Normal, matching the paper's own convention. Only a capture with
no metadata entry at all falls back to a coarse whole-file Attack label.
"""

import glob
import json
import os
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROAD_DIR = os.path.join(BASE_DIR, "ROAD Dataset")
DATASET_KEY = "road_blk"
# Fraction of the ~19M-row training pool this client actually trains on.
# Sized so the client fits in a 1GB Raspberry Pi; the test split is unchanged.
CLIENT_TRAIN_FRACTION = 0.10
SOURCE_URL = "https://zenodo.org/records/10462796"
ZENODO_RECORD_API = "https://zenodo.org/api/records/10462796"

reshape_for_cnn = common_cav.reshape_for_cnn


def _find_log_files():
    return sorted(glob.glob(os.path.join(ROAD_DIR, "**", "*.log"), recursive=True))


def _download_road_dataset() -> None:
    os.makedirs(ROAD_DIR, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD_API) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    files = record.get("files") or []
    if not files:
        raise RuntimeError(f"Zenodo record returned no downloadable files: {ZENODO_RECORD_API}")

    for entry in files:
        filename = entry["key"]
        download_url = entry["links"]["self"]
        target_path = os.path.join(ROAD_DIR, filename)
        if os.path.exists(target_path):
            continue
        urllib.request.urlretrieve(download_url, target_path)
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(target_path, "r") as zf:
                zf.extractall(ROAD_DIR)


def _ensure_raw_files() -> list:
    log_files = _find_log_files()
    if log_files:
        return log_files

    try:
        _download_road_dataset()
    except Exception as error:
        raise FileNotFoundError(
            f"No ROAD .log files found under '{ROAD_DIR}', and automatic "
            f"download from Zenodo failed ({error}). Download manually from "
            f"{SOURCE_URL} and place the extracted captures under '{ROAD_DIR}'."
        ) from error

    log_files = _find_log_files()
    if not log_files:
        raise FileNotFoundError(
            f"Downloaded the ROAD Zenodo record but found no .log capture "
            f"files under '{ROAD_DIR}' afterwards. Inspect the downloaded "
            f"archive contents and adjust this loader if the layout differs."
        )
    return log_files


def _is_ambient(path: str) -> bool:
    return "ambient" in os.path.basename(path).lower()


def _find_metadata_files():
    return sorted(glob.glob(os.path.join(ROAD_DIR, "**", "capture_metadata.json"), recursive=True))


def _load_all_capture_metadata() -> dict:
    """Merge every ``capture_metadata.json`` found under ROAD_DIR (one per
    ``ambient/`` and ``attacks/`` subfolder in the official archive) into a
    single lookup keyed by capture name.
    """
    metadata = {}
    for path in _find_metadata_files():
        with open(path, "r", encoding="utf-8") as f:
            metadata.update(json.load(f))
    return metadata


def _capture_arrays(path: str, capture_metadata: dict):
    """Features and labels for one capture, in time order."""
    rows = common_cav.parse_candump_file(path)
    x = np.asarray(
        [common_cav.build_frame_features(dlc, data) for _, _, dlc, data in rows],
        dtype=np.float32,
    )
    if _is_ambient(path):
        y = np.zeros(len(rows), dtype=np.int64)
        return x, y

    capture_name = os.path.splitext(os.path.basename(path))[0]
    entry = capture_metadata.get(capture_name)
    interval = entry.get("injection_interval") if entry else None
    if entry is None:
        # No metadata entry at all: whole-file fallback.
        y = np.ones(len(rows), dtype=np.int64)
    elif interval is None:
        # Metadata entry with null interval: accelerator-attack captures,
        # which have no injected messages (ROAD readme).
        y = np.zeros(len(rows), dtype=np.int64)
    else:
        t0 = rows[0][0] if rows else 0.0
        t = np.array([r[0] - t0 for r in rows])
        y = ((t >= interval[0]) & (t <= interval[1])).astype(np.int64)
    return x, y


def _load_block_split():
    capture_metadata = _load_all_capture_metadata()
    # Masquerade variants are near-copies of their original capture; keeping
    # both would let the same frames appear on both sides of the split.
    log_files = [p for p in _ensure_raw_files() if "_masquerade" not in os.path.basename(p)]

    parts = []
    for path in log_files:
        x, y = _capture_arrays(path, capture_metadata)
        train_idx, test_idx = common_cav.split_time_blocks(len(y))
        parts.append((x[train_idx], y[train_idx], x[test_idx], y[test_idx]))

    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test = np.concatenate([p[2] for p in parts])
    y_test = np.concatenate([p[3] for p in parts])
    return x_train, x_test, y_train, y_test


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        # A split_cache/*.npz already exists -- use it without ever
        # touching the (3GB) raw dataset, e.g. on a disk-constrained device
        # that only has the cache copied onto it.
        return cached

    log_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(log_files + _find_metadata_files())
    x_train, x_test, y_train, y_test = _load_block_split()
    return common_cav.cache_precomputed_split(
        DATASET_KEY, BASE_DIR, x_train, x_test, y_train, y_test,
        source_version, test_size, random_state,
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    pool = common_cav.load_cached_train_fraction(
        DATASET_KEY, BASE_DIR, CLIENT_TRAIN_FRACTION, test_size, random_state,
    )
    if pool is None:
        x_full, _, y_full, _ = get_global_train_test_split(test_size, random_state)
        pool = common_cav.subsample_rows(x_full, y_full, CLIENT_TRAIN_FRACTION, random_state)
    x_train_pool, y_train_pool = pool
    return common_cav.local_train_val_split(x_train_pool, y_train_pool, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
