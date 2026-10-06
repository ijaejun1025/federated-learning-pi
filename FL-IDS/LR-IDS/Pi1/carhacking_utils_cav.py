#!/usr/bin/env python
# coding: utf-8

"""HCRL Car-Hacking Dataset loader -- Pi1 (Hyundai YF Sonata).

Raw data is auto-downloaded from HCRL's Dropbox share. Each of the four
capture files (DoS, Fuzzy, gear, RPM) is time-ordered and contains both
normal and injected frames, labeled by the trailing R/T flag.

Split: each file is cut into 10 contiguous time blocks. Blocks 2, 5 and 8
are test; the rest are train (see ``common_cav.split_time_blocks``). A
the full blocks are used for test; each client trains on a random 15%
of the training pool (see CLIENT_TRAIN_FRACTION). Rows with missing fields (corrupt lines) are dropped.
"""

import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAV_DIR = os.path.join(BASE_DIR, "Car-Hacking Dataset")
DATASET_KEY = "carhacking_full"
DROPBOX_CAV_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/AF7JeRW893grZkigkulkAHk"
    "?rlkey=3h6zamu3kc262lrnipu5qden8&dl=1"
)
REQUIRED_CAV_FILES = [
    "DoS_dataset.csv",
    "Fuzzy_dataset.csv",
    "gear_dataset.csv",
    "RPM_dataset.csv",
]
# Fraction of the training pool each client trains on (test split is never sampled).
CLIENT_TRAIN_FRACTION = 0.15

reshape_for_cnn = common_cav.reshape_for_cnn


def _find_file_recursively(root_dir: str, file_name: str):
    import glob

    matches = glob.glob(os.path.join(root_dir, "**", file_name), recursive=True)
    return matches[0] if matches else None


def _download_and_extract_cav_dataset() -> None:
    os.makedirs(CAV_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "cav_dataset_download.zip")

    urllib.request.urlretrieve(DROPBOX_CAV_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(CAV_DIR)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    for file_name in REQUIRED_CAV_FILES:
        target_path = os.path.join(CAV_DIR, file_name)
        if not os.path.exists(target_path):
            located = _find_file_recursively(CAV_DIR, file_name)
            if located:
                shutil.copy2(located, target_path)


def _ensure_raw_cav_files() -> None:
    missing = [
        name for name in REQUIRED_CAV_FILES
        if not os.path.exists(os.path.join(CAV_DIR, name))
    ]
    if not missing:
        return

    try:
        _download_and_extract_cav_dataset()
    except Exception as error:
        raise FileNotFoundError(
            f"Car-Hacking raw CSV files were not found, and automatic download "
            f"also failed ({error}). Required: {REQUIRED_CAV_FILES}. Place them "
            f"under '{CAV_DIR}'."
        ) from error

    still_missing = [
        name for name in REQUIRED_CAV_FILES
        if not os.path.exists(os.path.join(CAV_DIR, name))
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Still missing after automatic download: {still_missing}. "
            f"Place them under '{CAV_DIR}'."
        )


def _capture_arrays(path: str):
    """Parse one time-ordered capture into (features, labels) in file order."""
    df = pd.read_csv(path, header=None, names=range(12), low_memory=False).dropna()
    dlc = df[2].astype(np.int64).to_numpy()
    data = np.column_stack(
        [df[c].map(lambda s: int(str(s), 16)).to_numpy(dtype=np.int64) for c in range(3, 11)]
    )
    x = np.column_stack([dlc, data]).astype(np.float32)
    y = (df[11].astype(str).str.strip().str.upper() == "T").to_numpy().astype(np.int64)
    return x, y


def _capture_split(path: str):
    x, y = _capture_arrays(path)
    train_idx, test_idx = common_cav.split_time_blocks(len(y))
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def _load_block_split(random_state: int):
    parts = [
        _capture_split(os.path.join(CAV_DIR, name))
        for name in REQUIRED_CAV_FILES
    ]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test = np.concatenate([p[2] for p in parts])
    y_test = np.concatenate([p[3] for p in parts])
    return x_train, x_test, y_train, y_test


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    """Leak-safe split: time-block train/test per capture file, scaled with a
    scaler fit on train only, and cached to split_cache/."""
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    _ensure_raw_cav_files()
    source_version = common_cav.directory_signature(
        [os.path.join(CAV_DIR, name) for name in REQUIRED_CAV_FILES]
    )
    x_train, x_test, y_train, y_test = _load_block_split(random_state)
    return common_cav.cache_precomputed_split(
        DATASET_KEY, BASE_DIR, x_train, x_test, y_train, y_test,
        source_version, test_size, random_state,
    )


def get_client_data(
    test_size: float = 0.33,
    random_state: int = 41,
    local_val_size: float = 0.2,
):
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
