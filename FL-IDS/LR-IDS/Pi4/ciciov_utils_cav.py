#!/usr/bin/env python
# coding: utf-8

"""CIC IoV Dataset 2024 (CICIoV2024) loader -- Pi4 (2019 Ford, CAN bus DoS
and spoofing attacks).

Raw data: the official download is behind a personal-info request form at
https://www.unb.ca/cic/datasets/iov-dataset-2024.html. The project owner's
Google Drive mirror of ``CICIoV2024.tar.xz`` is used for automatic download;
the archive contains binary/, decimal/ and hexadecimal/ representations and
only ``decimal_*.csv`` matches this pipeline's schema.

The files carry no timestamp and each file holds a single class, so the
split is by file where possible:
- ``decimal_DoS.csv`` (attack) is held out entirely as the test attack type.
- spoofing files (GAS/RPM/SPEED/STEERING_WHEEL) are training attacks.
- the single benign file is split by row: 33% of rows go to test.

Known limitation: the frames are highly repetitive (roughly 40 distinct
attack frames in total), so the evaluation on this dataset is weak and
should be reported as such.
"""

import glob
import os
import tarfile

import numpy as np
import pandas as pd

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CICIOV_DIR = os.path.join(BASE_DIR, "CICIoV2024")
DATASET_KEY = "ciciov_grp"
SOURCE_URL = "https://www.unb.ca/cic/datasets/iov-dataset-2024.html"
GOOGLE_DRIVE_FILE_ID = "1BTHtpfNhe0VN7dLTVhQz4-Q-A1szrQyD"

TEST_ATTACK_FILE = "decimal_DoS.csv"
BENIGN_TEST_FRACTION = 0.33

CANDIDATE_DLC_COLUMNS = ["DLC", "dlc"]
CANDIDATE_DATA_COLUMNS = [
    [f"DATA_{i}" for i in range(8)],
    [f"DATA{i}" for i in range(8)],
    [f"Data{i}" for i in range(8)],
    [f"data{i}" for i in range(8)],
]
CANDIDATE_LABEL_COLUMNS = ["label", "Label", "category", "Category", "specific_class"]
BENIGN_TOKENS = {"benign", "normal"}

reshape_for_cnn = common_cav.reshape_for_cnn


def _find_csv_files():
    return sorted(glob.glob(os.path.join(CICIOV_DIR, "**", "decimal_*.csv"), recursive=True))


def _find_tar_archives():
    return sorted(glob.glob(os.path.join(CICIOV_DIR, "**", "*.tar.xz"), recursive=True))


def _extract_tar_archives() -> None:
    for tar_path in _find_tar_archives():
        with tarfile.open(tar_path, "r:xz") as tf:
            tf.extractall(os.path.dirname(tar_path))
        os.remove(tar_path)


def _download_and_extract_ciciov_dataset() -> None:
    os.makedirs(CICIOV_DIR, exist_ok=True)
    tar_path = os.path.join(CICIOV_DIR, "CICIoV2024.tar.xz")
    common_cav.download_from_google_drive(GOOGLE_DRIVE_FILE_ID, tar_path)
    _extract_tar_archives()


def _ensure_raw_files() -> list:
    csv_files = _find_csv_files()
    if csv_files:
        return csv_files

    if _find_tar_archives():
        _extract_tar_archives()
        csv_files = _find_csv_files()

    if not csv_files:
        try:
            _download_and_extract_ciciov_dataset()
        except Exception as error:
            raise FileNotFoundError(
                f"No CICIoV2024 'decimal_*.csv' files found under '{CICIOV_DIR}', "
                f"and automatic download failed ({error}). Request and download "
                f"the dataset from {SOURCE_URL}, then place CICIoV2024.tar.xz "
                f"under '{CICIOV_DIR}' or the decimal/ CSVs there."
            ) from error
        csv_files = _find_csv_files()

    if not csv_files:
        raise FileNotFoundError(
            f"Downloaded/extracted CICIoV2024 but found no 'decimal_*.csv' "
            f"files under '{CICIOV_DIR}'. Inspect the contents."
        )
    return csv_files


def _resolve_data_columns(df: pd.DataFrame):
    for candidate_set in CANDIDATE_DATA_COLUMNS:
        if all(name in df.columns for name in candidate_set):
            return candidate_set
    raise ValueError(
        f"Could not find 8 DATA columns (tried {CANDIDATE_DATA_COLUMNS}) in "
        f"CICIoV2024 CSV columns: {list(df.columns)}"
    )


def _file_arrays(path: str):
    df = pd.read_csv(path)
    dlc_col = next((c for c in CANDIDATE_DLC_COLUMNS if c in df.columns), None)
    data_cols = _resolve_data_columns(df)
    label_col = next((c for c in CANDIDATE_LABEL_COLUMNS if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column in {path}: {list(df.columns)}")

    dlc = df[dlc_col].to_numpy(dtype=np.float32) if dlc_col else np.full(len(df), 8, np.float32)
    data = df[list(data_cols)].to_numpy(dtype=np.float32)
    x = np.column_stack([dlc, data]).astype(np.float32)

    label = df[label_col].astype(str).str.strip().str.lower()
    y = (~label.isin(BENIGN_TOKENS)).to_numpy().astype(np.int64)
    return x, y


KIND_BENIGN, KIND_SPOOF, KIND_DOS = 0, 1, 2


def _load_split(random_state: int):
    """Split by frame content. Identical feature rows (which are heavily
    repeated in this dataset) are kept together on one side, so no test
    frame value appears in training. DoS frames go to test, spoofing frames
    to train; a frame value shared between DoS and spoofing goes to test."""
    xs, ys, kinds = [], [], []
    for path in _ensure_raw_files():
        x, y = _file_arrays(path)
        name = os.path.basename(path)
        if name == TEST_ATTACK_FILE:
            kind = KIND_DOS
        elif y.sum() == 0:
            kind = KIND_BENIGN
        else:
            kind = KIND_SPOOF
        xs.append(x)
        ys.append(y)
        kinds.append(np.full(len(y), kind, dtype=np.int8))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    kind = np.concatenate(kinds)

    _, group = np.unique(x, axis=0, return_inverse=True)
    group = group.ravel()
    n_groups = int(group.max()) + 1
    has_dos = np.bincount(group, weights=(kind == KIND_DOS), minlength=n_groups) > 0
    has_spoof = np.bincount(group, weights=(kind == KIND_SPOOF), minlength=n_groups) > 0

    rng = np.random.default_rng(random_state)
    benign_group_to_test = rng.random(n_groups) < BENIGN_TEST_FRACTION
    group_to_test = has_dos | (~has_spoof & benign_group_to_test)
    is_test = group_to_test[group]

    return x[~is_test], x[is_test], y[~is_test], y[is_test]


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    csv_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(csv_files)
    x_train, x_test, y_train, y_test = _load_split(random_state)
    return common_cav.cache_precomputed_split(
        DATASET_KEY, BASE_DIR, x_train, x_test, y_train, y_test,
        source_version, test_size, random_state,
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    x_train_pool, _, y_train_pool, _ = get_global_train_test_split(test_size, random_state)
    return common_cav.local_train_val_split(x_train_pool, y_train_pool, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
