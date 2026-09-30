#!/usr/bin/env python
# coding: utf-8

"""CIC IoV Dataset 2024 (CICIoV2024) loader — Pi4 (2019 Ford, CAN bus DoS
and spoofing attacks).

UNB's Canadian Institute for Cybersecurity gates the original download
behind a personal-info request form (name/email/organization/job
title/country submitted to https://cicresearch.ca/IOTDataset/CICIoV2024/),
confirmed by fetching that page directly -- there is no stable public URL
to script around that, so a human has to submit the form once:
    https://www.unb.ca/cic/datasets/iov-dataset-2024.html
The form hands back a ``CICIoV2024.tar.xz`` download. To auto-download on
every Pi/run after that one-time step, this loader instead pulls the exact
same file from the project owner's own Google Drive mirror (uploaded after
completing the form; shared as "anyone with the link", confirmed to serve
the real file directly -- this only works for this project's own Pi4 setup,
since it's this owner's private mirror, not CIC's). If that download fails
(e.g. no network, or the mirror is removed), this loader falls back to
looking for a manually-placed file: drop ``CICIoV2024.tar.xz`` (unmodified)
directly under ``CICIoV2024/`` next to this module and it will be extracted
automatically (Python's stdlib ``tarfile`` reads ``.tar.xz`` natively);
extracting it yourself first and placing the ``decimal/`` folder's CSVs
there also works.

Confirmed against the real downloaded archive: it contains three parallel
representations of the same captures -- ``binary/`` (one column per bit,
incompatible with this pipeline's schema), ``hexadecimal/`` (hex-string
``ID``/``DLC`` fields), and ``decimal/`` (plain integer ``ID`` and
``DATA_0``..``DATA_7`` columns plus a ``label`` column of "BENIGN"/"ATTACK").
Only ``decimal/`` matches the feature schema this pipeline expects, so this
loader deliberately restricts itself to files named ``decimal_*.csv``
(matching the archive's real naming, e.g. ``decimal_benign.csv``,
``decimal_DoS.csv``, ``decimal_spoofing-GAS.csv``) rather than globbing
every ``*.csv`` -- picking up a ``binary``/``hexadecimal`` file by accident
would silently corrupt the feature values.
"""

import glob
import os
import tarfile

import numpy as np
import pandas as pd

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CICIOV_DIR = os.path.join(BASE_DIR, "CICIoV2024")
DATASET_KEY = "ciciov"
SOURCE_URL = "https://www.unb.ca/cic/datasets/iov-dataset-2024.html"
GOOGLE_DRIVE_FILE_ID = "1BTHtpfNhe0VN7dLTVhQz4-Q-A1szrQyD"

CANDIDATE_ID_COLUMNS = ["ID", "id", "can_id", "CAN_ID", "CAN ID"]
CANDIDATE_DLC_COLUMNS = ["DLC", "dlc"]
CANDIDATE_DATA_COLUMNS = [
    [f"DATA_{i}" for i in range(8)],
    [f"DATA{i}" for i in range(8)],
    [f"Data{i}" for i in range(8)],
    [f"data{i}" for i in range(8)],
]
CANDIDATE_LABEL_COLUMNS = ["label", "Label", "category", "Category", "specific_class"]
BENIGN_TOKENS = {"benign", "normal"}


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


reshape_for_cnn = common_cav.reshape_for_cnn


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
                f"No CICIoV2024 'decimal_*.csv' files found under "
                f"'{CICIOV_DIR}', and automatic download failed ({error}). "
                f"Request and download the dataset from {SOURCE_URL}, then "
                f"either place the resulting CICIoV2024.tar.xz directly "
                f"under '{CICIOV_DIR}' (it will be auto-extracted) or "
                f"extract it yourself and place the 'decimal/' folder's "
                f"CSVs there."
            ) from error
        csv_files = _find_csv_files()

    if not csv_files:
        raise FileNotFoundError(
            f"Downloaded/extracted CICIoV2024 but found no 'decimal_*.csv' "
            f"files under '{CICIOV_DIR}' afterwards. Inspect the contents "
            f"and adjust this loader if the layout differs."
        )
    return csv_files


def _resolve_single_column(df: pd.DataFrame, candidates, required: bool = True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(
            f"Could not find any of {candidates} in CICIoV2024 CSV columns: {list(df.columns)}"
        )
    return None


def _resolve_data_columns(df: pd.DataFrame):
    for candidate_set in CANDIDATE_DATA_COLUMNS:
        if all(name in df.columns for name in candidate_set):
            return candidate_set
    raise ValueError(
        f"Could not find 8 DATA columns (tried {CANDIDATE_DATA_COLUMNS}) in "
        f"CICIoV2024 CSV columns: {list(df.columns)}"
    )


def _read_frame_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    id_col = _resolve_single_column(df, CANDIDATE_ID_COLUMNS)
    dlc_col = _resolve_single_column(df, CANDIDATE_DLC_COLUMNS, required=False)
    data_cols = _resolve_data_columns(df)
    label_col = _resolve_single_column(df, CANDIDATE_LABEL_COLUMNS)

    out = pd.DataFrame()
    out["CAN ID"] = df[id_col]
    out["Byte"] = df[dlc_col] if dlc_col is not None else 8
    for i, col in enumerate(data_cols):
        out[f"DATA[{i}]"] = df[col]

    label_lower = df[label_col].astype(str).str.strip().str.lower()
    out["AttackType"] = np.where(label_lower.isin(BENIGN_TOKENS), "Normal", "ATTACK")
    return out.dropna()


def _load_raw_ciciov():
    csv_files = _ensure_raw_files()

    frames = [_read_frame_csv(path) for path in csv_files]
    dataset = pd.concat(frames, ignore_index=True)

    feature_cols = ["CAN ID", "Byte"] + [f"DATA[{i}]" for i in range(8)]
    x = dataset[feature_cols].to_numpy(dtype=np.float32)
    y = dataset["AttackType"].map(common_cav.LABEL_TO_INT).to_numpy(dtype=np.int64)
    return x, y


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    csv_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(csv_files)
    return common_cav.cache_scaled_split(
        DATASET_KEY, _load_raw_ciciov, source_version, BASE_DIR,
        test_size=test_size, random_state=random_state,
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    x_train_pool, _, y_train_pool, _ = get_global_train_test_split(test_size, random_state)
    return common_cav.local_train_val_split(x_train_pool, y_train_pool, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
