#!/usr/bin/env python
# coding: utf-8

"""HCRL CAN-intrusion-dataset (OTIDS) loader -- Pi2 (Kia Soul).

Raw data is auto-downloaded from HCRL's Dropbox share. Used captures:
attack-free (all normal), DoS and Impersonation. The Fuzzy capture is
excluded: its only label source is a time threshold (everything after
~250 s counted as attack), which is not per-frame ground truth.

Labels: DoS frames are the ones with arbitration ID 0x000, Impersonation
frames the ones with ID 0x164 (HCRL's injection IDs). CAN ID is used only
to derive the label, never as a feature.

Split: each capture is cut into 10 contiguous time blocks; blocks 2, 5 and 8
are test (see ``common_cav.split_time_blocks``).
"""

import glob
import os
import re
import shutil
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OTIDS_DIR = os.path.join(BASE_DIR, "OTIDS Dataset")
DATASET_KEY = "otids_blk"
SOURCE_URL = "https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset"
DROPBOX_OTIDS_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/8kll7yvbgogkp0vahowvm/ADhDIC8LRFL8wHUexib3C3w"
    "?rlkey=8cp7scxgw25yt4wp8v2c2v8mp&st=rplc74rm&dl=1"
)

REQUIRED_FILES = {
    "normal": "Attack_free_dataset.txt",
    "dos": "DoS_attack_dataset.txt",
    "impersonation": "Impersonation_attack_dataset.txt",
}

DOS_INJECTED_ID = 0x000
IMPERSONATION_INJECTED_ID = 0x164

reshape_for_cnn = common_cav.reshape_for_cnn

_LINE_RE = re.compile(
    r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*(\d+)\s+((?:[0-9a-fA-F]{2}\s*)*)"
)


def _find_file_recursively(file_name: str):
    matches = glob.glob(os.path.join(OTIDS_DIR, "**", file_name), recursive=True)
    return matches[0] if matches else None


def _download_and_extract_otids_dataset() -> None:
    os.makedirs(OTIDS_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "otids_dataset_download.zip")

    urllib.request.urlretrieve(DROPBOX_OTIDS_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OTIDS_DIR)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    for file_name in REQUIRED_FILES.values():
        target_path = os.path.join(OTIDS_DIR, file_name)
        if not os.path.exists(target_path):
            located = _find_file_recursively(file_name)
            if located:
                shutil.copy2(located, target_path)


def _ensure_raw_files() -> None:
    missing = [
        name for name in REQUIRED_FILES.values()
        if not os.path.exists(os.path.join(OTIDS_DIR, name))
    ]
    if not missing:
        return

    try:
        _download_and_extract_otids_dataset()
    except Exception as error:
        raise FileNotFoundError(
            "OTIDS raw files were not found, and automatic download also "
            f"failed. Required files: {list(REQUIRED_FILES.values())}, "
            f"error: {error}. Download manually from {SOURCE_URL} and place "
            f"the extracted .txt capture files under '{OTIDS_DIR}'."
        ) from error

    still_missing = [
        name for name in REQUIRED_FILES.values()
        if not os.path.exists(os.path.join(OTIDS_DIR, name))
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Some OTIDS files are still missing after automatic download: "
            f"{still_missing}. Place them under '{OTIDS_DIR}' with these names."
        )


def _parse_otids_file(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _LINE_RE.search(line)
            if not match:
                continue
            can_id = int(match.group(2), 16)
            dlc = int(match.group(3))
            data_bytes = [int(b, 16) for b in match.group(4).split()]
            rows.append((can_id, dlc, data_bytes))
    return rows


def _file_arrays(path: str, label_fn):
    rows = _parse_otids_file(path)
    x = np.asarray(
        [common_cav.build_frame_features(dlc, data) for _, dlc, data in rows],
        dtype=np.float32,
    )
    y = np.asarray([label_fn(can_id) for can_id, _, _ in rows], dtype=np.int64)
    return x, y


def _load_block_split():
    sources = [
        (REQUIRED_FILES["normal"], lambda _: 0),
        (REQUIRED_FILES["dos"], lambda cid: int(cid == DOS_INJECTED_ID)),
        (REQUIRED_FILES["impersonation"], lambda cid: int(cid == IMPERSONATION_INJECTED_ID)),
    ]
    parts = []
    for name, label_fn in sources:
        x, y = _file_arrays(os.path.join(OTIDS_DIR, name), label_fn)
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
        return cached

    _ensure_raw_files()
    source_version = common_cav.directory_signature(
        os.path.join(OTIDS_DIR, name) for name in REQUIRED_FILES.values()
    )
    x_train, x_test, y_train, y_test = _load_block_split()
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
