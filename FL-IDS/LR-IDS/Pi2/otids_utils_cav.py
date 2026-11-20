#!/usr/bin/env python
# coding: utf-8

"""HCRL CAN-intrusion-dataset (OTIDS) loader -- Pi2 (Kia Soul).

Raw data is auto-downloaded from HCRL's Dropbox share. Used captures:
attack-free (all normal) and DoS. Excluded:

- Fuzzy: its only label source is a time threshold (everything after
  ~250 s counted as attack), which is not per-frame ground truth, and it
  injects no ID the attack-free capture lacks, so ID can't label it either.
- Impersonation: labeling every 0x164 frame as attack isn't supported by
  the data. Apart from 5,697 frames with payload 01 02 .. 08, its 0x164
  frames match the genuine ECU's in payload, rate (107 vs 111 Hz) and
  remote-frame response time (median 0.258 vs 0.257 ms), so which frames
  came from the attacker can't be told.

Labels: DoS frames are the ones with arbitration ID 0x000 (HCRL's injection
ID), which never occurs in the attack-free capture. CAN ID is used only to
derive the label and to find each frame's previous same-ID frame, never as
a feature. Because the label is defined by the ID, the features must not
reveal "ID unseen in attack-free traffic" either (see ``common_cav``).

The attack-free capture's train blocks feed the ID period table.

Split: each capture is cut into 10 contiguous time blocks; blocks 2, 5 and 8
are test (see ``common_cav.time_block_layout``). The client trains on a
random 60% of the training pool (see CLIENT_TRAIN_FRACTION); the test split
is never sampled.
"""

import glob
import os
import shutil
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OTIDS_DIR = os.path.join(BASE_DIR, "OTIDS Dataset")
# New key whenever the rows, labels or features change: caches are found by
# key alone (see common_cav.find_any_cached_split), so an old one would be reused.
DATASET_KEY = "otids_dos_h12r"
SAMPLING = common_cav.RowSampling()
# Brings Pi2's ~2.5M-row pool to about the same training size as the other
# Pis, so it no longer has the slowest round or an outsized FedAvg weight.
CLIENT_TRAIN_FRACTION = 0.60
SOURCE_URL = "https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset"
DROPBOX_OTIDS_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/8kll7yvbgogkp0vahowvm/ADhDIC8LRFL8wHUexib3C3w"
    "?rlkey=8cp7scxgw25yt4wp8v2c2v8mp&st=rplc74rm&dl=1"
)

REQUIRED_FILES = {
    "normal": "Attack_free_dataset.txt",
    "dos": "DoS_attack_dataset.txt",
}

DOS_INJECTED_ID = 0x000

reshape_for_cnn = common_cav.reshape_for_cnn


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


def _capture_frames(path: str, is_dos: bool) -> common_cav.Frames:
    """Remote frames (frame type field ``100``, DLC 0, no data) are kept and
    marked. 9 IDs answer them about 0.25 ms later with an all-zero DLC-8 data
    frame, a millisecond or two after their own periodic frame; requests and
    replies get their own history streams (see ``common_cav.frame_streams``),
    otherwise those replies look exactly like the DoS frames."""
    ts, can_id, dlc, data, rtr = common_cav.parse_hcrl_txt(path)
    can_id = np.asarray(can_id)
    y = (can_id == DOS_INJECTED_ID) if is_dos else np.zeros(len(can_id), dtype=bool)
    return common_cav.make_frames(ts, can_id, dlc, data, y, rtr=rtr)


def _captures() -> list:
    normal = os.path.join(OTIDS_DIR, REQUIRED_FILES["normal"])
    dos = os.path.join(OTIDS_DIR, REQUIRED_FILES["dos"])
    return [
        common_cav.Capture("attack_free", lambda: _capture_frames(normal, False), split="blocks", attack_free=True),
        common_cav.Capture("dos", lambda: _capture_frames(dos, True), split="blocks"),
    ]


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    _ensure_raw_files()
    source_version = common_cav.directory_signature(
        os.path.join(OTIDS_DIR, name) for name in REQUIRED_FILES.values()
    )
    x_train, x_test, y_train, y_test, tables = common_cav.build_feature_split(_captures(), SAMPLING, random_state)
    return common_cav.cache_precomputed_split(
        DATASET_KEY, BASE_DIR, x_train, x_test, y_train, y_test,
        source_version, test_size, random_state, common_cav.period_table_arrays(tables),
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
