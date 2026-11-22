#!/usr/bin/env python
# coding: utf-8

"""HCRL Car-Hacking Dataset loader -- Pi1 (Hyundai YF Sonata).

Raw data is auto-downloaded from HCRL's Dropbox share. Each of the four
capture files (DoS, Fuzzy, gear, RPM) is time-ordered and contains both
normal and injected frames, labeled by the trailing R/T flag.

Rows are ``timestamp, CAN ID, DLC, <DLC data bytes>, R/T``, so a frame with
DLC < 8 is a shorter row; its missing data bytes are zero-padded. Only rows
whose field count doesn't match their DLC are dropped (none in the
published files).

Shared tail: every capture file is an attack session (300 injections of
3-5 s), then a time gap (22 s to 2.2 h), then the same 891,068 attack-free
frames (482 s, byte-identical in all four files). Used as is, that one
recording would be in the data four times and land in train for some files
and in test for others. So each file contributes only its attack session,
and the shared tail is used once, as its own attack-free capture. The tail
is checked against its known size, time span and hash.

``normal_run_data.txt`` (attack-free driving of the same car on another
day, shipped as ``normal_run_data.7z``) is a normal capture like the
shared tail: time blocks for train/test, and its train blocks feed the ID
period table (see ``common_cav``). All 27 of its IDs occur in the attack
captures, with matching periods (median ratio 1.00).

Split: each attack session and the shared tail are cut into 10 contiguous
time blocks. Blocks 2, 5 and 8 are test; the rest are train (see
``common_cav.time_block_layout``). Rows are then sampled as set in
``common_cav.Sampling`` (client training size, attack share, test normals).
"""

import hashlib
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAV_DIR = os.path.join(BASE_DIR, "Car-Hacking Dataset")
# New key whenever the rows, labels or features change: caches are found by
# key alone (see common_cav.find_any_cached_split), so an old one would be reused.
DATASET_KEY = "carhacking_h12t_s1_fs"
NORMAL_RUN_FILE = os.path.join("normal_run_data", "normal_run_data.txt")
SAMPLING = common_cav.Sampling()
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
# The attack-free recording appended to every capture file (see docstring).
SHARED_TAIL_ROWS = 891_068
SHARED_TAIL_SPAN = (1478200727.41947, 1478201209.058944)
SHARED_TAIL_SHA256 = "fcf24225a5a6287f2eccc2cb70d31837c9d0269921df19f9acb9b3ba222828b8"

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


def _capture_frames(path: str) -> common_cav.Frames:
    """Parse one time-ordered capture.

    The R/T flag is the row's last field, right after its DLC data bytes.
    Reading every row as 12 columns and dropping rows with empty cells would
    throw away every frame with DLC < 8 (about 1.2% of the rows, all normal).
    """
    df = pd.read_csv(path, header=None, names=range(12), dtype=str, low_memory=False)
    n_fields = df.notna().sum(axis=1).to_numpy()
    dlc = pd.to_numeric(df[2], errors="coerce").to_numpy()
    well_formed = ~np.isnan(dlc) & (dlc <= common_cav.NUM_DATA_BYTES) & (n_fields == dlc + 4)
    raw = df.to_numpy()[well_formed]
    del df
    n_fields = n_fields[well_formed]
    dlc = dlc[well_formed].astype(np.int64)

    data = np.zeros((len(raw), common_cav.NUM_DATA_BYTES), dtype=np.uint8)
    for j in range(common_cav.NUM_DATA_BYTES):
        has_byte = j < dlc
        data[has_byte, j] = [int(s, 16) for s in raw[has_byte, 3 + j]]

    flag = pd.Series(raw[np.arange(len(raw)), n_fields - 1])
    y = (flag.str.strip().str.upper() == "T").to_numpy()
    return common_cav.make_frames(
        raw[:, 0].astype(np.float64), [int(s, 16) for s in raw[:, 1]], dlc, data, y,
    )


def _split_shared_tail(frames: common_cav.Frames, name: str):
    """(attack session, shared tail) of one capture file; raises if the last
    rows are not exactly the known shared tail."""
    start = len(frames) - SHARED_TAIL_ROWS
    tail = frames.rows(slice(start, None))
    digest = hashlib.sha256()
    for column in (tail.ts, tail.can_id, tail.dlc, tail.data, tail.y):
        digest.update(np.ascontiguousarray(column).tobytes())
    if (
        start <= 0
        or (float(tail.ts[0]), float(tail.ts[-1])) != SHARED_TAIL_SPAN
        or frames.ts[start] - frames.ts[start - 1] <= common_cav.CAPTURE_GAP_S
        or digest.hexdigest() != SHARED_TAIL_SHA256
    ):
        raise ValueError(f"{name}: its last {SHARED_TAIL_ROWS} frames are not the known shared attack-free tail")
    return frames.rows(slice(0, start)), tail


def _session_frames(name: str) -> common_cav.Frames:
    return _split_shared_tail(_capture_frames(os.path.join(CAV_DIR, name)), name)[0]


def _shared_tail_frames() -> common_cav.Frames:
    name = REQUIRED_CAV_FILES[0]
    return _split_shared_tail(_capture_frames(os.path.join(CAV_DIR, name)), name)[1]


def _normal_run_frames(path: str) -> common_cav.Frames:
    ts, can_id, dlc, data, rtr = common_cav.parse_hcrl_txt(path)
    return common_cav.make_frames(ts, can_id, dlc, data, np.zeros(len(ts), dtype=np.int8), rtr=rtr)


def _normal_run_path() -> str:
    path = os.path.join(CAV_DIR, NORMAL_RUN_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' is missing. It ships inside the dataset as "
            f"normal_run_data.7z; extract it there (e.g. with 7-Zip)."
        )
    return path


def _captures() -> list:
    captures = [common_cav.Capture("normal_run_data", lambda: _normal_run_frames(_normal_run_path()),
                                   split="blocks", attack_free=True)]
    for name in REQUIRED_CAV_FILES:
        captures.append(common_cav.Capture(f"{name} (attack session)", lambda name=name: _session_frames(name),
                                           split="blocks"))
    captures.append(common_cav.Capture("shared attack-free tail", _shared_tail_frames, split="blocks", attack_free=True))
    return captures


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    """Leak-safe split: time-block train/test per capture file, scaled with
    the fixed scaling (common_cav.scale_features), and cached to split_cache/."""
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    _ensure_raw_cav_files()
    source_version = common_cav.directory_signature(
        [os.path.join(CAV_DIR, name) for name in REQUIRED_CAV_FILES] + [_normal_run_path()]
    )
    split = common_cav.build_feature_split(_captures(), SAMPLING, random_state)
    return common_cav.cache_split(
        DATASET_KEY, BASE_DIR, split, source_version, test_size, random_state,
        common_cav.period_table_arrays(split.tables),
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    """The client's training rows (already sampled to their final size when
    the cache was built), split into local train / validation."""
    cached = common_cav.load_cached(DATASET_KEY, BASE_DIR, ("x_train", "y_train"), test_size, random_state)
    if cached is None:
        x_train, _, y_train, _ = get_global_train_test_split(test_size, random_state)
    else:
        x_train, y_train = cached
    return common_cav.local_train_val_split(x_train, y_train, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
