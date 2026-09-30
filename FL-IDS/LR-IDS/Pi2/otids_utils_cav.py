#!/usr/bin/env python
# coding: utf-8

"""HCRL CAN-intrusion-dataset (OTIDS) loader — Pi2.

Raw data is auto-downloaded from HCRL's own Dropbox share for this dataset
(the "Download" link on the dataset page below resolves to this folder,
confirmed by fetching that page directly rather than guessing a URL):
    https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
If the automatic download fails (e.g. no network on the Pi, or HCRL rotates
the share link), download manually from that page and place the extracted
.txt capture files under ``OTIDS Dataset/`` next to this file (see
REQUIRED_FILES below for expected filenames -- if the archive's actual
filenames differ, update REQUIRED_FILES to match).

Row-level attack labeling caveat: HCRL does not publish per-row ground truth
for the Fuzzy/Impersonation captures. DoS and Impersonation are labeled here
using HCRL's documented fixed injection arbitration IDs (0x000 and 0x164
respectively); Fuzzy uses random IDs with no fixed pattern, so it falls back
to HCRL's documented coarse timing convention (attack-free for the first
250s of the capture, mixed afterwards). TODO: verify all three heuristics
against the actual downloaded files and refine if a more precise per-row
label source turns out to be available.
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
DATASET_KEY = "otids"
SOURCE_URL = "https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset"
DROPBOX_OTIDS_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/8kll7yvbgogkp0vahowvm/ADhDIC8LRFL8wHUexib3C3w"
    "?rlkey=8cp7scxgw25yt4wp8v2c2v8mp&st=rplc74rm&dl=1"
)

REQUIRED_FILES = {
    "normal": "Attack_free_dataset.txt",
    "dos": "DoS_attack_dataset.txt",
    "fuzzy": "Fuzzy_attack_dataset.txt",
    "impersonation": "Impersonation_attack_dataset.txt",
}

# Fixed arbitration IDs HCRL uses to inject DoS / Impersonation frames.
DOS_INJECTED_ID = 0x000
IMPERSONATION_INJECTED_ID = 0x164
FUZZY_ATTACK_FREE_SECONDS = 250.0

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
            f"{still_missing}. The downloaded archive's actual filenames may "
            f"differ from REQUIRED_FILES; inspect '{OTIDS_DIR}' and update "
            f"REQUIRED_FILES in this file to match, or place the files "
            f"manually with the expected names."
        )


def _parse_otids_file(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _LINE_RE.search(line)
            if not match:
                continue
            timestamp = float(match.group(1))
            can_id = int(match.group(2), 16)
            dlc = int(match.group(3))
            data_bytes = [int(b, 16) for b in match.group(4).split()]
            rows.append((timestamp, can_id, dlc, data_bytes))
    return rows


def _load_raw_otids():
    _ensure_raw_files()

    # GrowableArray fills numpy buffers directly instead of accumulating
    # Python lists first, keeping peak memory closer to the final arrays'
    # size (important on a Raspberry Pi's limited RAM).
    features = common_cav.GrowableArray(num_cols=10, dtype=np.float32)
    labels = common_cav.GrowableArray(num_cols=1, dtype=np.int64)

    def _add(can_id, dlc, data_bytes, label_str):
        features.append(common_cav.build_frame_features(can_id, dlc, data_bytes))
        labels.append((common_cav.LABEL_TO_INT[label_str],))

    for _, can_id, dlc, data_bytes in _parse_otids_file(os.path.join(OTIDS_DIR, REQUIRED_FILES["normal"])):
        _add(can_id, dlc, data_bytes, "Normal")

    for _, can_id, dlc, data_bytes in _parse_otids_file(os.path.join(OTIDS_DIR, REQUIRED_FILES["dos"])):
        _add(can_id, dlc, data_bytes, "ATTACK" if can_id == DOS_INJECTED_ID else "Normal")

    for _, can_id, dlc, data_bytes in _parse_otids_file(os.path.join(OTIDS_DIR, REQUIRED_FILES["impersonation"])):
        _add(can_id, dlc, data_bytes, "ATTACK" if can_id == IMPERSONATION_INJECTED_ID else "Normal")

    fuzzy_rows = _parse_otids_file(os.path.join(OTIDS_DIR, REQUIRED_FILES["fuzzy"]))
    if fuzzy_rows:
        t0 = fuzzy_rows[0][0]
        for timestamp, can_id, dlc, data_bytes in fuzzy_rows:
            is_attack = (timestamp - t0) >= FUZZY_ATTACK_FREE_SECONDS
            _add(can_id, dlc, data_bytes, "ATTACK" if is_attack else "Normal")

    return features.finalize(), labels.finalize().ravel()


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    _ensure_raw_files()
    source_version = common_cav.directory_signature(
        os.path.join(OTIDS_DIR, name) for name in REQUIRED_FILES.values()
    )
    return common_cav.cache_scaled_split(
        DATASET_KEY, _load_raw_otids, source_version, BASE_DIR,
        test_size=test_size, random_state=random_state,
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    x_train_pool, _, y_train_pool, _ = get_global_train_test_split(test_size, random_state)
    return common_cav.local_train_val_split(x_train_pool, y_train_pool, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
