#!/usr/bin/env python
# coding: utf-8

"""HCRL Survival Analysis Dataset loader — Pi3 (Chevrolet Spark / Hyundai
Sonata / Kia Soul, Flooding/Fuzzy/Malfunction attacks).

Raw data is auto-downloaded from HCRL's own Dropbox share for this dataset
(the real "Download" link found by fetching the dataset page directly
rather than guessing a URL):
    https://ocslab.hksecurity.net/Datasets/survival-ids
The Dropbox folder download itself unwraps to a *password-protected* inner
``survival.zip`` -- the password (``ai.spera!+``, published openly on the
dataset page) is applied automatically via Python's stdlib ``zipfile``
(this archive uses classic ZipCrypto encryption, which ``zipfile`` supports
natively, confirmed against the real downloaded file). If the automatic
download fails, download manually from that page and place the extracted
.txt capture files anywhere under ``Survival Analysis Dataset/`` next to
this file.

Real file/column layout (confirmed against the actual downloaded archive,
which differs from what an initial best-guess assumed): captures are
headerless, comma-separated ``.txt`` files (not fixed-width .csv) named
``Flooding_dataset_<VEHICLE>.txt``, ``Fuzzy_dataset_<VEHICLE>.txt``,
``Malfunction<id>_dataset_<VEHICLE>.txt``, and
``FreeDrivingData_<date>_<VEHICLE>.txt``, one set per vehicle subfolder
(Sonata/Soul/Spark). Each row is:
    timestamp, CAN_ID(hex), DLC, <DLC data bytes as hex>, [R|T]
DLC is variable per row, so the row width varies -- the R/T flag (R=normal,
T=attack, confirmed present on every row of the Flooding/Fuzzy/Malfunction
files) is always the last field. ``FreeDrivingData`` files carry no flag at
all (verified empirically: field count is exactly 3 + DLC with nothing
after the data bytes) because the whole file is normal driving.
"""

import glob
import os
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SURVIVAL_DIR = os.path.join(BASE_DIR, "Survival Analysis Dataset")
DATASET_KEY = "survival"
SOURCE_URL = "https://ocslab.hksecurity.net/Datasets/survival-ids"
DROPBOX_SURVIVAL_ZIP_URL = (
    "https://www.dropbox.com/scl/fo/mqsp1orzaqjwu61j69amr/ANiI9XBqJHH5kLDUbrjrFw4"
    "?rlkey=87nphryu8yz0zz9atd436wx36&st=9cqxwvj6&dl=1"
)
SURVIVAL_ZIP_PASSWORD = b"ai.spera!+"

reshape_for_cnn = common_cav.reshape_for_cnn


def _find_txt_files():
    return sorted(glob.glob(os.path.join(SURVIVAL_DIR, "**", "*.txt"), recursive=True))


def _find_file_recursively(file_name: str):
    matches = glob.glob(os.path.join(SURVIVAL_DIR, "**", file_name), recursive=True)
    return matches[0] if matches else None


def _download_and_extract_survival_dataset() -> None:
    os.makedirs(SURVIVAL_DIR, exist_ok=True)
    outer_zip_path = os.path.join(BASE_DIR, "survival_dataset_download.zip")
    urllib.request.urlretrieve(DROPBOX_SURVIVAL_ZIP_URL, outer_zip_path)

    # The Dropbox folder download wraps a single password-protected
    # "survival.zip" inside an outer, unencrypted zip.
    with zipfile.ZipFile(outer_zip_path, "r") as outer_zip:
        outer_zip.extractall(SURVIVAL_DIR)
    os.remove(outer_zip_path)

    inner_zip_path = _find_file_recursively("survival.zip")
    if inner_zip_path:
        with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
            inner_zip.extractall(SURVIVAL_DIR, pwd=SURVIVAL_ZIP_PASSWORD)
        os.remove(inner_zip_path)


def _ensure_raw_files() -> list:
    txt_files = _find_txt_files()
    if txt_files:
        return txt_files

    try:
        _download_and_extract_survival_dataset()
    except Exception as error:
        raise FileNotFoundError(
            f"No Survival Analysis .txt files found under '{SURVIVAL_DIR}', "
            f"and automatic download failed ({error}). Download manually "
            f"from {SOURCE_URL} (Dropbox folder password: "
            f"{SURVIVAL_ZIP_PASSWORD.decode()}) and place the extracted "
            f".txt files under '{SURVIVAL_DIR}'."
        ) from error

    txt_files = _find_txt_files()
    if not txt_files:
        raise FileNotFoundError(
            f"Downloaded the Survival Analysis archive but found no .txt "
            f"capture files under '{SURVIVAL_DIR}' afterwards. Inspect the "
            f"downloaded contents and adjust this loader if the layout "
            f"differs."
        )
    return txt_files


def _parse_survival_file(path: str):
    """Two raw row formats exist in this dataset (confirmed against the real
    files): most captures are fully comma-separated
    (``ts,ID,DLC,B0,B1,...,B(DLC-1)[,R|T]``), but the Soul/Spark
    ``FreeDrivingData`` files instead pack the data bytes space-separated
    into a single trailing field (``ts,ID,DLC,"B0 B1 ... B(DLC-1)"``), with
    no R/T flag at all. Both are handled here by field count rather than by
    filename, since that's what actually distinguishes them.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            can_id = int(parts[1], 16)
            dlc = int(parts[2])
            rest = parts[3:]

            if len(rest) == 1 and " " in rest[0].strip():
                data_bytes = [int(b, 16) for b in rest[0].split()]
                flag = None
            else:
                data_bytes = [int(b, 16) for b in rest[:dlc]]
                flag = rest[dlc].strip().upper() if len(rest) > dlc else None

            label = "ATTACK" if flag == "T" else "Normal"
            rows.append((can_id, dlc, data_bytes, label))
    return rows


def _load_raw_survival():
    txt_files = _ensure_raw_files()

    # GrowableArray fills numpy buffers directly instead of accumulating
    # Python lists first, keeping peak memory closer to the final arrays'
    # size (important on a Raspberry Pi's limited RAM).
    features = common_cav.GrowableArray(num_cols=10, dtype=np.float32)
    labels = common_cav.GrowableArray(num_cols=1, dtype=np.int64)

    for path in txt_files:
        for can_id, dlc, data_bytes, label in _parse_survival_file(path):
            features.append(common_cav.build_frame_features(can_id, dlc, data_bytes))
            labels.append((common_cav.LABEL_TO_INT[label],))

    return features.finalize(), labels.finalize().ravel()


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    txt_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(txt_files)
    return common_cav.cache_scaled_split(
        DATASET_KEY, _load_raw_survival, source_version, BASE_DIR,
        test_size=test_size, random_state=random_state,
    )


def get_client_data(test_size: float = 0.33, random_state: int = 41, local_val_size: float = 0.2):
    x_train_pool, _, y_train_pool, _ = get_global_train_test_split(test_size, random_state)
    return common_cav.local_train_val_split(x_train_pool, y_train_pool, local_val_size, random_state)


def get_test_split(test_size: float = 0.33, random_state: int = 41):
    _, x_test, _, y_test = get_global_train_test_split(test_size, random_state)
    return x_test, y_test
