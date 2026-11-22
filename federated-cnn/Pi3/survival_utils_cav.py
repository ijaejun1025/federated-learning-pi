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

Each vehicle has its own ID period table, built from the train blocks of
its ``FreeDrivingData`` capture (see ``common_cav``): the three cars use
different IDs.

Label fixes (frames sent by the attacker are attacks, frames sent by an ECU
are normal, as in the Car-Hacking dataset); only these two, each checked to
hit exactly the expected frames:

- ``Fuzzy_dataset_Spark``: the 88 frames right after the last T frame are
  labeled R but are injections: they start 1 ms after it, keep the fuzzer's
  0.78 ms rhythm for 80 ms, and each uses a random ID that occurs nowhere
  else. The labeled window simply ended 80 ms early. -> attack.
- ``Fuzzy_dataset_SONATA``: 15 T frames are ``0x6AA 12 00 00 00 00`` with
  DLC 5, while the fuzzer always sends DLC 8; it is the ECU reply that the
  Car-Hacking Fuzzy capture labels R. -> normal.

Other doubtful frames (reply-shaped T frames in the KIA/Sonata Fuzzy
captures, ~55) are left as labeled: only payload shape or timing thresholds
would identify them, and those rules disagree with each other.

Split: each capture is cut into 10 contiguous time blocks; blocks 2, 5 and 8
are test (see ``common_cav.time_block_layout``).
"""

import glob
import os
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SURVIVAL_DIR = os.path.join(BASE_DIR, "Survival Analysis Dataset")
# Cache name (split_cache/<DATASET_KEY>.npz): dataset name + version. Bump the
# version whenever the rows, labels, features or sampling change: a cache is
# found by this key alone, so an old one would otherwise be reused.
DATASET_KEY = "survival_v1"
SAMPLING = common_cav.Sampling()
# See "Label fixes" in the module docstring.
SPARK_FUZZY_LATE_INJECTIONS = 88
SPARK_LATE_WINDOW_S = 0.1
SONATA_FUZZY_ECU_REPLIES = 15
SONATA_ECU_REPLY_ID = 0x6AA
FUZZER_DLC = 8
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
    ts, ids, dlcs, data, y = [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            dlc = int(parts[2])
            rest = parts[3:]

            if len(rest) == 1 and " " in rest[0].strip():
                data_bytes = [int(b, 16) for b in rest[0].split()]
                flag = None
            else:
                data_bytes = [int(b, 16) for b in rest[:dlc]]
                flag = rest[dlc].strip().upper() if len(rest) > dlc else None

            data_bytes = data_bytes[:common_cav.NUM_DATA_BYTES]
            ts.append(float(parts[0]))
            ids.append(int(parts[1], 16))
            dlcs.append(dlc)
            data.append(data_bytes + [0] * (common_cav.NUM_DATA_BYTES - len(data_bytes)))
            y.append(flag == "T")
    return common_cav.make_frames(ts, ids, dlcs, np.array(data, dtype=np.uint8), y)


def _vehicle(path: str) -> str:
    """``Flooding_dataset_SONATA.txt`` -> ``SONATA``."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1].upper()


def _free_driving_path(vehicle: str) -> str:
    matches = [p for p in _ensure_raw_files()
               if _vehicle(p) == vehicle and os.path.basename(p).startswith("FreeDrivingData")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one FreeDrivingData capture for {vehicle}, found {matches}")
    return matches[0]


def _fix_labels(path: str, frames: common_cav.Frames) -> common_cav.Frames:
    """Apply the two label fixes (module docstring); raise if either no
    longer hits exactly the expected frames."""
    name = os.path.basename(path)
    y = frames.y.copy()
    if name == "Fuzzy_dataset_Spark.txt":
        free_ids = np.unique(_parse_survival_file(_free_driving_path("SPARK")).can_id)
        last_attack = frames.ts[y == 1].max()
        hit = (y == 0) & (frames.ts > last_attack) & ~np.isin(frames.can_id, free_ids)
        if int(hit.sum()) != SPARK_FUZZY_LATE_INJECTIONS or frames.ts[hit].max() - last_attack > SPARK_LATE_WINDOW_S:
            raise ValueError(f"{name}: late-injection fix would change {int(hit.sum())} frames, "
                             f"expected {SPARK_FUZZY_LATE_INJECTIONS} within {SPARK_LATE_WINDOW_S} s")
        y[hit] = 1
    elif name == "Fuzzy_dataset_SONATA.txt":
        hit = (y == 1) & (frames.dlc != FUZZER_DLC)
        if int(hit.sum()) != SONATA_FUZZY_ECU_REPLIES or set(frames.can_id[hit].tolist()) != {SONATA_ECU_REPLY_ID}:
            raise ValueError(f"{name}: ECU-reply fix would change {int(hit.sum())} frames, "
                             f"expected {SONATA_FUZZY_ECU_REPLIES} frames of ID {SONATA_ECU_REPLY_ID:#x}")
        y[hit] = 0
    else:
        return frames
    return common_cav.Frames(frames.ts, frames.can_id, frames.dlc, frames.data, y, frames.rtr)


def _capture_frames(path: str) -> common_cav.Frames:
    return _fix_labels(path, _parse_survival_file(path))


def _captures() -> list:
    return [
        common_cav.Capture(
            os.path.basename(path), lambda path=path: _capture_frames(path), split="blocks",
            attack_free=os.path.basename(path).startswith("FreeDrivingData"), group=_vehicle(path),
        )
        for path in _ensure_raw_files()
    ]


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    txt_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(txt_files)
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
