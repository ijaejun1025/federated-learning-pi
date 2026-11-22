#!/usr/bin/env python
# coding: utf-8

"""can-train-and-test loader -- Pi4 (set_04: 2017 Subaru Forester).

Source: https://bitbucket.org/brooke-lampe/can-train-and-test, curated from
the same author's can-dataset / can-ml repositories. Only set_04 is used:
``train_01`` for training and ``test_01_known_vehicle_known_attack`` (same
car, attack types seen in training) for test, the dataset's own split.
Files are auto-downloaded one by one from the Bitbucket API if missing and
checked against their published sizes.

Each CSV is ``timestamp,arbitration_id,data_field,attack`` with the ID and
payload in hex and a per-frame 0/1 label. DLC is the payload length (the
"systematic" captures include empty payloads, DLC 0). A few spoofing files
are slightly out of time order (speed-accessory-4: 32 rows, at most 0.11 ms),
so every capture is sorted by timestamp (stable) first.

How the data was made (can-dataset helper.sh, can-ml helper/label.py):
attacks were injected on the real car with SocketCAN ``cansend``, either at
a fixed interval or right after each genuine frame of the target ID, and
logged with ``candump -L any``. Timestamps were shifted so every capture
starts at 1672531200. Labels: a frame of the attack capture is 1 when its
(ID, payload) never occurs in the matching pre-attack capture. That rule
could in principle miss injections that reuse a seen payload or flag genuine
frames with a new payload; checked here, it does neither: inside every
attack window each attacked ID keeps exactly as many label-0 frames as its
period predicts. Bus-level gaps in the attack captures are often a few
microseconds (frames logged in bursts), but same-ID timing matches the
attack-free captures (median period ratio 1.0005), which is all the
features use.

The attack-free and accessory captures in train_01 feed the ID period table
(see ``common_cav``); the 48 IDs they share have identical periods.

Generalization sets (``iter_generalization_sets``, separate cache
``GEN_DATASET_KEY``): unseen attacks on this car and the 2011 Chevrolet
Traverse, which no client trains on. The server evaluates them once after the
last round; see the GEN_* constants for the files and the overlaps avoided.
"""

import os
import time
import urllib.request

import numpy as np
import pandas as pd

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANTT_DIR = os.path.join(BASE_DIR, "can-train-and-test")
# New key whenever the rows, labels or features change: caches are found by
# key alone (see common_cav.find_any_cached_split), so an old one would be reused.
DATASET_KEY = "cantt04_h12_s1_fs"
SAMPLING = common_cav.Sampling()
SOURCE_URL = "https://bitbucket.org/brooke-lampe/can-train-and-test"
API_FILE_URL = "https://api.bitbucket.org/2.0/repositories/brooke-lampe/can-train-and-test/src/master/{path}"

TRAIN_DIR = "set_04/train_01"
TEST_DIR = "set_04/test_01_known_vehicle_known_attack"
# Published file sizes in bytes (Bitbucket API, checked on download).
FILES = {
    f"{TRAIN_DIR}/accessory-3.csv": 1262428,
    f"{TRAIN_DIR}/accessory-4.csv": 2113444,
    f"{TRAIN_DIR}/attack-free-3.csv": 36838630,
    f"{TRAIN_DIR}/attack-free-4.csv": 13555370,
    f"{TRAIN_DIR}/interval-3.csv": 15924892,
    f"{TRAIN_DIR}/interval-4.csv": 36715373,
    f"{TRAIN_DIR}/rpm-3.csv": 27688947,
    f"{TRAIN_DIR}/rpm-4.csv": 25863545,
    f"{TRAIN_DIR}/rpm-accessory-3.csv": 6347809,
    f"{TRAIN_DIR}/rpm-accessory-4.csv": 7094561,
    f"{TRAIN_DIR}/speed-3.csv": 25981307,
    f"{TRAIN_DIR}/speed-4.csv": 19603552,
    f"{TRAIN_DIR}/speed-accessory-3.csv": 10058153,
    f"{TRAIN_DIR}/speed-accessory-4.csv": 15533840,
    f"{TRAIN_DIR}/standstill-3.csv": 32472163,
    f"{TRAIN_DIR}/standstill-4.csv": 14179466,
    f"{TRAIN_DIR}/systematic-3.csv": 40594450,
    f"{TRAIN_DIR}/systematic-4.csv": 49950983,
    f"{TEST_DIR}/interval-1.csv": 34641107,
    f"{TEST_DIR}/interval-2.csv": 18663978,
    f"{TEST_DIR}/rpm-1.csv": 19720399,
    f"{TEST_DIR}/rpm-2.csv": 22946502,
    f"{TEST_DIR}/rpm-accessory-1.csv": 5220106,
    f"{TEST_DIR}/rpm-accessory-2.csv": 5596879,
    f"{TEST_DIR}/speed-1.csv": 30421072,
    f"{TEST_DIR}/speed-2.csv": 28643415,
    f"{TEST_DIR}/speed-accessory-1.csv": 10558989,
    f"{TEST_DIR}/speed-accessory-2.csv": 6647294,
    f"{TEST_DIR}/standstill-1.csv": 12404934,
    f"{TEST_DIR}/standstill-2.csv": 35735469,
    f"{TEST_DIR}/systematic-1.csv": 20886277,
    f"{TEST_DIR}/systematic-2.csv": 25236201,
}
ATTACK_FREE_PREFIXES = ("attack-free", "accessory")
DOWNLOAD_ATTEMPTS = 5

# ---- Generalization sets (server-side evaluation only, never trained on) ----
# Evaluated once, after the last round, and reported apart from the 6 main
# test sets (see Server/server_cav.py):
#   forester_unknown_attack  set_04 test_03: same car, attack types not in train_01
#   traverse_known_attack    set_04 test_02: 2011 Chevrolet Traverse (never trained on)
#   traverse_unknown_attack  set_04 test_04: Traverse and unseen attack types
# Attack captures in this dataset are a replayed attack-free log plus the
# injected frames, so the same base traffic recurs across files. Two such
# overlaps would leak and are avoided:
# - Traverse calibration (period table, threshold calibration) uses
#   only set_02/train_01 attack-free-1/2: accessory-1/2 there are the base
#   traffic of the test_02 accessory attack captures.
# - test_03 double-1 is dropped: its base traffic is train_01 systematic-4.
GEN_DATASET_KEY = "cantt04_gen_h12_s1_fs"
TRAVERSE_CAL_DIR = "set_02/train_01"
FORESTER_UNKNOWN_DIR = "set_04/test_03_known_vehicle_unknown_attack"
TRAVERSE_KNOWN_DIR = "set_04/test_02_unknown_vehicle_known_attack"
TRAVERSE_UNKNOWN_DIR = "set_04/test_04_unknown_vehicle_unknown_attack"
TRAVERSE_CALIBRATION_FILES = {
    f"{TRAVERSE_CAL_DIR}/attack-free-1.csv": 33719258,
    f"{TRAVERSE_CAL_DIR}/attack-free-2.csv": 21757010,
}
GEN_FILES = {
    "forester_unknown_attack": {
        f"{FORESTER_UNKNOWN_DIR}/DoS-1.csv": 25795857,
        f"{FORESTER_UNKNOWN_DIR}/DoS-2.csv": 28481072,
        f"{FORESTER_UNKNOWN_DIR}/double-2.csv": 27856480,
        f"{FORESTER_UNKNOWN_DIR}/force-neutral-1.csv": 21158090,
        f"{FORESTER_UNKNOWN_DIR}/force-neutral-2.csv": 19047891,
        f"{FORESTER_UNKNOWN_DIR}/fuzzing-1.csv": 26438538,
        f"{FORESTER_UNKNOWN_DIR}/fuzzing-2.csv": 26107792,
        f"{FORESTER_UNKNOWN_DIR}/triple-1.csv": 22022316,
        f"{FORESTER_UNKNOWN_DIR}/triple-2.csv": 28462391,
    },
    "traverse_known_attack": {
        f"{TRAVERSE_KNOWN_DIR}/interval-1.csv": 23948856,
        f"{TRAVERSE_KNOWN_DIR}/interval-2.csv": 62683875,
        f"{TRAVERSE_KNOWN_DIR}/rpm-1.csv": 125672401,
        f"{TRAVERSE_KNOWN_DIR}/rpm-2.csv": 74511748,
        f"{TRAVERSE_KNOWN_DIR}/rpm-accessory-1.csv": 19994030,
        f"{TRAVERSE_KNOWN_DIR}/rpm-accessory-2.csv": 11574573,
        f"{TRAVERSE_KNOWN_DIR}/speed-1.csv": 99358561,
        f"{TRAVERSE_KNOWN_DIR}/speed-2.csv": 104275419,
        f"{TRAVERSE_KNOWN_DIR}/speed-accessory-1.csv": 20175669,
        f"{TRAVERSE_KNOWN_DIR}/speed-accessory-2.csv": 11605872,
        f"{TRAVERSE_KNOWN_DIR}/standstill-1.csv": 45606102,
        f"{TRAVERSE_KNOWN_DIR}/standstill-2.csv": 23370239,
        f"{TRAVERSE_KNOWN_DIR}/systematic-1.csv": 20551827,
        f"{TRAVERSE_KNOWN_DIR}/systematic-2.csv": 17343447,
    },
    "traverse_unknown_attack": {
        f"{TRAVERSE_UNKNOWN_DIR}/DoS-1.csv": 15324766,
        f"{TRAVERSE_UNKNOWN_DIR}/DoS-2.csv": 15998666,
        f"{TRAVERSE_UNKNOWN_DIR}/double-1.csv": 41185016,
        f"{TRAVERSE_UNKNOWN_DIR}/double-2.csv": 17526419,
        f"{TRAVERSE_UNKNOWN_DIR}/force-neutral-1.csv": 19220276,
        f"{TRAVERSE_UNKNOWN_DIR}/force-neutral-2.csv": 17227950,
        f"{TRAVERSE_UNKNOWN_DIR}/fuzzing-1.csv": 46782254,
        f"{TRAVERSE_UNKNOWN_DIR}/fuzzing-2.csv": 44490428,
        f"{TRAVERSE_UNKNOWN_DIR}/triple-1.csv": 67253172,
        f"{TRAVERSE_UNKNOWN_DIR}/triple-2.csv": 24947957,
    },
}
GEN_SET_VEHICLE = {"forester_unknown_attack": "forester", "traverse_known_attack": "traverse",
                   "traverse_unknown_attack": "traverse"}

reshape_for_cnn = common_cav.reshape_for_cnn


def _local_path(rel_path: str) -> str:
    return os.path.join(CANTT_DIR, *rel_path.split("/"))


def _download(rel_path: str, size: int) -> None:
    path = _local_path(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    error = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            urllib.request.urlretrieve(API_FILE_URL.format(path=rel_path), path + ".part")
            if os.path.getsize(path + ".part") != size:
                raise IOError(f"got {os.path.getsize(path + '.part')} bytes, expected {size}")
            os.replace(path + ".part", path)
            return
        except Exception as exc:  # Bitbucket rate-limits bursts; back off and retry
            error = exc
            time.sleep(5 * 2 ** attempt)
    raise FileNotFoundError(
        f"Could not download {rel_path} from {SOURCE_URL} ({error}). Download set_04's "
        f"train_01 and test_01_known_vehicle_known_attack folders manually into '{CANTT_DIR}'."
    )


def _ensure_raw_files(files: dict = None) -> list:
    files = FILES if files is None else files
    for rel_path, size in files.items():
        path = _local_path(rel_path)
        if not os.path.exists(path) or os.path.getsize(path) != size:
            _download(rel_path, size)
    return [_local_path(rel_path) for rel_path in files]


def _capture_frames(path: str) -> common_cav.Frames:
    df = pd.read_csv(path, dtype={"arbitration_id": str, "data_field": str})
    expected = ["timestamp", "arbitration_id", "data_field", "attack"]
    if list(df.columns) != expected:
        raise ValueError(f"{path}: columns {list(df.columns)}, expected {expected}")
    if not df["attack"].isin([0, 1]).all():
        raise ValueError(f"{path}: 'attack' column has values other than 0/1")
    dlc, data = common_cav.hex_payloads(df["data_field"])
    can_id = np.array([int(s, 16) for s in df["arbitration_id"]], dtype=np.int64)
    return common_cav.make_frames(df["timestamp"].to_numpy(), can_id, dlc, data,
                                  df["attack"].to_numpy() == 1, sort_by_time=True)


def _captures() -> list:
    captures = []
    for rel_path in FILES:
        folder, name = rel_path.rsplit("/", 1)
        path = _local_path(rel_path)
        captures.append(common_cav.Capture(
            f"{folder.rsplit('/', 1)[-1]}/{name}", lambda path=path: _capture_frames(path),
            split="train" if folder == TRAIN_DIR else "test",
            attack_free=folder == TRAIN_DIR and name.startswith(ATTACK_FREE_PREFIXES),
        ))
    return captures


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    """The dataset's own split (train_01 / test_01), scaled with the fixed
    scaling (common_cav.scale_features) and cached to split_cache/.
    ``test_size`` only names the cache."""
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        return cached

    files = _ensure_raw_files()
    source_version = common_cav.directory_signature(files)
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


# ------------------------------------------------------- generalization sets
def _gen_captures(set_name: str) -> list:
    """Reference captures for the set's vehicle, then the set's test captures.

    Forester: train_01's attack-free/accessory captures as baseline only, so
    the period table is exactly the one the training data used. Traverse:
    its calibration captures as "train" rows (label-free normal traffic,
    used for the period table and threshold calibration).
    """
    captures = []
    if GEN_SET_VEHICLE[set_name] == "forester":
        for rel_path in FILES:
            folder, name = rel_path.rsplit("/", 1)
            if folder == TRAIN_DIR and name.startswith(ATTACK_FREE_PREFIXES):
                path = _local_path(rel_path)
                captures.append(common_cav.Capture(f"reference/{name}", lambda path=path: _capture_frames(path),
                                                   split="baseline", attack_free=True))
    else:
        for rel_path in TRAVERSE_CALIBRATION_FILES:
            path = _local_path(rel_path)
            captures.append(common_cav.Capture(f"calibration/{rel_path.rsplit('/', 1)[-1]}",
                                               lambda path=path: _capture_frames(path), split="train", attack_free=True))
    for rel_path in GEN_FILES[set_name]:
        path = _local_path(rel_path)
        captures.append(common_cav.Capture(f"{set_name}/{rel_path.rsplit('/', 1)[-1]}",
                                           lambda path=path: _capture_frames(path), split="test"))
    return captures


def _build_generalization_cache(test_size: float, random_state: int) -> str:
    all_files = dict(FILES)
    all_files.update(TRAVERSE_CALIBRATION_FILES)
    for files in GEN_FILES.values():
        all_files.update(files)
    source_version = common_cav.directory_signature(_ensure_raw_files(all_files))

    arrays = {"set_names": np.array(list(GEN_FILES), dtype=str)}
    # Calibration rows ("train") are all kept; test normals are capped as for
    # the main test sets, with weights.
    sampling = common_cav.Sampling(train_rows=None)
    for set_name in GEN_FILES:
        captures = _gen_captures(set_name)
        split = common_cav.build_feature_split(captures, sampling, random_state)
        x_cal, x_test, y_test, tables, test_capture = split.x_train, split.x_test, split.y_test, split.tables, split.test_capture
        arrays[f"w_{set_name}"] = split.w_test
        # Every vehicle gets the same fixed scaling as the training data.
        if GEN_SET_VEHICLE[set_name] == "forester":
            arrays[f"calibration_{set_name}"] = np.array("")
        else:
            arrays["x_traverse_calibration"] = common_cav.scale_features(x_cal)
            arrays[f"calibration_{set_name}"] = np.array("traverse_calibration")
        first_test = len(captures) - len(GEN_FILES[set_name])
        arrays[f"x_{set_name}"] = common_cav.scale_features(x_test)
        arrays[f"y_{set_name}"] = y_test.astype(np.int8)
        arrays[f"file_{set_name}"] = (test_capture - first_test).astype(np.int16)
        arrays[f"files_{set_name}"] = np.array([c.name.split("/", 1)[1] for c in captures[first_test:]], dtype=str)
        for key, value in common_cav.period_table_arrays(tables).items():
            arrays[f"{key}_{set_name}"] = value
        del split, x_cal, x_test

    cache_dir = os.path.join(BASE_DIR, "split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{GEN_DATASET_KEY}_ts{test_size}_rs{random_state}_src{source_version}.npz")
    np.savez(path, **arrays)
    return path


def iter_generalization_sets(test_size: float = 0.33, random_state: int = 41):
    """Yield one generalization set at a time (to keep server memory low):
    ``(name, x, y, weights, file_index, file_names, x_calibration)``;
    ``weights`` is how many rows each row stands for (test normals are
    capped, see common_cav.Sampling); ``x_calibration`` is the new vehicle's
    scaled attack-free traffic for threshold calibration, or None for the
    known vehicle."""
    path = common_cav.cached_split_path(GEN_DATASET_KEY, BASE_DIR, test_size, random_state)
    if path is None:
        path = _build_generalization_cache(test_size, random_state)
    with np.load(path) as data:
        for name in data["set_names"].tolist():
            calibration = str(data[f"calibration_{name}"])
            yield (name, data[f"x_{name}"], data[f"y_{name}"], data[f"w_{name}"], data[f"file_{name}"],
                   data[f"files_{name}"].tolist(), data[f"x_{calibration}"] if calibration else None)
