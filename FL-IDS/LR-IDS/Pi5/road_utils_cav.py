#!/usr/bin/env python
# coding: utf-8

"""ROAD (Real ORNL Automotive Dynamometer) Dataset loader — Pi5 (real,
non-simulated fuzzing/fabrication/masquerade attacks).

Raw data is auto-downloaded from Zenodo record 10462796 via Zenodo's public
REST API (https://zenodo.org/api/records/10462796), which is queried at
runtime for the record's real file list rather than guessing a zip URL. If
the automatic download fails (e.g. no network on the Pi), download manually
from https://zenodo.org/records/10462796 and place the extracted files
anywhere under ``ROAD Dataset/`` next to this file.

File classification: ROAD's published capture files follow the dataset's
own naming convention -- ambient (attack-free) captures are named
``ambient_*.log``. Every ``.log`` found recursively under ``ROAD Dataset/``
that isn't an ambient capture is looked up in the dataset's own
``attacks/capture_metadata.json`` (confirmed against the real downloaded
archive; every ``capture_metadata.json`` found anywhere under the folder is
merged into one lookup, keyed by capture name = log filename without
extension).

Frame-level labeling: each attack capture's metadata entry gives the
``injection_interval: [start, end]`` (seconds since the capture's first
frame), the ``injection_id`` and the ``injection_data_str`` (hex, 'X' = a
nibble left unaltered). A frame is Attack only if it is inside the interval,
has the injected ID ("XXX" = any ID, the fuzzing captures) and its data
matches the pattern. Labeling the whole interval instead marked ~96% normal
frames (every other ID, and the genuine frames of the target ID) as attack.
These labels match the authors' own per-frame labels in
``signal_extractions/`` exactly for all 13 captures those cover. The
interval check is needed: e.g. reverse_light_off's pattern also occurs in
normal traffic outside the injection.

The "accelerator attack" captures have ``injection_interval: null`` because
no messages were injected (ROAD readme), so they are entirely Normal, as in
the paper. The ``_masquerade`` variants (same capture with the conflicting
genuine frames removed) are not used.

The ambient captures' train blocks feed the ID period table (see
``common_cav``). The accelerator captures are normal too, but they are not
used for it: "no injected messages" is not the same as normal driving.

ID 0xFFF frames are dropped. 0xFFF cannot be a standard (11-bit, max 0x7FF)
CAN ID, it is not among the 105 messages of the dataset's own
``anonymized.dbc``, and it looks like a logging/anonymization artifact:
~6.3% of every capture, payload always zero, irregular spacing, often two at
the same timestamp. Kept, these "normal" all-zero DLC-8 frames at short
intervals would teach the shared model the opposite of the other clients'
DoS frames (all-zero DLC-8 at 0.5 ms = attack). They are dropped after
labeling, so injection intervals stay relative to each capture's first line.

Labels follow the authors' intervals exactly; a handful of injected frames
(1-7 per capture) fall just outside an interval and stay labeled normal, as
in the authors' own signal_extractions labels.

Split:
- Attack types recorded three times (``*_attack_1/_2/_3``): captures 1 and
  2 are train, 3 is test, so no injection session is on both sides.
- Everything else (ambient, accelerator, the single coolant capture): 10
  contiguous time blocks, blocks 2, 5 and 8 test (see
  ``common_cav.time_block_layout``).
"""

import glob
import json
import os
import re
import urllib.request
import zipfile

import numpy as np

import common_cav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROAD_DIR = os.path.join(BASE_DIR, "ROAD Dataset")
# New key whenever the rows, labels or features change: caches are found by
# key alone (see common_cav.find_any_cached_split), so an old one would be reused.
DATASET_KEY = "road_inj_h12x_s1"
SAMPLING = common_cav.Sampling()
ARTIFACT_ID = 0xFFF            # see module docstring
MAX_STANDARD_ID = 0x7FF
SOURCE_URL = "https://zenodo.org/records/10462796"
ZENODO_RECORD_API = "https://zenodo.org/api/records/10462796"

reshape_for_cnn = common_cav.reshape_for_cnn


def _find_log_files():
    return sorted(glob.glob(os.path.join(ROAD_DIR, "**", "*.log"), recursive=True))


def _download_road_dataset() -> None:
    os.makedirs(ROAD_DIR, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD_API) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    files = record.get("files") or []
    if not files:
        raise RuntimeError(f"Zenodo record returned no downloadable files: {ZENODO_RECORD_API}")

    for entry in files:
        filename = entry["key"]
        download_url = entry["links"]["self"]
        target_path = os.path.join(ROAD_DIR, filename)
        if os.path.exists(target_path):
            continue
        urllib.request.urlretrieve(download_url, target_path)
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(target_path, "r") as zf:
                zf.extractall(ROAD_DIR)


def _ensure_raw_files() -> list:
    log_files = _find_log_files()
    if log_files:
        return log_files

    try:
        _download_road_dataset()
    except Exception as error:
        raise FileNotFoundError(
            f"No ROAD .log files found under '{ROAD_DIR}', and automatic "
            f"download from Zenodo failed ({error}). Download manually from "
            f"{SOURCE_URL} and place the extracted captures under '{ROAD_DIR}'."
        ) from error

    log_files = _find_log_files()
    if not log_files:
        raise FileNotFoundError(
            f"Downloaded the ROAD Zenodo record but found no .log capture "
            f"files under '{ROAD_DIR}' afterwards. Inspect the downloaded "
            f"archive contents and adjust this loader if the layout differs."
        )
    return log_files


def _is_ambient(path: str) -> bool:
    return "ambient" in os.path.basename(path).lower()


def _find_metadata_files():
    return sorted(glob.glob(os.path.join(ROAD_DIR, "**", "capture_metadata.json"), recursive=True))


def _load_all_capture_metadata() -> dict:
    """Merge every ``capture_metadata.json`` found under ROAD_DIR (one per
    ``ambient/`` and ``attacks/`` subfolder in the official archive) into a
    single lookup keyed by capture name.
    """
    metadata = {}
    for path in _find_metadata_files():
        with open(path, "r", encoding="utf-8") as f:
            metadata.update(json.load(f))
    return metadata


def _capture_frames(path: str, capture_metadata: dict) -> common_cav.Frames:
    """Frames and per-frame labels for one capture, in time order."""
    ts, can_id, payload_hex, _ = common_cav.read_candump_log(path, with_label=False)
    dlc, data = common_cav.hex_payloads(payload_hex)
    y = np.zeros(len(ts), dtype=bool)
    if not _is_ambient(path):
        capture_name = os.path.splitext(os.path.basename(path))[0]
        entry = capture_metadata.get(capture_name)
        if entry is None:
            raise ValueError(f"No capture_metadata.json entry for attack capture {capture_name!r}")
        # Accelerator-attack captures have no interval: no injected messages (ROAD readme).
        if entry.get("injection_interval") is not None:
            y = _injected_frames(ts, can_id, payload_hex, entry)
    # Drop the 0xFFF artifact frames (after labeling; see module docstring).
    invalid = can_id > MAX_STANDARD_ID
    if np.any(can_id[invalid] != ARTIFACT_ID):
        raise ValueError(f"{path}: unexpected IDs above 0x7FF: {sorted(set(can_id[invalid].tolist()) - {ARTIFACT_ID})}")
    if y[invalid].any():
        raise ValueError(f"{path}: {int(y[invalid].sum())} attack-labeled 0xFFF frames")
    keep = ~invalid
    return common_cav.make_frames(ts[keep], can_id[keep], dlc[keep], data[keep], y[keep])


def _injected_frames(ts, can_id, payload_hex, entry) -> np.ndarray:
    """True for frames inside the injection interval whose ID and data match
    the injected ones (see the module docstring)."""
    start, end = entry["injection_interval"]
    relative = ts - ts[0]
    hit = (relative >= start) & (relative <= end)
    target = entry["injection_id"]
    if target != "XXX":
        hit &= can_id == int(target, 16)
    pattern = "^" + "".join("." if c == "X" else c for c in entry["injection_data_str"].upper()) + "$"
    return hit & payload_hex.str.match(pattern).to_numpy()


REPEATED_ATTACK_RE = re.compile(r"_attack_([123])$")
TEST_REPEAT = "3"


def _split_of(path: str) -> str:
    """Repeated attack captures by file (1, 2 train; 3 test), the rest by time blocks."""
    name = os.path.splitext(os.path.basename(path))[0]
    match = REPEATED_ATTACK_RE.search(name)
    if match is None or name.startswith("accelerator_attack"):
        return "blocks"
    return "test" if match.group(1) == TEST_REPEAT else "train"


def _captures() -> list:
    capture_metadata = _load_all_capture_metadata()
    # Masquerade variants are near-copies of their original capture; keeping
    # both would let the same frames appear on both sides of the split.
    log_files = [p for p in _ensure_raw_files() if "_masquerade" not in os.path.basename(p)]
    captures = [
        common_cav.Capture(
            os.path.basename(path), lambda path=path: _capture_frames(path, capture_metadata),
            split=_split_of(path), attack_free=_is_ambient(path),
        )
        for path in log_files
    ]
    # Every repeated attack type must have both train and test captures.
    repeats = {}
    for c in captures:
        match = REPEATED_ATTACK_RE.search(os.path.splitext(c.name)[0])
        if c.split != "blocks" and match:
            repeats.setdefault(c.name[:match.start()], set()).add(c.split)
    missing = {kind: sides for kind, sides in repeats.items() if sides != {"train", "test"}}
    if missing:
        raise ValueError(f"repeated attack types without both train and test captures: {missing}")
    return captures


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    cached = common_cav.find_any_cached_split(DATASET_KEY, BASE_DIR, test_size, random_state)
    if cached is not None:
        # A split_cache/*.npz already exists -- use it without ever
        # touching the (3GB) raw dataset, e.g. on a disk-constrained device
        # that only has the cache copied onto it.
        return cached

    log_files = _ensure_raw_files()
    source_version = common_cav.directory_signature(log_files + _find_metadata_files())
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
