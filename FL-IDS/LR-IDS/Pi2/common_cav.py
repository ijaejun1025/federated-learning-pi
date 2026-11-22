#!/usr/bin/env python
# coding: utf-8

"""Dataset-agnostic helpers shared by every ``*_utils_cav.py`` module.

Each per-dataset module only turns its raw captures into frames (timestamp,
CAN ID, DLC, data bytes, per-frame label) and says how each capture is split.
Everything after that -- per-frame features, the ID period table, row
sampling, scaling, caching and the client's local train/val split -- is the
same for every dataset and lives here.

Features (12 per frame, ``FEATURE_NAMES``):

- ``DLC``, ``DATA0`` .. ``DATA7``: the frame itself (short frames zero-padded).
- ``log_dt``: log10 of the time since the previous frame of the same stream.
- ``log_dt_over_period``: log10 of that time divided by the stream's usual period.
- ``bits_rel``: data bits that differ from the previous frame of the stream,
  relative to the stream's usual change: log2((bits + 1) / (mean_bits + 1)),
  with ``mean_bits`` from the period table. 0 is a usual change for this ID;
  negative is fewer bits than usual (e.g. a repeated payload on an ID whose
  payload normally changes). The raw bit count meant opposite things on
  different datasets: an unchanged payload is mostly an attack (DoS,
  spoofing) in Car-Hacking, OTIDS, Survival and can-train-and-test, but
  mostly normal in ROAD, where many IDs never change.

A stream is a CAN ID plus a frame kind (``frame_streams``): data frame,
remote frame (RTR bit set: a request without data), or the data frame that
answers a remote request (the ID's previous frame was a remote request).
Kept apart, a periodic data frame is compared with the previous periodic
one, not with a request or reply a millisecond earlier. The CAN ID itself is
not a feature (IDs mean different things on different vehicles); it is only
used to find each frame's previous frame. The frame kind needs only the
frames themselves, not labels.

No feature uses labels or test data:

- History (previous frame of the stream) never crosses a capture, a time
  block, or a recording gap (more than ``CAPTURE_GAP_S`` with no frame at
  all, i.e. the recording stopped), so a train frame never looks at a test
  frame and vice versa. The first frame of a stream in such a segment has no
  previous frame (``FIRST_FRAME_DT``, 0 bits).
- The period table (each stream's period and mean bits changed) comes only
  from the train part of attack-free captures (a capture-level fact from the
  dataset documentation, not per-frame labels). A stream missing from it gets
  the lower median period (and mean bits) of the table's streams of the same
  kind, which is the value of one real stream of that kind, so "ID never seen
  in attack-free traffic" is not encoded.
"""

import os
import re
import urllib.parse
import urllib.request

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_TO_INT = {"Normal": 0, "ATTACK": 1}

NUM_DATA_BYTES = 8
FEATURE_NAMES = (
    ["DLC"] + [f"DATA{i}" for i in range(NUM_DATA_BYTES)]
    + ["log_dt", "log_dt_over_period", "bits_rel"]
)
NUM_FEATURES = len(FEATURE_NAMES)

# Seconds used as the time since the previous same-ID frame when there is none
# (first frame of an ID in a capture or time block); log_dt is capped here too.
FIRST_FRAME_DT = 1.0
# A live bus carries hundreds to thousands of frames per second, so a second
# without any frame means the recording was interrupted.
CAPTURE_GAP_S = 1.0
MIN_DT = 1e-6
RATIO_RANGE = (1e-4, 1e3)

# Fixed scaling, the same on every client, the server and a new vehicle:
# scaled = (feature - FEATURE_CENTER) / FEATURE_SCALE. A scaler fit on each
# client's own rows would turn the same frame into different inputs on
# different clients (e.g. log_dt_over_period's spread differs ~9x between
# datasets, and ROAD's DLC is always 8, so its std is 0), and the one global
# model would have to learn six input scales at once. The ranges are known:
# DLC 0..8 and bytes 0..255 map to -1..1, log_dt (-6..0) to -4..2,
# log_dt_over_period (-4..3) stays as it is, bits_rel (-6..6) maps to -3..3.
FEATURE_CENTER = np.array([4.0] + [127.5] * NUM_DATA_BYTES + [-2.0, 0.0, 0.0])
FEATURE_SCALE = np.array([4.0] + [127.5] * NUM_DATA_BYTES + [1.0, 1.0, 2.0])

TIME_BLOCKS = 10
TEST_BLOCKS = (2, 5, 8)
BLOCK_BUFFER_FRAC = 0.005

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HEX_VALUE = np.zeros(256, dtype=np.uint8)
for _i, _c in enumerate("0123456789ABCDEF"):
    _HEX_VALUE[ord(_c)] = _i
    _HEX_VALUE[ord(_c.lower())] = _i


def download_from_google_drive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive by ID, handling the "Google Drive
    can't scan this file for viruses" interstitial Google shows for large
    files (confirmed against a real >300MB download): that page is an HTML
    form whose hidden ``id``/``export``/``confirm``/``uuid`` fields must be
    resubmitted as a GET to ``drive.usercontent.google.com/download`` --
    it is NOT a simple ``...&confirm=<token>`` query param on the original
    URL, which is a common but outdated assumption.
    """
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "fl-ids-dataset-downloader")]

    with opener.open(base_url) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()

    if "text/html" in content_type:
        html = body.decode("utf-8", errors="ignore")

        def _hidden_input(name):
            match = re.search(rf'name="{name}"\s+value="([^"]*)"', html)
            return match.group(1) if match else None

        action_match = re.search(r'<form[^>]+action="([^"]+)"', html)
        if not action_match:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of the file, "
                "and no download form could be found in it."
            )
        action_url = action_match.group(1)

        params = {
            "id": _hidden_input("id") or file_id,
            "export": _hidden_input("export") or "download",
            "confirm": _hidden_input("confirm") or "t",
        }
        uuid_value = _hidden_input("uuid")
        if uuid_value:
            params["uuid"] = uuid_value

        confirm_url = f"{action_url}?{urllib.parse.urlencode(params)}"
        with opener.open(confirm_url) as resp:
            body = resp.read()

    with open(dest_path, "wb") as f:
        f.write(body)


# ------------------------------------------------------------------ frames
@dataclass
class Frames:
    """One capture in time order: ``data`` is (n, 8) uint8, zero-padded;
    ``rtr`` marks remote frames."""
    ts: np.ndarray
    can_id: np.ndarray
    dlc: np.ndarray
    data: np.ndarray
    y: np.ndarray
    rtr: np.ndarray

    def __len__(self):
        return len(self.ts)

    def rows(self, index) -> "Frames":
        return Frames(self.ts[index], self.can_id[index], self.dlc[index], self.data[index],
                      self.y[index], self.rtr[index])


def make_frames(ts, can_id, dlc, data, y, rtr=None, sort_by_time: bool = False) -> Frames:
    n = len(ts)
    frames = Frames(
        ts=np.asarray(ts, dtype=np.float64),
        can_id=np.asarray(can_id, dtype=np.int64),
        dlc=np.asarray(dlc, dtype=np.int8),
        data=np.ascontiguousarray(data, dtype=np.uint8).reshape(-1, NUM_DATA_BYTES),
        y=np.asarray(y, dtype=np.int8),
        rtr=np.zeros(n, dtype=np.int64) if rtr is None else np.asarray(rtr, dtype=np.int64),
    )
    if not all(len(a) == n for a in (frames.can_id, frames.dlc, frames.data, frames.y, frames.rtr)):
        raise ValueError("frame columns have different lengths")
    if sort_by_time:
        frames = frames.rows(np.argsort(frames.ts, kind="stable"))
    elif n > 1 and np.any(np.diff(frames.ts) < 0):
        raise ValueError("timestamps are not in time order; pass sort_by_time=True if that is expected")
    return frames


def hex_payloads(hex_strings) -> tuple:
    """(dlc, data) from hex payload strings such as ``"0EA48100"``; an empty
    or missing payload is DLC 0."""
    s = pd.Series(hex_strings).fillna("").astype(str).str.strip()
    lengths = s.str.len().to_numpy()
    if np.any(lengths > 2 * NUM_DATA_BYTES) or np.any(lengths % 2):
        bad = s[(lengths > 2 * NUM_DATA_BYTES) | (lengths % 2 == 1)].iloc[0]
        raise ValueError(f"payload is not 0-8 whole bytes of hex: {bad!r}")
    padded = "".join(s.str.ljust(2 * NUM_DATA_BYTES, "0"))
    raw = np.frombuffer(padded.encode("ascii"), dtype=np.uint8).reshape(-1, 2 * NUM_DATA_BYTES)
    nibbles = _HEX_VALUE[raw]
    data = nibbles[:, 0::2] * 16 + nibbles[:, 1::2]
    return (lengths // 2).astype(np.int8), data.astype(np.uint8)


def read_candump_log(path: str, with_label: bool):
    """Read a ``(timestamp) iface ID#HEXDATA [label]`` candump log.

    Returns ``(ts, can_id, hex_payload, label)``; ``label`` is None unless
    ``with_label``, in which case every frame must end with 0 or 1.
    """
    names = ["ts", "iface", "frame", "label"] if with_label else ["ts", "iface", "frame"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, usecols=range(len(names)),
                     dtype=str, engine="c")
    ts = df["ts"].str.strip("()").astype(np.float64).to_numpy()
    parts = df["frame"].str.split("#", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(f"{path}: no ID#DATA field found")
    can_id = np.array([int(s, 16) for s in parts[0]], dtype=np.int64)
    label = None
    if with_label:
        ok = df["label"].isin(["0", "1"]).to_numpy()
        if not ok.all():
            row = int(np.flatnonzero(~ok)[0])
            raise ValueError(f"{path}: frame {row + 1} has no 0/1 label: {df.iloc[row].tolist()}")
        label = (df["label"] == "1").to_numpy()
    return ts, can_id, parts[1].fillna("").str.upper(), label


_HCRL_TXT_RE = re.compile(
    r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+(\d+)\s+DLC:\s*(\d+)\s*((?:[0-9a-fA-F]{2}\s*)*)"
)
_HCRL_FRAME_TYPES = {"000": 0, "100": 1}  # data frame, remote frame


def parse_hcrl_txt(path: str):
    """``Timestamp: t  ID: 0316  000  DLC: 8  05 20 ...`` captures (OTIDS and
    Car-Hacking's normal_run_data). The field after the ID is ``000`` for a
    data frame and ``100`` for a remote frame (DLC 0, no bytes).
    Returns ``(ts, can_id, dlc, data, rtr)``."""
    ts, ids, dlcs, data, rtr = [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _HCRL_TXT_RE.search(line)
            if not match:
                continue
            frame_type = match.group(3)
            if frame_type not in _HCRL_FRAME_TYPES:
                raise ValueError(f"{path}: unknown frame type field {frame_type!r} in {line.strip()!r}")
            payload = [int(b, 16) for b in match.group(5).split()][:NUM_DATA_BYTES]
            ts.append(float(match.group(1)))
            ids.append(int(match.group(2), 16))
            rtr.append(_HCRL_FRAME_TYPES[frame_type])
            dlcs.append(int(match.group(4)))
            data.append(payload + [0] * (NUM_DATA_BYTES - len(payload)))
    return ts, ids, dlcs, np.array(data, dtype=np.uint8).reshape(-1, NUM_DATA_BYTES), rtr


# ------------------------------------------------------------- captures
@dataclass
class Capture:
    """How one raw capture is used.

    split: ``"blocks"`` (10 time blocks, blocks 2/5/8 test), ``"train"``,
    ``"test"``, or ``"baseline"`` (only feeds the ID period table, never
    becomes train or test rows). attack_free: the whole capture is known to
    contain no attacks, so its train part may feed the ID period table.
    group: frames of one vehicle share an ID period table.
    """
    name: str
    load: Callable[[], Frames]
    split: str
    attack_free: bool = False
    group: str = "vehicle"


# Class balance (see Sampling). Every client gets the same number of training
# rows, so FedAvg (which weights clients by row count) treats them alike and
# no Pi is much slower than the others. Interim value: fixed after the model
# review, since it sets the round time on the Pis.
TRAIN_ROWS_PER_CLIENT = 600_000
TRAIN_ATTACK_SHARE_MAX = 0.20
TEST_NORMALS_MAX = 2_000_000


@dataclass
class Sampling:
    """Which rows a dataset keeps, drawn uniformly at random after the
    features are computed (so time features always see every frame):

    - train: every attack row up to ``train_attack_share_max`` of
      ``train_rows``, then normal rows up to ``train_rows`` in total.
      ``train_rows=None`` keeps every train row.
    - test: every attack row, and at most ``test_normals_max`` normal rows
      (None: all). Kept test normals get weight total/kept, so weighted
      metrics estimate the full test set.
    """
    train_rows: Optional[int] = TRAIN_ROWS_PER_CLIENT
    train_attack_share_max: float = TRAIN_ATTACK_SHARE_MAX
    test_normals_max: Optional[int] = TEST_NORMALS_MAX


@dataclass
class FeatureSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    w_test: np.ndarray       # per test row: how many test rows it stands for
    tables: dict             # {vehicle group: PeriodTable}
    test_capture: np.ndarray  # per test row: index of its capture in the input list


def time_block_layout(n_rows: int):
    """Per row: time block index, train mask, test mask.

    Blocks in ``TEST_BLOCKS`` are test. Train blocks give up
    ``BLOCK_BUFFER_FRAC`` of the capture's rows at each end, so train rows
    are not right next to a test block.
    """
    block = np.zeros(n_rows, dtype=np.int16)
    train = np.zeros(n_rows, dtype=bool)
    test = np.zeros(n_rows, dtype=bool)
    buf = int(n_rows * BLOCK_BUFFER_FRAC)
    for b, rows in enumerate(np.array_split(np.arange(n_rows), TIME_BLOCKS)):
        block[rows] = b
        if b in TEST_BLOCKS:
            test[rows] = True
        else:
            train[rows[buf:len(rows) - buf] if len(rows) > 2 * buf else rows] = True
    return block, train, test


DATA_FRAME, REMOTE_FRAME, REMOTE_REPLY = 0, 1, 2
FRAME_KINDS = 3


def _previous_in_group(key: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Index of the previous row with the same key in the same segment, or -1."""
    n = len(key)
    order = np.lexsort((np.arange(n), key, segment))
    prev = np.full(n, -1, dtype=np.int64)
    same = (key[order[1:]] == key[order[:-1]]) & (segment[order[1:]] == segment[order[:-1]])
    prev[order[1:][same]] = order[:-1][same]
    return prev


def frame_streams(frames: Frames, segment: np.ndarray) -> np.ndarray:
    """Stream key per frame: ``can_id * FRAME_KINDS + kind``, where kind is
    remote frame, data frame answering a remote request (the previous frame
    of its ID in the segment is a remote frame), or plain data frame."""
    prev = _previous_in_group(frames.can_id, segment)
    answers_request = (frames.rtr == 0) & (prev >= 0) & (frames.rtr[np.maximum(prev, 0)] == 1)
    kind = np.where(frames.rtr == 1, REMOTE_FRAME, np.where(answers_request, REMOTE_REPLY, DATA_FRAME))
    return frames.can_id * FRAME_KINDS + kind


def same_id_history(frames: Frames, segment: np.ndarray):
    """Time since, and data bits changed from, the previous frame of the
    same stream (``frame_streams``) in the same segment; NaN / 0 when there
    is none. Returns ``(dt, bits, stream)``."""
    n = len(frames)
    stream = frame_streams(frames, segment)
    prev = _previous_in_group(stream, segment)
    has_prev = prev >= 0
    dt = np.full(n, np.nan)
    dt[has_prev] = frames.ts[has_prev] - frames.ts[prev[has_prev]]
    words = frames.data.view(np.uint64).ravel()
    changed = np.zeros(n, dtype=np.uint64)
    changed[has_prev] = words[has_prev] ^ words[prev[has_prev]]
    bits = _POPCOUNT[changed.view(np.uint8).reshape(n, NUM_DATA_BYTES)].sum(axis=1).astype(np.uint8)
    return dt, bits, stream


@dataclass
class PeriodTable:
    """Per stream (``frame_streams``) in attack-free traffic: the median time
    between its frames and the mean number of data bits changed from its
    previous frame. A stream missing from the table gets the fallback of its
    frame kind (``fallbacks`` / ``bits_fallbacks``)."""
    ids: np.ndarray
    periods: np.ndarray
    fallbacks: np.ndarray
    mean_bits: np.ndarray
    bits_fallbacks: np.ndarray

    @property
    def fallback(self) -> float:
        return float(self.fallbacks[DATA_FRAME])

    def _find(self, stream: np.ndarray):
        pos = np.clip(np.searchsorted(self.ids, stream), 0, max(len(self.ids) - 1, 0))
        return pos, self.ids[pos] == stream

    def lookup(self, stream: np.ndarray) -> np.ndarray:
        pos, found = self._find(stream)
        return np.where(found, self.periods[pos], self.fallbacks[stream % FRAME_KINDS])

    def lookup_mean_bits(self, stream: np.ndarray) -> np.ndarray:
        pos, found = self._find(stream)
        return np.where(found, self.mean_bits[pos], self.bits_fallbacks[stream % FRAME_KINDS])


def _lower_median(values: np.ndarray) -> float:
    """Always one of ``values``: np.median would average the two middle
    values for an even count, giving a period no real stream has."""
    return float(np.sort(values)[(len(values) - 1) // 2])


def _period_table(parts: list) -> PeriodTable:
    """``parts``: (stream, dt, bits) of the attack-free train frames that have a previous frame."""
    ids = np.concatenate([p[0] for p in parts])
    dts = np.concatenate([p[1] for p in parts])
    bits = np.concatenate([p[2] for p in parts]).astype(np.float64)
    if len(ids) == 0:
        raise ValueError("no attack-free train frames to build the ID period table from")
    medians = pd.Series(dts).groupby(ids).median()
    streams = medians.index.to_numpy(dtype=np.int64)
    periods = medians.to_numpy(dtype=np.float64)
    _, inverse = np.unique(ids, return_inverse=True)  # same sorted streams as the groupby
    mean_bits = np.bincount(inverse, weights=bits) / np.bincount(inverse)
    # Fallbacks come from streams of the same kind, so a missing data stream
    # looks like some real data stream (and not, say, like the remote-frame
    # streams, which a data frame never is).
    kinds = streams % FRAME_KINDS
    if not np.any(kinds == DATA_FRAME):
        raise ValueError("no data frames in the attack-free train frames")

    def per_kind(values):
        data_fallback = _lower_median(values[kinds == DATA_FRAME])
        return np.array([
            _lower_median(values[kinds == kind]) if np.any(kinds == kind) else data_fallback
            for kind in range(FRAME_KINDS)
        ])
    return PeriodTable(ids=streams, periods=periods, fallbacks=per_kind(periods),
                       mean_bits=mean_bits, bits_fallbacks=per_kind(mean_bits))


def _features(dlc, data, dt, bits, period, mean_bits):
    first = np.isnan(dt)
    dt = np.where(first, FIRST_FRAME_DT, dt)
    ratio = np.where(first, 1.0, np.maximum(dt, MIN_DT) / period)
    return np.column_stack([
        dlc, data,
        np.log10(np.clip(dt, MIN_DT, FIRST_FRAME_DT)),
        np.log10(np.clip(ratio, *RATIO_RANGE)),
        np.log2((bits.astype(np.float64) + 1.0) / (mean_bits + 1.0)),
    ]).astype(np.float32)


TRAIN_NORMAL, TRAIN_ATTACK, TEST_NORMAL, TEST_ATTACK = 0, 1, 2, 3
POOL_NAMES = ("train normal", "train attack", "test normal", "test attack")


def _pool_caps(sampling: Sampling) -> list:
    """Upper bound of rows kept per pool while streaming (None = all)."""
    if sampling.train_rows is None:
        train_normal = train_attack = None
    else:
        train_attack = int(sampling.train_rows * sampling.train_attack_share_max)
        train_normal = sampling.train_rows
    return [train_normal, train_attack, sampling.test_normals_max, None]


def _prune(kept: list, thresholds: np.ndarray) -> None:
    """Drop kept rows whose random key is above their pool's threshold."""
    for part in kept:
        mask = part["key"] <= thresholds[part["pool"]]
        if not mask.all():
            for name in list(part):
                if isinstance(part[name], np.ndarray):
                    part[name] = part[name][mask]


def _kth_key(kept: list, pool: int, k: int) -> float:
    keys = np.concatenate([part["key"][part["pool"] == pool] for part in kept])
    return float(np.partition(keys, k - 1)[k - 1]) if len(keys) > k else np.inf


def build_feature_split(captures: Iterable[Capture], sampling: Sampling, random_state: int = 41,
                        log=print) -> FeatureSplit:
    """Load every capture once, compute the 12 features, sample rows.

    Rows are drawn per pool (train normal / train attack / test normal /
    test attack) as a uniform random sample of the whole dataset: each row
    gets a random key and a pool keeps its rows with the smallest keys, so
    the result does not depend on capture order or size, and memory stays
    bounded by the pool sizes.
    """
    caps = _pool_caps(sampling)
    thresholds = np.full(4, np.inf)
    totals = np.zeros(4, dtype=np.int64)
    kept = []
    baseline_parts = {}
    for index, capture in enumerate(captures):
        frames = capture.load()
        n = len(frames)
        if capture.attack_free and frames.y.any():
            raise ValueError(f"{capture.name} is marked attack-free but has {int(frames.y.sum())} attack labels")
        if capture.split == "blocks":
            segment, train, test = time_block_layout(n)
        elif capture.split in ("train", "test", "baseline"):
            segment = np.zeros(n, dtype=np.int16)
            train = np.full(n, capture.split == "train")
            test = np.full(n, capture.split == "test")
        else:
            raise ValueError(f"{capture.name}: unknown split {capture.split!r}")
        recording_gaps = np.diff(frames.ts) > CAPTURE_GAP_S
        segment = np.cumsum(np.r_[0, (np.diff(segment) != 0) | recording_gaps]).astype(np.int32)
        dt, bits, stream = same_id_history(frames, segment)

        if capture.attack_free:
            usable = (train | (capture.split == "baseline")) & ~np.isnan(dt)
            baseline_parts.setdefault(capture.group, []).append(
                (stream[usable], dt[usable].astype(np.float32), bits[usable]))

        attack = frames.y == 1
        pool = np.full(n, -1, dtype=np.int8)
        pool[train & ~attack] = TRAIN_NORMAL
        pool[train & attack] = TRAIN_ATTACK
        pool[test & ~attack] = TEST_NORMAL
        pool[test & attack] = TEST_ATTACK
        totals += np.bincount(pool[pool >= 0], minlength=4)
        key = np.random.default_rng([random_state, index]).random(n)
        keep = (pool >= 0) & (key <= thresholds[np.maximum(pool, 0)])
        kept.append({
            "index": index, "group": capture.group, "pool": pool[keep], "key": key[keep],
            "stream": stream[keep], "dlc": frames.dlc[keep], "data": frames.data[keep],
            "dt": dt[keep], "bits": bits[keep], "y": frames.y[keep],
        })
        replies = int((stream % FRAME_KINDS == REMOTE_REPLY).sum())
        log(f"[DATA] {capture.name}: {n} frames ({int(attack.sum())} attack, "
            f"{int(frames.rtr.sum())} remote, {replies} remote replies), "
            f"{int(recording_gaps.sum())} recording gaps, {int(train.sum())} train / {int(test.sum())} test rows")
        del frames, dt, bits
        # Tighten each capped pool once it holds more than twice its cap.
        for p, cap in enumerate(caps):
            if cap is not None and sum(int((part["pool"] == p).sum()) for part in kept) > 2 * cap:
                thresholds[p] = _kth_key(kept, p, cap)
        _prune(kept, thresholds)

    # Final sizes: attacks first, then normals fill the training rows.
    for p in (TRAIN_ATTACK, TEST_NORMAL):
        if caps[p] is not None:
            thresholds[p] = _kth_key(kept, p, caps[p])
    _prune(kept, thresholds)
    if caps[TRAIN_NORMAL] is not None:
        n_attack = sum(int((part["pool"] == TRAIN_ATTACK).sum()) for part in kept)
        thresholds[TRAIN_NORMAL] = _kth_key(kept, TRAIN_NORMAL, caps[TRAIN_NORMAL] - n_attack)
        _prune(kept, thresholds)
    counts = np.array([sum(int((part["pool"] == p).sum()) for part in kept) for p in range(4)])
    log("[SAMPLE] " + ", ".join(f"{POOL_NAMES[p]} {counts[p]}/{totals[p]}" for p in range(4)))

    tables = {group: _period_table(parts) for group, parts in baseline_parts.items()}
    missing = {part["group"] for part in kept if len(part["y"])} - set(tables)
    if missing:
        raise ValueError(f"no attack-free capture for vehicle group(s) {sorted(missing)}")

    normal_weight = totals[TEST_NORMAL] / counts[TEST_NORMAL] if counts[TEST_NORMAL] else 1.0
    x_train, x_test, y_train, y_test, w_test, test_capture = [], [], [], [], [], []
    for part in kept:
        if not len(part["y"]):
            continue
        table = tables[part["group"]]
        x = _features(part["dlc"], part["data"], part["dt"], part["bits"],
                      table.lookup(part["stream"]), table.lookup_mean_bits(part["stream"]))
        is_train = part["pool"] <= TRAIN_ATTACK
        x_train.append(x[is_train]); y_train.append(part["y"][is_train])
        x_test.append(x[~is_train]); y_test.append(part["y"][~is_train])
        w_test.append(np.where(part["pool"][~is_train] == TEST_NORMAL, normal_weight, 1.0).astype(np.float32))
        test_capture.append(np.full(int((~is_train).sum()), part["index"], dtype=np.int16))
    del kept
    empty = np.zeros((0, NUM_FEATURES), dtype=np.float32)
    return FeatureSplit(
        x_train=np.concatenate(x_train) if x_train else empty,
        y_train=np.concatenate(y_train).astype(np.int64) if y_train else np.zeros(0, np.int64),
        x_test=np.concatenate(x_test) if x_test else empty,
        y_test=np.concatenate(y_test).astype(np.int64) if y_test else np.zeros(0, np.int64),
        w_test=np.concatenate(w_test) if w_test else np.zeros(0, np.float32),
        tables=tables,
        test_capture=np.concatenate(test_capture) if test_capture else np.zeros(0, np.int16),
    )


def period_table_arrays(tables: dict) -> dict:
    """Period tables as plain arrays, for storing next to a cached split."""
    groups = sorted(tables)
    return {
        "period_group": np.concatenate([[g] * len(tables[g].ids) for g in groups]).astype(str),
        "period_can_id": np.concatenate([tables[g].ids for g in groups]) // FRAME_KINDS,
        "period_frame_kind": np.concatenate([tables[g].ids for g in groups]) % FRAME_KINDS,
        "period_seconds": np.concatenate([tables[g].periods for g in groups]),
        "period_fallback_group": np.array(groups, dtype=str),
        # one column per frame kind: data, remote, remote reply
        "period_fallback_seconds": np.stack([tables[g].fallbacks for g in groups]),
        "period_mean_bits": np.concatenate([tables[g].mean_bits for g in groups]),
        "period_fallback_mean_bits": np.stack([tables[g].bits_fallbacks for g in groups]),
        "feature_names": np.array(FEATURE_NAMES, dtype=str),
    }


def scale_features(x: np.ndarray) -> np.ndarray:
    """The fixed scaling (``FEATURE_CENTER`` / ``FEATURE_SCALE``)."""
    return ((x - FEATURE_CENTER) / FEATURE_SCALE).astype(np.float32)


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # Conv1D expects (samples, timesteps, channels)
        return x[:, :, np.newaxis].astype(np.float32)
    return x.astype(np.float32)


# ---------------------------------------------------------------- caching
# A cache is split_cache/<dataset_key>.npz, e.g. split_cache/road_v1.npz. The
# key (dataset name + version) is the only thing that identifies it, so bump
# the version (v1 -> v2) whenever the rows, labels, features or sampling
# change; otherwise an old cache would be read as if it were current. The
# test size, random state and raw-file signature it was built with are stored
# inside (``cache_metadata``) and checked when it is found.


def cache_file_path(dataset_key: str, base_dir: str) -> str:
    return os.path.join(base_dir, "split_cache", f"{dataset_key}.npz")


def cache_metadata(source_version: int, test_size: float, random_state: int) -> dict:
    """Arrays stored in every cache: what it was built with."""
    return {"cache_test_size": np.array(test_size), "cache_random_state": np.array(random_state),
            "cache_source_version": np.array(str(source_version))}


def cache_split(
    dataset_key: str,
    base_dir: str,
    split: FeatureSplit,
    source_version: int,
    test_size: float = 0.33,
    random_state: int = 41,
    extra_arrays: dict = None,
):
    """Scale ``split`` with the fixed scaling (``scale_features``) and cache it.

    The file holds the client's training rows (``x_train``/``y_train``,
    already sampled to its final size) and the server's test rows with
    their weights (``x_test``/``y_test``/``w_test``). ``scaler_mean`` /
    ``scaler_scale`` are the fixed scaling's center and scale (raw = x *
    scale + mean). ``extra_arrays`` (e.g. the ID period tables) are stored
    alongside for reference. Returns ``(x_train, x_test, y_train, y_test)`` scaled.
    """
    x_train = scale_features(split.x_train)
    x_test = scale_features(split.x_test)

    cache_path = cache_file_path(dataset_key, base_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path, x_train=x_train, y_train=split.y_train, x_test=x_test, y_test=split.y_test,
        w_test=split.w_test, test_capture=split.test_capture,
        scaler_mean=FEATURE_CENTER, scaler_scale=FEATURE_SCALE,
        **cache_metadata(source_version, test_size, random_state), **(extra_arrays or {}),
    )
    return x_train, x_test, split.y_train, split.y_test


def directory_signature(paths: Iterable[str]) -> int:
    """Combine the mtimes of a set of files into one integer: the raw files'
    "source version", stored inside each cache for reference (a cache is
    found by its key alone, see ``cache_file_path``)."""
    total = 0
    for path in paths:
        if os.path.exists(path):
            total += os.stat(path).st_mtime_ns
    return total


def cached_split_path(dataset_key: str, base_dir: str, test_size: float = 0.33, random_state: int = 41):
    """Path of the cached split for ``dataset_key``, or None.

    Found by the key alone: a Pi or the Jetson has only ``split_cache/*.npz``
    copied onto it, not the (often multi-GB) raw files, so the cache cannot
    be checked against them -- ``dataset_key`` must change whenever the rows,
    labels, features or sampling change. A cache built with another test
    size or random state is an error, not a silent mismatch.
    """
    path = cache_file_path(dataset_key, base_dir)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        built = (float(data["cache_test_size"]), int(data["cache_random_state"]))
    if built != (float(test_size), int(random_state)):
        raise ValueError(f"{path} was built with test_size, random_state = {built}, "
                         f"not {(test_size, random_state)}: rebuild it or use another key")
    return path


def load_cached(dataset_key: str, base_dir: str, names, test_size: float = 0.33, random_state: int = 41):
    """Only the requested arrays from the cache (an .npz reads members on
    access, so a Pi never loads the test rows nor the server the train
    rows). None if there is no cache."""
    path = cached_split_path(dataset_key, base_dir, test_size, random_state)
    if path is None:
        return None
    with np.load(path) as data:
        return tuple(data[name] for name in names)


def find_any_cached_split(dataset_key: str, base_dir: str, test_size: float = 0.33, random_state: int = 41):
    """``(x_train, x_test, y_train, y_test)`` from the cache, or None."""
    return load_cached(dataset_key, base_dir, ("x_train", "x_test", "y_train", "y_test"), test_size, random_state)


def local_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 41,
):
    return train_test_split(
        x,
        y,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
