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
- ``bits_changed``: data bits that differ from the previous frame of the stream.

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
- The period table comes only from the train part of attack-free captures (a
  capture-level fact from the dataset documentation, not per-frame labels).
  A stream missing from it gets the lower median period of the table's
  streams of the same kind, which is the period of one real stream of that
  kind, so "ID never seen in attack-free traffic" is not encoded.
"""

import glob
import os
import re
import urllib.parse
import urllib.request
import zipfile

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_TO_INT = {"Normal": 0, "ATTACK": 1}

NUM_DATA_BYTES = 8
FEATURE_NAMES = (
    ["DLC"] + [f"DATA{i}" for i in range(NUM_DATA_BYTES)]
    + ["log_dt", "log_dt_over_period", "bits_changed"]
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


@dataclass
class RowSampling:
    """Fraction of rows kept, drawn after the features are computed:
    train normal rows, train attack rows, and test rows (any label)."""
    train_normal: float = 1.0
    train_attack: float = 1.0
    test: float = 1.0


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
    """Median time between frames of each stream (``frame_streams``) in
    attack-free traffic. A stream missing from the table gets
    ``fallbacks[kind]`` for its frame kind."""
    ids: np.ndarray
    periods: np.ndarray
    fallbacks: np.ndarray

    @property
    def fallback(self) -> float:
        return float(self.fallbacks[DATA_FRAME])

    def lookup(self, stream: np.ndarray) -> np.ndarray:
        pos = np.clip(np.searchsorted(self.ids, stream), 0, max(len(self.ids) - 1, 0))
        found = self.ids[pos] == stream
        return np.where(found, self.periods[pos], self.fallbacks[stream % FRAME_KINDS])


def _lower_median(values: np.ndarray) -> float:
    """Always one of ``values``: np.median would average the two middle
    values for an even count, giving a period no real stream has."""
    return float(np.sort(values)[(len(values) - 1) // 2])


def _period_table(parts: list) -> PeriodTable:
    ids = np.concatenate([p[0] for p in parts])
    dts = np.concatenate([p[1] for p in parts])
    if len(ids) == 0:
        raise ValueError("no attack-free train frames to build the ID period table from")
    medians = pd.Series(dts).groupby(ids).median()
    streams = medians.index.to_numpy(dtype=np.int64)
    periods = medians.to_numpy(dtype=np.float64)
    # Fallbacks come from streams of the same kind, so a missing data stream
    # looks like some real data stream (and not, say, like the remote-frame
    # streams, which a data frame never is).
    kinds = streams % FRAME_KINDS
    if not np.any(kinds == DATA_FRAME):
        raise ValueError("no data frames in the attack-free train frames")
    data_fallback = _lower_median(periods[kinds == DATA_FRAME])
    fallbacks = np.array([
        _lower_median(periods[kinds == kind]) if np.any(kinds == kind) else data_fallback
        for kind in range(FRAME_KINDS)
    ])
    return PeriodTable(ids=streams, periods=periods, fallbacks=fallbacks)


def _features(dlc, data, dt, bits, period):
    first = np.isnan(dt)
    dt = np.where(first, FIRST_FRAME_DT, dt)
    ratio = np.where(first, 1.0, np.maximum(dt, MIN_DT) / period)
    return np.column_stack([
        dlc, data,
        np.log10(np.clip(dt, MIN_DT, FIRST_FRAME_DT)),
        np.log10(np.clip(ratio, *RATIO_RANGE)),
        bits,
    ]).astype(np.float32)


def build_feature_split(captures: Iterable[Capture], sampling: RowSampling, random_state: int = 41, log=print,
                        return_test_capture: bool = False):
    """Load every capture once, compute the 12 features, sample rows.

    Returns ``(x_train, x_test, y_train, y_test, period_tables)`` with
    ``period_tables`` a {group: PeriodTable} dict; with
    ``return_test_capture`` also, per test row, the index of its capture in
    ``captures``.
    """
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
                (stream[usable], dt[usable].astype(np.float32)))

        rng = np.random.default_rng([random_state, index])
        u = rng.random(n)
        attack = frames.y == 1
        keep = (
            (train & ~attack & (u < sampling.train_normal))
            | (train & attack & (u < sampling.train_attack))
            | (test & (u < sampling.test))
        )
        kept.append({
            "index": index, "group": capture.group, "train": train[keep], "stream": stream[keep],
            "dlc": frames.dlc[keep], "data": frames.data[keep], "dt": dt[keep],
            "bits": bits[keep], "y": frames.y[keep],
        })
        replies = int((stream % FRAME_KINDS == REMOTE_REPLY).sum())
        log(f"[DATA] {capture.name}: {n} frames ({int(attack.sum())} attack, "
            f"{int(frames.rtr.sum())} remote, {replies} remote replies), "
            f"{int(recording_gaps.sum())} recording gaps, "
            f"kept {int((keep & train).sum())} train / {int((keep & test).sum())} test")
        del frames, dt, bits

    tables = {group: _period_table(parts) for group, parts in baseline_parts.items()}
    missing = {part["group"] for part in kept} - set(tables)
    if missing:
        raise ValueError(f"no attack-free capture for vehicle group(s) {sorted(missing)}")

    x_train, x_test, y_train, y_test, test_capture = [], [], [], [], []
    for part in kept:
        x = _features(part["dlc"], part["data"], part["dt"], part["bits"], tables[part["group"]].lookup(part["stream"]))
        x_train.append(x[part["train"]])
        x_test.append(x[~part["train"]])
        y_train.append(part["y"][part["train"]])
        y_test.append(part["y"][~part["train"]])
        test_capture.append(np.full(int((~part["train"]).sum()), part["index"], dtype=np.int16))
    del kept
    result = (
        np.concatenate(x_train), np.concatenate(x_test),
        np.concatenate(y_train).astype(np.int64), np.concatenate(y_test).astype(np.int64),
        tables,
    )
    return result + (np.concatenate(test_capture),) if return_test_capture else result


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
        "feature_names": np.array(FEATURE_NAMES, dtype=str),
    }


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # Conv1D expects (samples, timesteps, channels)
        return x[:, :, np.newaxis].astype(np.float32)
    return x.astype(np.float32)


# ---------------------------------------------------------------- caching
def cache_precomputed_split(
    dataset_key: str,
    base_dir: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    source_version: int,
    test_size: float = 0.33,
    random_state: int = 41,
    extra_arrays: dict = None,
):
    """Scale a split whose train/test membership was decided by the caller
    (e.g. time blocks or whole files) and cache it. The scaler is fit on the
    train side only. ``extra_arrays`` (e.g. the ID period tables) are stored
    alongside for reference; nothing reads them back at training time."""
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    cache_dir = os.path.join(base_dir, "split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{dataset_key}_ts{test_size}_rs{random_state}_src{source_version}.npz",
    )
    np.savez(
        cache_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, **(extra_arrays or {}),
    )
    return x_train, x_test, y_train, y_test


def directory_signature(paths: Iterable[str]) -> int:
    """Combine the mtimes of a set of files into one cache-key integer.

    Used as the "source version" for split caching so a cache is invalidated
    automatically if any of a dataset's underlying files change.
    """
    total = 0
    for path in paths:
        if os.path.exists(path):
            total += os.stat(path).st_mtime_ns
    return total


def _read_exact(f, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        data = f.read(remaining)
        if not data:
            raise EOFError("unexpected end of cached array data")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _iter_npy_rows(zf: zipfile.ZipFile, member: str, chunk_rows: int):
    """Yield an ``.npy`` array stored inside an ``.npz`` in row chunks,
    without loading the whole array into memory."""
    with zf.open(member) as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
        if fortran_order:
            raise ValueError(f"{member} is Fortran-ordered; expected row-major")
        n_rows = shape[0]
        row_shape = tuple(shape[1:])
        row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
        done = 0
        while done < n_rows:
            take = min(chunk_rows, n_rows - done)
            buf = _read_exact(f, take * row_bytes)
            yield np.frombuffer(buf, dtype=dtype).reshape((take,) + row_shape)
            done += take


def load_cached_train_fraction(
    dataset_key: str,
    base_dir: str,
    fraction: float,
    test_size: float = 0.33,
    random_state: int = 41,
    chunk_rows: int = 1_000_000,
):
    """Return a uniform random ``fraction`` of the cached training pool,
    reading only ``x_train``/``y_train`` and streaming in chunks so the full
    pool (and the test arrays) never sit in memory at once.

    Every row is kept with the same probability, so each class keeps
    (in expectation) its original share -- stratified sampling in effect.
    Returns ``None`` if no cache file exists for this dataset.
    """
    pattern = os.path.join(
        base_dir, "split_cache",
        f"{dataset_key}_ts{test_size}_rs{random_state}_src*.npz",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    rng = np.random.default_rng(random_state)
    x_parts = []
    y_parts = []
    with zipfile.ZipFile(matches[0]) as zf:
        x_chunks = _iter_npy_rows(zf, "x_train.npy", chunk_rows)
        y_chunks = _iter_npy_rows(zf, "y_train.npy", chunk_rows)
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            keep = rng.random(len(x_chunk)) < fraction
            x_parts.append(x_chunk[keep])
            y_parts.append(y_chunk[keep])
    return np.concatenate(x_parts), np.concatenate(y_parts)


def subsample_rows(x: np.ndarray, y: np.ndarray, fraction: float, random_state: int = 41):
    """Same uniform-rate subsampling as ``load_cached_train_fraction``, for
    data that is already fully in memory (fallback when no cache exists)."""
    rng = np.random.default_rng(random_state)
    keep = rng.random(len(x)) < fraction
    return x[keep], y[keep]


def cached_split_path(dataset_key: str, base_dir: str, test_size: float = 0.33, random_state: int = 41):
    """Path of the cached split for ``dataset_key`` (any source version), or None."""
    pattern = os.path.join(base_dir, "split_cache", f"{dataset_key}_ts{test_size}_rs{random_state}_src*.npz")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def find_any_cached_split(
    dataset_key: str,
    base_dir: str,
    test_size: float = 0.33,
    random_state: int = 41,
):
    """Load a cached split for ``dataset_key`` if one exists, regardless of
    the raw source files' mtimes (i.e. without needing the raw dataset
    present at all).

    The cache filename carries a ``source_version`` signature computed from
    the raw files' mtimes; recomputing it would require the (often multi-GB)
    raw dataset just to find the cache, which is pointless on a
    disk-constrained device like a Jetson that only has the small
    ``split_cache/*.npz`` copied onto it. This instead globs for *any* cache
    file matching this dataset/test_size/random_state, ignoring the source
    suffix -- so ``dataset_key`` must change whenever the rows, labels or
    features change. Callers should call this first and only fall back to
    the raw files if it returns ``None``.
    """
    pattern = os.path.join(base_dir, "split_cache", f"{dataset_key}_ts{test_size}_rs{random_state}_src*.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with np.load(matches[0]) as data:
        return data["x_train"], data["x_test"], data["y_train"], data["y_test"]


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
