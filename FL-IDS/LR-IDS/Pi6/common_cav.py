#!/usr/bin/env python
# coding: utf-8

"""Dataset-agnostic helpers shared by every ``*_utils_cav.py`` module.

Each per-dataset module is responsible for parsing its own raw files into a
common feature schema: ``[CAN_ID, DLC, DATA0, DATA1, ..., DATA7]`` (10
numeric features) plus a binary label (0 = Normal, 1 = Attack). Everything
downstream of that (train/test caching, scaling, local train/val split) is
identical across datasets and lives here so it isn't copy-pasted six times.
"""

import glob
import os
import re
import urllib.parse
import urllib.request
import zipfile

from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_TO_INT = {"Normal": 0, "ATTACK": 1}

NUM_DATA_BYTES = 8


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

_CANDUMP_RE = re.compile(r"\((?P<ts>\d+\.\d+)\)\s+\S+\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)")


def parse_candump_line(line: str):
    """Parse one ``can-utils candump`` log line: ``(ts) can0 ID#HEXDATA``.

    Returns ``(timestamp, can_id, dlc, data_bytes)`` or ``None`` if the line
    doesn't match (used by the ROAD and CAN-MIRGU loaders, both of which
    ship raw captures in this format).
    """
    match = _CANDUMP_RE.search(line)
    if not match:
        return None
    timestamp = float(match.group("ts"))
    can_id = int(match.group("id"), 16)
    data_hex = match.group("data")
    data_bytes = [int(data_hex[i:i + 2], 16) for i in range(0, len(data_hex), 2)]
    return timestamp, can_id, len(data_bytes), data_bytes


def parse_candump_file(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_candump_line(line)
            if parsed:
                rows.append(parsed)
    return rows


def iter_candump_file(path: str, stride: int = 1):
    """Like ``parse_candump_file``, but a generator, and optionally
    sub-sampling every ``stride``-th matched frame instead of every one.

    Needed for very large captures (e.g. CAN-MIRGU's ~127M total lines)
    where collecting every row into a Python list first -- as
    ``parse_candump_file`` does -- balloons memory far past what the final
    numpy arrays need (observed: a single client's raw parse alone grew
    past 7GB and was still climbing before being killed). Filtering while
    streaming keeps peak memory proportional to the *sampled* row count.
    """
    counter = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_candump_line(line)
            if not parsed:
                continue
            if counter % stride == 0:
                yield parsed
            counter += 1


class GrowableArray:
    """A numpy array that grows by doubling, for building large feature/label
    matrices from a line-by-line parse without the per-element overhead of
    accumulating Python lists first (a Python list of millions of small
    lists/tuples uses several times more memory than the numpy array it's
    destined to become).
    """

    def __init__(self, num_cols: int, dtype=np.float32, initial_capacity: int = 1 << 16):
        self._data = np.empty((initial_capacity, num_cols), dtype=dtype)
        self._size = 0

    def append(self, row) -> None:
        if self._size == self._data.shape[0]:
            self._grow()
        self._data[self._size] = row
        self._size += 1

    def _grow(self) -> None:
        new_data = np.empty((self._data.shape[0] * 2, self._data.shape[1]), dtype=self._data.dtype)
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data

    def __len__(self) -> int:
        return self._size

    def finalize(self) -> np.ndarray:
        """Trim to the actual size and release the oversized backing buffer."""
        result = self._data[: self._size].copy()
        self._data = None
        return result


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # Conv1D expects (samples, timesteps, channels)
        return x[:, :, np.newaxis].astype(np.float32)
    return x.astype(np.float32)


def build_frame_features(dlc: int, data_bytes: Sequence[int]) -> list:
    """Build one ``[DLC, DATA0..DATA7]`` feature row.

    CAN ID is deliberately not a feature: the OTIDS DoS/Impersonation labels
    are defined on it, so including it lets the model read the label off the
    input. ``data_bytes`` shorter than 8 is zero-padded.
    """
    padded = list(data_bytes[:NUM_DATA_BYTES])
    padded += [0] * (NUM_DATA_BYTES - len(padded))
    return [dlc] + padded


def split_time_blocks(n_rows: int, test_blocks=(2, 5, 8), n_blocks: int = 10, buffer_frac: float = 0.005):
    """Split one time-ordered capture into contiguous blocks.

    Blocks listed in ``test_blocks`` go to test. Train blocks give up
    ``buffer_frac`` of their rows at each end so frames next to a boundary
    never land on the other side. Returns row indices (train, test).
    """
    idx = np.arange(n_rows)
    buf = int(n_rows * buffer_frac)
    train, test = [], []
    for b, block in enumerate(np.array_split(idx, n_blocks)):
        if b in test_blocks:
            test.append(block)
        else:
            train.append(block[buf:len(block) - buf] if len(block) > 2 * buf else block)
    return np.concatenate(train), np.concatenate(test)


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
):
    """Scale a split whose train/test membership was decided by the caller
    (e.g. time blocks or whole files) and cache it. The scaler is fit on the
    train side only."""
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    cache_dir = os.path.join(base_dir, "split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{dataset_key}_ts{test_size}_rs{random_state}_src{source_version}.npz",
    )
    np.savez(cache_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
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


def find_any_cached_split(
    dataset_key: str,
    base_dir: str,
    test_size: float = 0.33,
    random_state: int = 41,
):
    """Load a cached split for ``dataset_key`` if one exists, regardless of
    the raw source files' mtimes (i.e. without needing the raw dataset
    present at all).

    ``cache_scaled_split`` normally keys its cache filename on a
    ``source_version`` signature computed from the raw files' mtimes, which
    means using the cache still requires the (often multi-GB) raw dataset to
    be present locally just to recompute that signature -- pointless once
    the cache has already been built once (e.g. on a dev machine) and only
    the small ``split_cache/*.npz`` file is copied to a disk-constrained
    device like a Jetson. This instead globs for *any* cache file matching
    this dataset/test_size/random_state, ignoring the source suffix, and
    returns it if found -- callers should call this first and only fall
    back to ensuring the raw files exist if it returns ``None``.
    """
    import glob as _glob

    pattern = os.path.join(base_dir, "split_cache", f"{dataset_key}_ts{test_size}_rs{random_state}_src*.npz")
    matches = sorted(_glob.glob(pattern))
    if not matches:
        return None
    data = np.load(matches[0])
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"]


def cache_scaled_split(
    dataset_key: str,
    load_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    source_version: int,
    base_dir: str,
    test_size: float = 0.33,
    random_state: int = 41,
):
    """Compute (or load from cache) a leak-safe scaled train/test split.

    The scaler is fit on the train portion only. The split+scale result is
    cached to ``<base_dir>/split_cache/<dataset_key>_...npz`` keyed by
    ``source_version`` so repeated client/server processes share the same
    split without recomputation or re-parsing large raw files.
    """
    cache_dir = os.path.join(base_dir, "split_cache")
    cache_path = os.path.join(
        cache_dir,
        f"{dataset_key}_ts{test_size}_rs{random_state}_src{source_version}.npz",
    )

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["x_train"], data["x_test"], data["y_train"], data["y_test"]

    x, y = load_fn()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y))

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit scaler on train only — no leakage
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    return x_train, x_test, y_train, y_test


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
