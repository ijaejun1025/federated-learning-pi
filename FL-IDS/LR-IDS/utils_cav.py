#!/usr/bin/env python
# coding: utf-8

import datetime
import glob
import os
import shutil
import urllib.request
import zipfile

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAV_DIR = os.path.join(BASE_DIR, "cav")
ANOMALY_CSV_PATH = os.path.join(BASE_DIR, "cav_anoamly.csv")
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

# Columns that are labels or direct label encodings and must never be used as features.
LABEL_LEAKAGE_COLUMNS = {"AttackType", "intrusion", "Normal", "ATTACK"}


def _find_file_recursively(root_dir: str, file_name: str):
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
        file_name
        for file_name in REQUIRED_CAV_FILES
        if not os.path.exists(os.path.join(CAV_DIR, file_name))
    ]
    if not missing:
        return

    try:
        _download_and_extract_cav_dataset()
    except Exception as error:
        raise FileNotFoundError(
            "CAV raw CSV files were not found, and automatic download also failed. "
            f"Required files: {REQUIRED_CAV_FILES}, error: {error}"
        ) from error

    still_missing = [
        file_name
        for file_name in REQUIRED_CAV_FILES
        if not os.path.exists(os.path.join(CAV_DIR, file_name))
    ]
    if still_missing:
        raise FileNotFoundError(
            "Some files are still missing after automatic download: "
            f"{still_missing}. Please place them manually under {CAV_DIR}."
        )


def changecolumn(file_path, attack_type):
    df = pd.read_csv(file_path).sample(frac=0.05, random_state=20, replace=False).reset_index(drop=True)
    df.columns = [
        "Timestamp",
        "CAN ID",
        "Byte",
        "DATA[0]",
        "DATA[1]",
        "DATA[2]",
        "DATA[3]",
        "DATA[4]",
        "DATA[5]",
        "DATA[6]",
        "DATA[7]",
        "AttackType",
    ]
    df["AttackType"] = np.where(df["AttackType"] == "T", attack_type, "Normal")
    df = df.dropna()
    return df


def changecolumntype(df):
    for column in ["CAN ID", "DATA[0]", "DATA[1]", "DATA[2]", "DATA[3]", "DATA[4]", "DATA[5]", "DATA[6]", "DATA[7]"]:
        df[column] = df[column].apply(lambda x: int(str(x), base=16))
    return df


def _build_cav_anomaly_csv() -> None:
    _ensure_raw_cav_files()

    df_dos = changecolumn(os.path.join(CAV_DIR, "DoS_dataset.csv"), "DoS")
    df_fuzzy = changecolumn(os.path.join(CAV_DIR, "Fuzzy_dataset.csv"), "Fuzzy")
    df_gear = changecolumn(os.path.join(CAV_DIR, "gear_dataset.csv"), "Gear-Spooing")
    df_rpm = changecolumn(os.path.join(CAV_DIR, "RPM_dataset.csv"), "RPM-Spoofing")

    dataset = pd.concat([df_dos, df_fuzzy, df_gear, df_rpm]).dropna()
    dataset = changecolumntype(dataset)

    dateformat = "%Y-%m-%d %H:%M:%S.%f"
    dataset["Timestamp"] = dataset["Timestamp"].apply(
        lambda x: datetime.datetime.fromtimestamp(float(x)).strftime(dateformat)
    )

    bin_label = pd.DataFrame(dataset.AttackType.map(lambda x: "Normal" if x == "Normal" else "ATTACK"))
    bin_data = dataset.copy()
    bin_data["AttackType"] = bin_label

    le1 = preprocessing.LabelEncoder()
    enc_label = bin_label.apply(le1.fit_transform)
    bin_data["intrusion"] = enc_label

    bin_data = pd.get_dummies(bin_data, columns=["AttackType"], prefix="", prefix_sep="")
    bin_data["AttackType"] = bin_label

    bin_data.to_csv(ANOMALY_CSV_PATH, index=False)


def load_cav() -> Tuple[np.ndarray, np.ndarray]:
    """Load CAV dataset without label leakage.

    Returns
    -------
    x : np.ndarray
        Standardized numeric feature matrix excluding all label-related columns.
    y : np.ndarray
        Encoded binary labels (0/1).
    """
    if not os.path.exists(ANOMALY_CSV_PATH):
        _build_cav_anomaly_csv()

    data = pd.read_csv(ANOMALY_CSV_PATH).reset_index(drop=True)

    y = data["AttackType"].values

    feature_cols = [
        col for col in data.columns
        if col not in LABEL_LEAKAGE_COLUMNS and pd.api.types.is_numeric_dtype(data[col])
    ]
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found after removing label leakage columns.")

    X = data[feature_cols].to_numpy(dtype=np.float32)
    x = StandardScaler().fit_transform(X).astype(np.float32)

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int64)
    return x, y


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # Conv1D expects (samples, timesteps, channels)
        return x[:, :, np.newaxis].astype(np.float32)
    return x.astype(np.float32)


def get_global_train_test_split(test_size: float = 0.33, random_state: int = 41):
    x, y = load_cav()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y))
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def get_client_partition(
    client_id: int,
    num_clients: int,
    test_size: float = 0.33,
    random_state: int = 41,
    local_val_size: float = 0.2,
):
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id must be between 0 and {num_clients - 1}, got {client_id}")

    x_train_pool, _, y_train_pool, _ = get_global_train_test_split(
        test_size=test_size,
        random_state=random_state,
    )

    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=random_state)
    folds = list(skf.split(x_train_pool, y_train_pool))
    _, client_indices = folds[client_id]

    x_client = x_train_pool[client_indices]
    y_client = y_train_pool[client_indices]

    x_local_train, x_local_val, y_local_train, y_local_val = train_test_split(
        x_client,
        y_client,
        test_size=local_val_size,
        random_state=random_state + client_id,
        shuffle=True,
        stratify=y_client,
    )

    return x_local_train, x_local_val, y_local_train, y_local_val