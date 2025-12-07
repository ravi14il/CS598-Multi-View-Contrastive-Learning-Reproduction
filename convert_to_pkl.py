import os
import pickle

import numpy as np
import torch

TFC_ROOT = "./data/"
DATASET_DIRS = {
    "SleepEEG": "SleepEEG",
    "ECG": "ECG",
    "Epilepsy": "Epilepsy",
    "FD-B": "FD-B",
    "Gesture": "Gesture",
    "EMG": "EMG",
}
OUT_DIR = "data/Domain_ts"
os.makedirs(OUT_DIR, exist_ok=True)

def load_split(pt_path):
    d = torch.load(pt_path, map_location="cpu")
    X = d["samples"].numpy()  # shape: (N, C, L)
    y = d["labels"].numpy()   # shape: (N,)
    return X, y

for data_name, folder in DATASET_DIRS.items():
    folder_path = os.path.join(TFC_ROOT, folder)

    train_pt = os.path.join(folder_path, "train.pt")
    val_pt   = os.path.join(folder_path, "val.pt")
    test_pt  = os.path.join(folder_path, "test.pt")

    if not os.path.exists(train_pt):
        print(f"Skipping {data_name}: {train_pt} not found")
        continue

    print(f"Converting {data_name} from {folder_path}")

    X_train, y_train = load_split(train_pt)
    X_val,   y_val   = load_split(val_pt)
    X_test,  y_test  = load_split(test_pt)

    out_path = os.path.join(OUT_DIR, f"{data_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump([X_train, X_val, X_test,
                     y_train, y_val, y_test], f)

    print(f"  -> wrote {out_path}")
