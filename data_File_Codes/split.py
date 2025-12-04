"""
sender_split.py  ▸  Create a sender-aware 80/20 split of X.csv and y.csv
"""

import numpy as np
import pandas as pd
import sys
import os

# ---------- CONFIG ----------
X_PATH = "X.csv"         
Y_PATH = "y.csv"
TRAIN_VAL_RATIO = 0.80    # 80 % senders for train/val
RANDOM_STATE = 42         # reproducibility
# ----------------------------

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def load_data(x_path, y_path):
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError("Could not find X.csv and/or y.csv.")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]      # assumes single-column y
    if len(X) != len(y):
        raise ValueError("X and y have different lengths.")
    print_flush(f"Loaded X shape: {X.shape}, y length: {len(y)}")
    return X, y

def sender_aware_split(X, y, ratio=0.8, seed=42):
    senders = X["senderpseudo"].values
    unique_senders = np.unique(senders)
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_senders)
    n_train_senders = int(ratio * len(unique_senders))
    train_senders = shuffled[:n_train_senders]
    test_senders  = shuffled[n_train_senders:]

    train_mask = np.isin(senders, train_senders)
    test_mask  = np.isin(senders, test_senders)

    X_train_val = X[train_mask].reset_index(drop=True)
    y_train_val = y[train_mask].reset_index(drop=True)
    X_test_unseen = X[test_mask].reset_index(drop=True)
    y_test_unseen = y[test_mask].reset_index(drop=True)

    # Sanity check: no sender overlap
    overlap = set(X_train_val["senderpseudo"]).intersection(X_test_unseen["senderpseudo"])
    if overlap:
        raise RuntimeError("Sender leakage detected!")

    print_flush(f"Train/Val senders: {len(train_senders)}  •  "
                f"Test senders: {len(test_senders)}")
    print_flush(f"Train/Val rows: {len(X_train_val)}  •  Test rows: {len(X_test_unseen)}")
    return X_train_val, y_train_val, X_test_unseen, y_test_unseen

def save_splits(X_tv, y_tv, X_test, y_test):
    X_tv.to_csv("X_train_val.csv", index=False)
    y_tv.to_csv("y_train_val.csv", index=False, header=["class"])
    X_test.to_csv("X_test_unseen.csv", index=False)
    y_test.to_csv("y_test_unseen.csv", index=False, header=["class"])
    print_flush("✅ Split files saved:")
    print_flush("   • X_train_val.csv / y_train_val.csv")
    print_flush("   • X_test_unseen.csv / y_test_unseen.csv")

def main():
    X, y = load_data(X_PATH, Y_PATH)
    X_tv, y_tv, X_test, y_test = sender_aware_split(
        X, y, ratio=TRAIN_VAL_RATIO, seed=RANDOM_STATE
    )
    save_splits(X_tv, y_tv, X_test, y_test)

if __name__ == "__main__":
    main()
