# src/data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class HARDataset(Dataset):
    """
    Simple wrapper for HAR data.

    X: (N, D) or (N, T, C)
    y: (N,)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   # e.g. shape (561,) or (T, C)
        y = self.y[idx]

        # Case 1: flat 561-dim features → make (1, 561) for Conv1d
        if x.ndim == 1:
            x = x[None, :]      # (D,) -> (1, D)

        # Case 2: already (T, C) (for later raw-signal experiments)
        elif x.ndim == 2:
            x = x.T             # (T, C) -> (C, T)

        return torch.from_numpy(x), torch.tensor(y)


def _load_X_y(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    root: path to UCI_HAR_Dataset
    split: "train" or "test"
    """
    X_path = os.path.join(root, split, f"X_{split}.txt")
    y_path = os.path.join(root, split, f"y_{split}.txt")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"{X_path} not found.")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"{y_path} not found.")

    X = np.loadtxt(X_path)
    y = np.loadtxt(y_path).astype(int) - 1  # 1..6 → 0..5
    return X, y


def load_har(data_root: str):
    dataset_dir = os.path.join(data_root, "UCI_HAR_Dataset")
    print("dataset_dir:", dataset_dir)  # temporary debug

    X_train, y_train = _load_X_y(dataset_dir, "train")
    X_test,  y_test  = _load_X_y(dataset_dir, "test")

    return X_train, y_train, X_test, y_test


def make_train_val_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split original train into train/val; keep test as given.
    """
    rng = np.random.RandomState(seed)
    n = len(X_train)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_te, y_te = X_test, y_test
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def make_dataloaders(
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    batch_size: int = 128,
    num_workers: int = 0,
):
    train_ds = HARDataset(X_tr, y_tr)
    val_ds = HARDataset(X_val, y_val)
    test_ds = HARDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader
