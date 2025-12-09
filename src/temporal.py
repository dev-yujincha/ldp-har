# src/temporal.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def detect_bouts(
    user_ids: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    max_gap: int = 1,
) -> np.ndarray:
    """
    Simple bout detection using user + label + time contiguity.

    A new bout starts when:
      - user changes, or
      - label changes, or
      - time gap between consecutive windows > max_gap.

    Args:
        user_ids: (N,) array of user / subject ids.
        labels: (N,) array of activity labels.
        timestamps: (N,) array of (monotone) time indices.
        max_gap: maximum allowed gap (in window units) inside one bout.

    Returns:
        bout_ids: (N,) array of integer bout ids in [0, B-1].
    """
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps)
    N = len(user_ids)
    assert labels.shape[0] == N and timestamps.shape[0] == N

    bout_ids = np.zeros(N, dtype=np.int64)
    curr_bout = 0
    bout_ids[0] = curr_bout

    for i in range(1, N):
        new_bout = (
            user_ids[i] != user_ids[i - 1]
            or labels[i] != labels[i - 1]
            or (timestamps[i] - timestamps[i - 1]) > max_gap
        )
        if new_bout:
            curr_bout += 1
        bout_ids[i] = curr_bout

    return bout_ids


def shuffle_within_bouts(
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    timestamps: np.ndarray,
    bout_ids: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """
    Locally shuffle the window order inside each bout.

    Args:
        X, y, user_ids, timestamps, bout_ids: all (N,) aligned.
        rng: optional Generator for reproducibility.

    Returns:
        X_shuf, y_shuf, user_shuf, ts_shuf, bout_shuf, order

        where 'order' is the permutation applied to the original arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(bout_ids)
    idx = np.arange(N)

    permuted_indices = []
    for b in np.unique(bout_ids):
        bout_idx = idx[bout_ids == b]
        rng.shuffle(bout_idx)
        permuted_indices.append(bout_idx)

    order = np.concatenate(permuted_indices, axis=0)

    X_shuf = X[order]
    y_shuf = y[order]
    user_shuf = user_ids[order]
    ts_shuf = timestamps[order]
    bout_shuf = bout_ids[order]

    return X_shuf, y_shuf, user_shuf, ts_shuf, bout_shuf, order


def assign_pseudonyms(
    bout_ids: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Rotate pseudonyms at the bout level.

    Each bout gets a random pseudonym id, and all windows
    in that bout share the same pseudonym. This breaks
    long-term linkability across bouts for the same user.

    Args:
        bout_ids: (N,) integer bout ids.

    Returns:
        pseudo_ids: (N,) integer pseudonym ids.
        bout_to_pseudo: mapping from bout id -> pseudonym id.
    """
    if rng is None:
        rng = np.random.default_rng()

    bout_ids = np.asarray(bout_ids)
    uniq_bouts = np.unique(bout_ids)
    B = len(uniq_bouts)

    # Random permutation of pseudonym labels
    pseudo_labels = rng.permutation(B)

    bout_to_pseudo = {
        int(b): int(pseudo_labels[i]) for i, b in enumerate(uniq_bouts)
    }
    pseudo_ids = np.array(
        [bout_to_pseudo[int(b)] for b in bout_ids], dtype=np.int64
    )

    return pseudo_ids, bout_to_pseudo


def perturb_timestamps(
    timestamps: np.ndarray,
    bout_ids: np.ndarray,
    bin_size: float = 60.0,
    jitter_std: float = 15.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Coarsen and perturb timestamps at the bout level.

    For each bout:
      1) Take the true bout start time.
      2) Snap to nearest time bin of width 'bin_size'.
      3) Add Gaussian jitter with std 'jitter_std'.
      4) Inside the bout, preserve only *relative* offsets
         from this noisy start.

    Args:
        timestamps: (N,) array of original timestamps (float or int).
        bout_ids: (N,) array of bout ids.
        bin_size: size of coarse time bins (e.g., seconds or minutes).
        jitter_std: std of additive gaussian noise on bout start.
        rng: optional Generator.

    Returns:
        new_ts: (N,) array of perturbed timestamps.
    """
    if rng is None:
        rng = np.random.default_rng()

    timestamps = np.asarray(timestamps, dtype=float)
    bout_ids = np.asarray(bout_ids)
    new_ts = np.zeros_like(timestamps, dtype=float)

    uniq_bouts = np.unique(bout_ids)
    for b in uniq_bouts:
        mask = bout_ids == b
        ts_b = timestamps[mask]
        t0 = ts_b.min()

        # coarse binning of start time
        t0_bin = bin_size * np.round(t0 / bin_size)

        # jitter the bin center
        jitter = rng.normal(loc=0.0, scale=jitter_std)
        noisy_start = t0_bin + jitter

        # keep only relative structure within the bout
        new_ts[mask] = noisy_start + (ts_b - t0)

    return new_ts


def build_temporal_view(
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    timestamps: np.ndarray,
    max_gap: int = 1,
    bin_size: float = 60.0,
    jitter_std: float = 15.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full Step-5 pipeline: bouts + shuffling + pseudonyms + timestamp perturbation.

    Args:
        X, y, user_ids, timestamps: input arrays (N, ...).
            Assumed sorted by (user_ids, timestamps).
        max_gap: max allowed time gap inside a bout.
        bin_size: coarse time bin size for bout start times.
        jitter_std: std of timestamp jitter.
        seed: random seed.

    Returns:
        X_shuf, y_shuf: shuffled feature/label arrays.
        meta_df: pandas DataFrame with columns:
            - 'orig_idx': original index before shuffling
            - 'user': original user id
            - 'pseudo': pseudonym id (rotated per bout)
            - 'bout': bout id
            - 't': perturbed timestamp
            - 'label': activity label
    """
    rng = np.random.default_rng(seed)

    # 1) bout detection
    bout_ids = detect_bouts(
        user_ids=user_ids, labels=y, timestamps=timestamps, max_gap=max_gap
    )

    # 2) shuffle within bouts
    X_shuf, y_shuf, user_shuf, ts_shuf, bout_shuf, order = shuffle_within_bouts(
        X=X,
        y=y,
        user_ids=user_ids,
        timestamps=timestamps,
        bout_ids=bout_ids,
        rng=rng,
    )

    # 3) pseudonym rotation per bout
    pseudo_ids, bout_to_pseudo = assign_pseudonyms(bout_shuf, rng=rng)

    # 4) timestamp perturbation
    ts_pert = perturb_timestamps(
        timestamps=ts_shuf,
        bout_ids=bout_shuf,
        bin_size=bin_size,
        jitter_std=jitter_std,
        rng=rng,
    )

    meta_df = pd.DataFrame(
        {
            "orig_idx": order,
            "user": user_shuf,
            "pseudo": pseudo_ids,
            "bout": bout_shuf,
            "t": ts_pert,
            "label": y_shuf,
        }
    )

    return X_shuf, y_shuf, meta_df
