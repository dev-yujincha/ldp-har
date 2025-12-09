# src/unlinkability.py
"""Unlinkability attack utilities.

This module implements a very simple but reasonably strong linkage attack
on temporally perturbed HAR data.

Inputs:
    - meta_df: pandas DataFrame produced by src.temporal.build_temporal_view,
      with at least the columns:
        * 'user'   : original user / subject id (for ground truth only)
        * 'bout'   : bout id
        * 't'      : perturbed timestamps
        * 'label'  : activity label (int)

Outputs / main entry point:
    - run_histogram_attack(meta_df, ...) -> (auc, bout_df, scores, labels)

The adversary is assumed to *see* only (pseudo, t, label) but not the true
'user' column. We still use 'user' internally to construct ground-truth
same-user vs different-user labels for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import pandas as pd


@dataclass
class BoutSummary:
    """Per-bout summary statistics used by the attacker."""
    user: int
    bout: int
    n: int
    t_start: float
    t_end: float
    t_mean: float
    t_std: float
    label_hist: np.ndarray  # shape (K,), counts
    label_prob: np.ndarray  # shape (K,), normalized


def _infer_num_labels(meta_df: pd.DataFrame) -> int:
    labels = meta_df["label"].to_numpy()
    return int(labels.max()) + 1


def summarize_bouts(
    meta_df: pd.DataFrame,
    num_labels: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate window-level metadata into bout-level features.

    Returns:
        bout_df: one row per (user, bout) with columns:
            - user, bout, n, t_start, t_end, t_mean, t_std
            - p_label_0, ..., p_label_{K-1}
        label_prob_cols: list of column names for the p_label_* columns.
    """
    if num_labels is None:
        num_labels = _infer_num_labels(meta_df)

    records = []
    for (user, bout), g in meta_df.groupby(["user", "bout"], sort=True):
        ts = g["t"].to_numpy(dtype=float)
        ys = g["label"].to_numpy(dtype=int)

        n = len(g)
        t_start = float(ts.min())
        t_end = float(ts.max())
        t_mean = float(ts.mean())
        t_std = float(ts.std()) if n > 1 else 0.0

        hist = np.bincount(ys, minlength=num_labels).astype(float)
        if hist.sum() > 0:
            prob = hist / hist.sum()
        else:
            prob = np.ones(num_labels, dtype=float) / num_labels

        rec: Dict[str, float] = {
            "user": int(user),
            "bout": int(bout),
            "n": int(n),
            "t_start": t_start,
            "t_end": t_end,
            "t_mean": t_mean,
            "t_std": t_std,
        }
        for k in range(num_labels):
            rec[f"p_label_{k}"] = prob[k]
        records.append(rec)

    bout_df = pd.DataFrame.from_records(records)
    label_prob_cols = [f"p_label_{k}" for k in range(num_labels)]
    return bout_df, label_prob_cols


def _similarity_scores_for_pairs(
    F: np.ndarray,
    pairs: np.ndarray,
    metric: str = "dot",
) -> np.ndarray:
    """Compute similarity scores for a list of (i, j) bout index pairs.

    Args:
        F: (B, D) feature matrix.
        pairs: (M, 2) array of bout indices.
        metric: 'dot' for dot product, or 'neg_l1' for negative L1 distance.

    Returns:
        scores: (M,) similarity scores, higher = more similar.
    """
    v1 = F[pairs[:, 0]]
    v2 = F[pairs[:, 1]]

    if metric == "dot":
        scores = np.einsum("ij,ij->i", v1, v2)
    elif metric == "neg_l1":
        scores = -np.abs(v1 - v2).sum(axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return scores


def build_pair_dataset(
    bout_df: pd.DataFrame,
    label_prob_cols: Sequence[str],
    max_pos_pairs_per_user: int = 50,
    seed: int = 0,
    metric: str = "dot",
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct pairwise scores and labels for the linkage attack.

    For each user with >= 2 bouts, we:
        - enumerate all positive pairs (two bouts from same user),
        - optionally subsample up to max_pos_pairs_per_user per user,
        - for each positive pair, sample one negative pair where the
          second bout comes from a different user.

    Features for the attacker are the per-bout label distributions;
    scores are either dot-product similarity or negative L1 distance.

    Args:
        bout_df: output of summarize_bouts.
        label_prob_cols: list of p_label_* column names.
        max_pos_pairs_per_user: cap on positive pairs per user to control size.
        seed: RNG seed for reproducibility.
        metric: similarity metric, 'dot' or 'neg_l1'.

    Returns:
        scores: (M,) similarity scores.
        labels: (M,) binary ground-truth labels (1 = same user, 0 = different users).
    """
    rng = np.random.default_rng(seed)

    users = bout_df["user"].to_numpy()
    uniq_users = np.unique(users)
    B = len(bout_df)

    # Feature matrix for bouts, attacker only sees these
    F = bout_df.loc[:, label_prob_cols].to_numpy(dtype=float)

    # Precompute indices per user
    user_to_indices: Dict[int, np.ndarray] = {}
    for u in uniq_users:
        user_to_indices[int(u)] = np.where(users == u)[0]

    all_indices = np.arange(B)

    pair_list: List[Tuple[int, int]] = []
    label_list: List[int] = []

    for u in uniq_users:
        idxs = user_to_indices[int(u)]
        if len(idxs) < 2:
            continue

        # All positive pairs for this user
        pos_pairs = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pos_pairs.append((idxs[i], idxs[j]))
        if not pos_pairs:
            continue

        pos_pairs = np.array(pos_pairs, dtype=int)

        # Subsample positives if too many
        if len(pos_pairs) > max_pos_pairs_per_user:
            chosen = rng.choice(len(pos_pairs), size=max_pos_pairs_per_user, replace=False)
            pos_pairs = pos_pairs[chosen]

        # For each positive pair, sample a matched negative
        other_indices = np.setdiff1d(all_indices, idxs, assume_unique=False)
        if len(other_indices) == 0:
            continue

        neg_pairs = []
        for i, _ in pos_pairs:
            j_neg = int(rng.choice(other_indices))
            neg_pairs.append((i, j_neg))

        # Append to global list
        for p in pos_pairs:
            pair_list.append((int(p[0]), int(p[1])))
            label_list.append(1)
        for p in neg_pairs:
            pair_list.append((int(p[0]), int(p[1])))
            label_list.append(0)

    if not pair_list:
        raise RuntimeError("Not enough bouts to build any positive pairs. Check your data.")

    pairs = np.array(pair_list, dtype=int)
    labels = np.array(label_list, dtype=int)

    scores = _similarity_scores_for_pairs(F, pairs, metric=metric)
    return scores, labels


def run_histogram_attack(
    meta_df: pd.DataFrame,
    num_labels: Optional[int] = None,
    max_pos_pairs_per_user: int = 50,
    seed: int = 0,
    metric: str = "dot",
):
    """Run the simple label-histogram-based linkage attack and compute AUC.

    This is the main public entry point for Step 6.

    Args:
        meta_df: window-level metadata DataFrame from build_temporal_view.
        num_labels: number of distinct activity labels; inferred if None.
        max_pos_pairs_per_user: cap on positive pairs per user.
        seed: RNG seed.
        metric: similarity metric ('dot' or 'neg_l1').

    Returns:
        auc: ROC-AUC of the attack (float in [0, 1]).
        bout_df: per-bout summary DataFrame used to build the attack dataset.
        scores: (M,) similarity scores.
        labels: (M,) ground-truth labels for pairs.
    """
    bout_df, label_prob_cols = summarize_bouts(meta_df, num_labels=num_labels)
    scores, labels = build_pair_dataset(
        bout_df,
        label_prob_cols=label_prob_cols,
        max_pos_pairs_per_user=max_pos_pairs_per_user,
        seed=seed,
        metric=metric,
    )

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for ROC-AUC computation. "
            "Install it via 'pip install scikit-learn'."
        ) from e

    auc = float(roc_auc_score(labels, scores))
    return auc, bout_df, scores, labels
