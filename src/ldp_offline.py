# src/ldp_offline.py

import numpy as np
from src.features import FeatureStats, standardize
from src.clip import clip_quantile_box
from numpy.random import default_rng


def apply_ldp_offline(X, stats: FeatureStats, eps: float, seed=0):
    """
    Offline LDP preprocessing for ALL samples.
    This reproduces the exact mechanism used in training.

    Args:
        X: numpy array (N, D)
        stats: FeatureStats (from training split)
        eps: epsilon for Laplace LDP
        seed: RNG seed

    Returns:
        X_ldp: numpy (N, D) privatized features
    """
    rng = default_rng(seed)
    N, D = X.shape
    X = X.astype(np.float32)

    # 1. standardize
    X_std = (X - stats.mean) / stats.std

    # 2. geometry-aware clip
    X_clipped = clip_quantile_box(X_std, stats)

    # 3. Laplace noise
    scale = 1.0 / eps
    noise = rng.laplace(loc=0.0, scale=scale, size=X_clipped.shape)
    X_noisy = X_clipped + noise

    # 4. unstandardize
    X_final = X_noisy * stats.std + stats.mean

    return X_final.astype(np.float32)