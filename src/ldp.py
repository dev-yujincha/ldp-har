# src/ldp.py
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from torch import Tensor

from .features import FeatureStats, _broadcast_stats_to_tensor, standardize
from .clip import clip_quantile_box


def laplace_noise_like(x: Tensor, scale: Tensor) -> Tensor:
    """
    Draw Laplace(0, scale) noise with per-feature scale.

    x, scale: same broadcastable shape, e.g. (B, 1, D).
    """
    # U in (-0.5, 0.5)
    u = torch.rand_like(x) - 0.5
    # Inverse CDF of Laplace
    return -scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))


def apply_laplace_ldp_qbox(
    x: Tensor,
    stats: FeatureStats,
    eps: float,
    expand_margin: float = 0.0,
    standardize_after: bool = True,
) -> Tensor:
    """
    Quantile-box clipping in raw space + Laplace LDP.

    x: (B, 1, D) or (B, D) tensor of *raw* features (before standardization).
    stats: FeatureStats computed on training set (raw domain quantiles).
    eps: LDP epsilon (scalar).
    expand_margin: optional margin around [q_low, q_high].
    standardize_after: if True, standardize with (x - mean)/std at the end.

    Returns tensor with the *same shape* as x.
    """
    # 1) clip each feature to its quantile box
    x_clip = clip_quantile_box(x, stats, expand_margin=expand_margin)

    # 2) per-feature sensitivity Δ_j = q_high_j - q_low_j
    delta = stats.q_high - stats.q_low            # (D,)
    sens = _broadcast_stats_to_tensor(x_clip, delta)  # broadcast to x_clip

    # 3) Laplace scale b_j = Δ_j / eps
    scale = sens / eps

    # 4) add Laplace noise
    noise = laplace_noise_like(x_clip, scale.to(x_clip.device))
    x_noisy = x_clip + noise

    # 5) optionally standardize (for CNN / LR training)
    if standardize_after:
        x_noisy = standardize(x_noisy, stats)

    return x_noisy
