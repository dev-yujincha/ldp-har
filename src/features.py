# features.py
"""
Feature preprocessing utilities for HAR:
- Compute per-feature statistics on the training set
- Standardize features
- Geometry-aware clipping (quantile box, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


# ---------------------------------------------------------
# 1. Statistic container
# ---------------------------------------------------------

@dataclass
class FeatureStats:
    """
    Per-feature statistics computed on the training set.

    All arrays have shape (D,), where D is the feature dimension (e.g., 561).
    """
    mean: np.ndarray
    std: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray

    @classmethod
    def from_array(
        cls,
        x: np.ndarray,
        low_q: float = 0.01,
        high_q: float = 0.99,
        eps: float = 1e-8,
    ) -> "FeatureStats":
        """
        Compute statistics from a 2D numpy array x with shape (N, D).

        low_q/high_q:
            Quantiles used for the 'quantile box' clipping (geometry-aware).
        eps:
            Lower bound for std to avoid division by zero.
        """
        assert x.ndim == 2, f"Expected 2D array (N, D), got shape {x.shape}"

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.maximum(std, eps)

        q_low = np.quantile(x, low_q, axis=0)
        q_high = np.quantile(x, high_q, axis=0)

        return cls(mean=mean, std=std, q_low=q_low, q_high=q_high)

    def save(self, path: PathLike) -> None:
        """
        Save statistics to a compressed npz file.
        """
        path = Path(path)
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            q_low=self.q_low,
            q_high=self.q_high,
        )

    @classmethod
    def load(cls, path: PathLike) -> "FeatureStats":
        """
        Load statistics previously saved with `save`.
        """
        path = Path(path)
        data = np.load(path)
        return cls(
            mean=data["mean"],
            std=data["std"],
            q_low=data["q_low"],
            q_high=data["q_high"],
        )


# ---------------------------------------------------------
# 2. Helper: Normalize shapes, broadcast to tensors
# ---------------------------------------------------------

def _as_numpy_2d(x: ArrayLike) -> np.ndarray:
    """
    Convert x to a 2D numpy array of shape (N, D).

    Handles:
    - numpy arrays: (N, D) or (N, 1, D) -> (N, D)
    - torch tensors: same, but converted via .cpu().numpy()
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3 and arr.shape[1] == 1:
        # e.g., (N, 1, D) -> (N, D)
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Expected (N, D) or (N, 1, D), got {arr.shape}")

    return arr


def _broadcast_stats_to_tensor(
    t: torch.Tensor,
    stats_vec: np.ndarray,
) -> torch.Tensor:
    """
    Convert 1D numpy stats_vec of shape (D,) to a tensor that
    broadcasts over t's batch/time dimensions.

    Example:
        t: (N, 1, D) or (N, D)
        stats_vec: (D,)
        -> tensor_stats: shape compatible with t for elementwise ops.
    """
    device = t.device
    stats = torch.from_numpy(stats_vec.astype(np.float32)).to(device)

    # Start as (D,)
    while stats.dim() < t.dim():
        # Insert singleton dimension at position 0 until shapes align
        stats = stats.unsqueeze(0)
    return stats


# ---------------------------------------------------------
# 3. Public: compute stats on training set
# ---------------------------------------------------------

def compute_feature_stats_from_loader(
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    max_batches: int | None = None,
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> FeatureStats:
    """
    Compute FeatureStats from a PyTorch DataLoader over the training set.

    loader:
        Yields (data, label) where data has shape (B, 1, D) or (B, D).
    max_batches:
        Optional cap on the number of batches to scan (for speed).

    Returns:
        FeatureStats instance computed on all collected samples.
    """
    xs = []

    for i, (data, _) in enumerate(loader):
        # data: (B, 1, D) or (B, D)
        arr = _as_numpy_2d(data)  # (B, D)
        xs.append(arr)

        if max_batches is not None and (i + 1) >= max_batches:
            break

    x_all = np.concatenate(xs, axis=0)  # (N, D)

    return FeatureStats.from_array(
        x_all,
        low_q=low_q,
        high_q=high_q,
    )


# ---------------------------------------------------------
# 4. Standardization
# ---------------------------------------------------------

def standardize(
    x: ArrayLike,
    stats: FeatureStats,
) -> ArrayLike:
    """
    Standardize features: (x - mean) / std per feature.

    Works with:
        - numpy arrays, shape (N, D) or (N, 1, D)
        - torch tensors, shape (N, D) or (N, 1, D)
    """
    if isinstance(x, torch.Tensor):
        mean_t = _broadcast_stats_to_tensor(x, stats.mean)
        std_t = _broadcast_stats_to_tensor(x, stats.std)
        return (x - mean_t) / std_t
    else:
        arr = np.asarray(x)
        # (N, 1, D) -> (N, D)
        squeeze_back = False
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
            squeeze_back = True
        elif arr.ndim != 2:
            raise ValueError(f"Expected (N, D) or (N, 1, D), got {arr.shape}")

        out = (arr - stats.mean) / stats.std
        if squeeze_back:
            out = out[:, np.newaxis, :]
        return out


# ---------------------------------------------------------
# 5. Geometry-aware clipping: quantile box
# ---------------------------------------------------------

def clip_quantile_box(
    x: ArrayLike,
    stats: FeatureStats,
    expand_margin: float = 0.0,
) -> ArrayLike:
    """
    Clip each feature into a 'quantile box' [q_low, q_high].

    expand_margin:
        If > 0, slightly expand the box around the midpoint:
        Let q_mid = (q_low + q_high) / 2, half_width = (q_high - q_low) / 2
        Then we use [q_mid - (1+expand_margin)*half_width,
                     q_mid + (1+expand_margin)*half_width].

        For expand_margin=0, this is exactly [q_low, q_high].

    This is the first geometry-aware clipping primitive.
    Other shapes (ℓ2-group, MEB) can be added on top of standardized data.
    """
    q_low = stats.q_low
    q_high = stats.q_high

    if expand_margin > 0:
        mid = (q_low + q_high) / 2.0
        half_width = (q_high - q_low) / 2.0
        half_width = (1.0 + expand_margin) * half_width
        q_low = mid - half_width
        q_high = mid + half_width

    if isinstance(x, torch.Tensor):
        low_t = _broadcast_stats_to_tensor(x, q_low)
        high_t = _broadcast_stats_to_tensor(x, q_high)
        return torch.clamp(x, min=low_t, max=high_t)
    else:
        arr = np.asarray(x)
        squeeze_back = False
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
            squeeze_back = True
        elif arr.ndim != 2:
            raise ValueError(f"Expected (N, D) or (N, 1, D), got {arr.shape}")

        out = np.clip(arr, q_low, q_high)
        if squeeze_back:
            out = out[:, np.newaxis, :]
        return out


# ---------------------------------------------------------
# 6. Stubs for other geometry-aware clipping (to fill later)
# ---------------------------------------------------------

def l2_group_clip(
    x: ArrayLike,
    max_norm: float,
    group_axes: Sequence[int] = (-1,),
) -> ArrayLike:
    """
    ℓ2 clipping over groups of features.

    x:
        Tensor or ndarray. For your current HAR setup, this will usually
        be (N, D) or (N, 1, D).
    max_norm:
        Maximum allowed ℓ2 norm per group; if norm > max_norm, rescale.

    group_axes:
        Axes over which to compute the ℓ2 norm. For simple per-window
        ℓ2 clipping over all features, use group_axes=(-1,).
    """
    if isinstance(x, torch.Tensor):
        # Compute squared norm along group_axes
        # (This version assumes features are along the last axis.)
        norm = torch.linalg.vector_norm(x, dim=group_axes, keepdim=True)
        factor = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        return x * factor
    else:
        arr = np.asarray(x)
        norm = np.linalg.norm(arr, axis=group_axes, keepdims=True)
        factor = np.minimum(max_norm / (norm + 1e-8), 1.0)
        return arr * factor

def meb_clip(
    x: ArrayLike,
    center: np.ndarray,
    radius: float,
    group_axes: Sequence[int] = (-1,),
) -> ArrayLike:
    """
    Approximate 'minimum enclosing ball' clipping.

    Works in standardized space:

        x_std -> translate by `center` -> scale back if norm > radius.

    center: (D,) in the same space as x (usually standardized features).
    radius: scalar, maximum allowed distance from center.

    group_axes: axes used for the ℓ2 norm, default: last axis.
    """
    if isinstance(x, torch.Tensor):
        device = x.device
        # Broadcast center to x
        c_t = _broadcast_stats_to_tensor(x, center.astype(np.float32))
        diff = x - c_t
        norm = torch.linalg.vector_norm(diff, dim=group_axes, keepdim=True)
        factor = torch.clamp(radius / (norm + 1e-8), max=1.0)
        return c_t + factor * diff
    else:
        arr = np.asarray(x)
        c = center.astype(np.float32)

        # Simple broadcasting for numpy (handles (N, 1, D) and (N, D))
        if arr.ndim == 3 and arr.shape[1] == 1:
            shape = (1, 1, c.shape[0])
        elif arr.ndim == 2:
            shape = (1, c.shape[0])
        else:
            raise ValueError(f"Expected (N, D) or (N, 1, D), got {arr.shape}")

        c_b = np.broadcast_to(c, shape if arr.ndim == 2 else (1, 1, c.shape[0]))
        diff = arr - c_b
        norm = np.linalg.norm(diff, axis=group_axes, keepdims=True)
        factor = np.minimum(radius / (norm + 1e-8), 1.0)
        return c_b + factor * diff