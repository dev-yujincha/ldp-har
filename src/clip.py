import numpy as np
import torch
from .features import _broadcast_stats_to_tensor
from .features import standardize


def clip_quantile_box(x, stats, expand_margin=0.0):
    """
    Clip each feature to [q_low - m, q_high + m] using raw-domain quantiles.
    """
    if isinstance(x, torch.Tensor):
        q_low = _broadcast_stats_to_tensor(x, stats.q_low) - expand_margin
        q_high = _broadcast_stats_to_tensor(x, stats.q_high) + expand_margin
        return torch.clamp(x, q_low, q_high)
    else:
        arr = np.asarray(x)
        lo = stats.q_low - expand_margin
        hi = stats.q_high + expand_margin
        return np.clip(arr, lo, hi)
    
def l2_group_clip(x, max_norm, group_axes=(-1,)):
    """
    ℓ2 clipping on standardized features: ensure ||x||₂ ≤ max_norm.
    """
    if isinstance(x, torch.Tensor):
        norm = torch.linalg.vector_norm(x, dim=group_axes, keepdim=True)
        factor = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        return x * factor
    else:
        arr = np.asarray(x)
        norm = np.linalg.norm(arr, axis=group_axes, keepdims=True)
        factor = np.minimum(max_norm / (norm + 1e-8), 1.0)
        return arr * factor

def meb_clip(x, center, radius, group_axes=(-1,)):
    """
    Approximate Minimum Enclosing Ball clipping.
    Works on standardized features.
    """
    if isinstance(x, torch.Tensor):
        c_t = _broadcast_stats_to_tensor(x, center.astype(np.float32))
        diff = x - c_t
        norm = torch.linalg.vector_norm(diff, dim=group_axes, keepdim=True)
        factor = torch.clamp(radius / (norm + 1e-8), max=1.0)
        return c_t + factor * diff

    else:
        arr = np.asarray(x)
        c = center.astype(np.float32)

        if arr.ndim == 3 and arr.shape[1] == 1:
            c_b = c.reshape(1, 1, -1)
        else:
            c_b = c.reshape(1, -1)

        diff = arr - c_b
        norm = np.linalg.norm(diff, axis=group_axes, keepdims=True)
        factor = np.minimum(radius / (norm + 1e-8), 1.0)
        return c_b + factor * diff
    

def estimate_l2_radius(loader, stats, quantile=0.99, max_batches=None):
    """
    Estimate ℓ2 radius in standardized space by scanning the loader
    and taking a given quantile of per-sample norms.
    """
    norms = []

    for i, (x_raw, _) in enumerate(loader):
        # Use the standalone helper, not a method on stats
        x_std = standardize(x_raw, stats)          # (B, 1, D) or (B, D)

        arr = x_std.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[1] == 1:    # collapse (B, 1, D) -> (B, D)
            arr = arr[:, 0, :]

        batch_norms = np.linalg.norm(arr, axis=-1)  # (B,)
        norms.append(batch_norms)

        if max_batches is not None and (i + 1) >= max_batches:
            break

    all_norms = np.concatenate(norms, axis=0)
    return float(np.quantile(all_norms, quantile))


def estimate_meb_params(loader, stats, quantile=0.99, max_batches=None):
    # first pass: center
    total = None
    count = 0
    for i, (x_raw, _) in enumerate(loader):
        x_std = standardize(x_raw, stats)
        arr = x_std.detach().cpu().numpy()
        if arr.ndim == 3: arr = arr[:,0,:]
        total = arr.sum(axis=0) if total is None else total + arr.sum(axis=0)
        count += arr.shape[0]
        if max_batches and i+1 >= max_batches: break

    center = total / count

    # second pass: distances
    dists = []
    for i, (x_raw, _) in enumerate(loader):
        x_std = standardize(x_raw, stats)
        arr = x_std.detach().cpu().numpy()
        if arr.ndim == 3: arr = arr[:,0,:]
        dists.append(np.linalg.norm(arr - center, axis=-1))
        if max_batches and i+1 >= max_batches: break

    d = np.concatenate(dists)
    radius = float(np.quantile(d, quantile))
    return center.astype(np.float32), radius
