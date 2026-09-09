from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class ChangeResult:
    difference: np.ndarray
    normalized_heatmap: np.ndarray
    threshold_mask: np.ndarray
    changed_fraction: float
    mean_absolute_change: float
    ssim: float


def _robust_normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def detect_change(before: np.ndarray, after: np.ndarray, threshold: float = 0.25) -> ChangeResult:
    """Compute a real pixel-level change map after both arrays are on the same grid."""
    if before.shape != after.shape:
        raise ValueError(f"Aligned rasters must have equal shapes, got {before.shape} and {after.shape}.")
    a = _robust_normalize(before)
    b = _robust_normalize(after)
    valid = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(b - a)
    diff[~valid] = np.nan
    heat = np.zeros_like(diff)
    if valid.any():
        q = np.nanpercentile(diff[valid], 98)
        heat = np.clip(diff / max(q, 1e-6), 0, 1)
    mask = valid & (heat >= threshold)
    changed_fraction = float(mask.sum() / max(valid.sum(), 1))
    mean_change = float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan")
    # SSIM requires finite values and a non-degenerate data range.
    af = np.nan_to_num(a, nan=0.0)
    bf = np.nan_to_num(b, nan=0.0)
    ssim = float(structural_similarity(af, bf, data_range=1.0))
    return ChangeResult(diff, heat, mask, changed_fraction, mean_change, ssim)
