from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FusionResult:
    fused: np.ndarray
    optical_mean: float
    sar_mean: float
    optical_std: float
    sar_std: float
    correlation: float


def _zscore(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    valid = np.isfinite(a)
    if not valid.any():
        raise ValueError("Fusion input contains no finite pixels.")
    mu, sigma = float(np.nanmean(a)), float(np.nanstd(a))
    return np.nan_to_num((a - mu) / max(sigma, 1e-8), nan=0.0)


def fuse_optical_sar(optical: np.ndarray, sar: np.ndarray, optical_weight: float = 0.5) -> FusionResult:
    if optical.shape != sar.shape:
        raise ValueError("Optical and SAR arrays must be aligned to the same grid before fusion.")
    if not 0 <= optical_weight <= 1:
        raise ValueError("optical_weight must be between 0 and 1.")
    o, s = _zscore(optical), _zscore(sar)
    fused = optical_weight * o + (1 - optical_weight) * s
    valid = np.isfinite(optical) & np.isfinite(sar)
    if valid.sum() < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(optical[valid].ravel(), sar[valid].ravel())[0, 1])
    return FusionResult(
        fused=fused,
        optical_mean=float(np.nanmean(optical)),
        sar_mean=float(np.nanmean(sar)),
        optical_std=float(np.nanstd(optical)),
        sar_std=float(np.nanstd(sar)),
        correlation=corr,
    )
