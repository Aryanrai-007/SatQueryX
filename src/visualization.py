from __future__ import annotations

import numpy as np


def percentile_clip(a: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        raise ValueError("No finite pixels available for visualization.")
    lo, hi = np.nanpercentile(a, [low, high])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)
