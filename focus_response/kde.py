"""Kernel Density Estimation for focus region detection."""

import numpy as np
import cv2
from typing import Tuple


def kde_on_fused(
    fused: np.ndarray,
    top_percent: float = 20.0,
    bandwidth_px: float = 10.0,
    include_strength: bool = False,
    clip_percentile: float = 99.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Approximate 2-D KDE by placing impulses at selected pixels and
    convolving with a Gaussian (via OpenCV GaussianBlur).

    Args:
        fused: Fused focus map (H×W)
        top_percent: Keep the top X% strongest focus pixels
        bandwidth_px: Gaussian sigma (in pixels) for KDE smoothing
        include_strength: Weight kernels by fused intensity if True
        clip_percentile: Optional display clipping percentile
        normalize: Normalize density to [0,1] if True

    Returns:
        tuple: (density, threshold)
            - density: KDE heatmap (H×W) in [0,1] if normalize=True
            - threshold: Threshold used to select top_percent pixels
    """
    fused = fused.astype(np.float32, copy=False)
    H, W = fused.shape

    # Optimized threshold computation: avoid creating filtered array if possible
    # Compute percentiles in one call for efficiency
    nz_mask = fused > 0
    percentiles = None

    if clip_percentile is not None:
        # Compute both percentiles at once
        if nz_mask.any():
            percentiles = np.percentile(
                fused[nz_mask], [100.0 - float(top_percent), clip_percentile]
            )
            thr = percentiles[0]
        else:
            percentiles = np.percentile(
                fused, [100.0 - float(top_percent), clip_percentile]
            )
            thr = percentiles[0]
    else:
        # Only compute threshold percentile
        if nz_mask.any():
            thr = np.percentile(fused[nz_mask], 100.0 - float(top_percent))
        else:
            thr = np.percentile(fused, 100.0 - float(top_percent))

    # Optimized impulse array creation
    sel = fused >= thr
    if include_strength:
        # Directly create impulses with values (avoid zero initialization)
        impulses = np.where(sel, fused, 0).astype(np.float32)
    else:
        # Create sparse array efficiently
        impulses = sel.astype(np.float32)

    # KDE via Gaussian smoothing of impulses
    # Note: ksize=(0,0) lets OpenCV determine kernel from sigma
    # Using BORDER_CONSTANT (zeros) is more appropriate for KDE than BORDER_REPLICATE
    density = cv2.GaussianBlur(
        impulses,
        ksize=(0, 0),
        sigmaX=float(bandwidth_px),
        sigmaY=float(bandwidth_px),
        borderType=cv2.BORDER_CONSTANT,
    ).astype(np.float32)

    # Optional clipping (for nicer visualization) - use pre-computed percentile if available
    if clip_percentile is not None:
        if percentiles is not None and len(percentiles) > 1:
            hi = percentiles[1]
        else:
            hi = np.percentile(density, clip_percentile)
        if hi > 0:
            density = np.minimum(density, hi)

    if normalize and density.max() > 0:
        density = density / (density.max() + 1e-8)
        density = density.astype(np.float32)

    return density, float(thr)
