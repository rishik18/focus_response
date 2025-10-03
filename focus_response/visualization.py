"""Visualization utilities for focus detection results."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def _to_uint8_gray(x: np.ndarray) -> np.ndarray:
    """
    Convert a float array to uint8 grayscale [0, 255].

    Args:
        x: Input array (any range)

    Returns:
        uint8 array normalized to [0, 255]
    """
    # Create a copy to avoid modifying input
    x = x.astype(np.float32, copy=True)

    # Handle NaN/Inf
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x_min = x.min()
    x_max = x.max()

    # Handle constant array
    if x_max - x_min < 1e-10:  # Essentially constant
        # Map to middle gray value
        return np.full(x.shape, 128, dtype=np.uint8)

    # Min-max normalization
    x_norm = (x - x_min) / (x_max - x_min)
    return (255.0 * x_norm).clip(0, 255).astype(np.uint8)


def visualize_kde_density(
    img: np.ndarray,
    fused: np.ndarray,
    density: np.ndarray,
    show_on: str = "image",
    alpha: float = 0.45,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Overlay the KDE heatmap on either the original image or the fused focus map.

    Args:
        img: Original image (H×W or H×W×3)
        fused: Fused focus map (H×W)
        density: KDE heatmap (H×W) in [0,1]
        show_on: "image" to overlay on original, "focus" to overlay on focus map
        alpha: Overlay transparency (0-1, higher = more heatmap visible)
        figsize: Figure size for matplotlib

    Returns:
        None (displays plot)
    """
    assert show_on in ("image", "focus")

    # Validate alpha
    alpha = np.clip(alpha, 0.0, 1.0)

    H, W = fused.shape
    # Base layer with proper type handling
    if show_on == "image":
        if img.ndim == 2:
            base_rgb = cv2.cvtColor(_to_uint8_gray(img), cv2.COLOR_GRAY2RGB)
        else:
            # Convert to uint8 first if needed
            if img.dtype != np.uint8:
                # Handle multi-channel images by converting to grayscale first
                if img.shape[2] >= 3:
                    img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    img_gray = img[..., 0]
                img_uint8 = _to_uint8_gray(img_gray)
                base_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            else:
                # Assume BGR from OpenCV imread
                if img.shape[2] >= 3:
                    base_rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
                else:
                    base_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = cv2.cvtColor(_to_uint8_gray(fused), cv2.COLOR_GRAY2RGB)

    # Heatmap (apply colormap to density)
    heat = (255.0 * np.clip(density, 0, 1)).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # Alpha blend
    overlay = cv2.addWeighted(heat_rgb, alpha, base_rgb, 1.0 - alpha, 0)

    # Show
    plt.figure(figsize=figsize)
    title = f"KDE overlay on {show_on}"
    plt.title(title)
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
