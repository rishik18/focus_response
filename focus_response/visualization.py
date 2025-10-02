"""Visualization utilities for focus detection results."""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def _to_uint8_gray(x: np.ndarray):
    """
    Convert a float array to uint8 grayscale [0, 255].

    Args:
        x: Input array (any range)

    Returns:
        uint8 array normalized to [0, 255]
    """
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    return (255.0 * x).clip(0, 255).astype(np.uint8)


def visualize_kde_density(
    img: np.ndarray,
    fused: np.ndarray,
    density: np.ndarray,
    show_on: str = "image",
    alpha: float = 0.45,
    figsize=(10, 8)
):
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

    H, W = fused.shape
    # Base layer
    if show_on == "image":
        if img.ndim == 2:
            base_rgb = cv2.cvtColor(_to_uint8_gray(img), cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img[..., :3]
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
