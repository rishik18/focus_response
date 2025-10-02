"""Ring Difference Filter (RDF) implementations for focus detection."""

import numpy as np
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Pre-computed common kernels for fastest access
_PRECOMPUTED_KERNELS = {}


def _ring_disk_masks(inner_r: int, outer_r: int):
    """
    Create binary masks for disk and ring regions (cached with pre-computation).

    Args:
        inner_r: Inner radius (disk radius)
        outer_r: Outer radius (ring outer boundary)

    Returns:
        tuple: (disk_mask, ring_mask) as float32 arrays
    """
    # Check pre-computed cache first (faster than LRU cache)
    key = (inner_r, outer_r)
    if key in _PRECOMPUTED_KERNELS:
        return _PRECOMPUTED_KERNELS[key]

    # Compute kernel
    if inner_r < 0 or outer_r <= inner_r:
        raise ValueError("Require 0 <= inner_r < outer_r.")
    R = outer_r
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    r2 = xx*xx + yy*yy
    disk = (r2 <= inner_r*inner_r)
    ring = (r2 <= outer_r*outer_r) & (~disk)
    if ring.sum() == 0 or disk.sum() == 0:
        raise ValueError("Degenerate kernel; increase outer_r or adjust inner_r.")

    result = (disk.astype(np.float32), ring.astype(np.float32))

    # Cache for future use
    _PRECOMPUTED_KERNELS[key] = result

    return result


# Pre-compute most common kernel sizes at module load time
def _precompute_common_kernels():
    """Pre-compute commonly used kernel sizes for instant access."""
    common_radii = [
        (1, 3),   # Default
        (2, 5),   # Common multi-scale
        (3, 7),   # Common multi-scale
        (1, 2),   # Very small
        (2, 4),   # Small
        (3, 6),   # Medium
    ]
    for inner_r, outer_r in common_radii:
        _ring_disk_masks(inner_r, outer_r)

# Pre-compute at import time
_precompute_common_kernels()


@lru_cache(maxsize=32)
def _get_coverage_masks(img_shape: tuple, inner_r: int, outer_r: int):
    """
    Pre-compute coverage masks for edge correction (cached).

    Args:
        img_shape: Shape of the image (H, W)
        inner_r: Inner radius
        outer_r: Outer radius

    Returns:
        tuple: (cnt_ring, cnt_disk) coverage count arrays
    """
    disk, ring = _ring_disk_masks(inner_r, outer_r)
    ones = np.ones(img_shape, dtype=np.float32)

    # Use OpenCV filter2D for better performance on small kernels
    cnt_ring = cv2.filter2D(ones, -1, ring, borderType=cv2.BORDER_CONSTANT)
    cnt_disk = cv2.filter2D(ones, -1, disk, borderType=cv2.BORDER_CONSTANT)

    return cnt_ring, cnt_disk


def _auto_select_convolution_method(kernel_size: int, image_size: tuple) -> str:
    """
    Automatically select the best convolution method based on kernel and image size.

    Args:
        kernel_size: Size of the kernel (2*outer_r + 1)
        image_size: Shape of the image (H, W)

    Returns:
        'opencv', 'scipy', or 'fft'
    """
    kernel_area = kernel_size * kernel_size
    image_area = image_size[0] * image_size[1]

    # For very small kernels, OpenCV is fastest
    if kernel_size <= 7:
        return 'opencv'
    # For medium kernels, scipy is good
    elif kernel_size <= 15:
        return 'scipy'
    # For large kernels or large images, FFT is better
    elif kernel_area > 225 or image_area > 4000000:  # > 15x15 kernel or > 2000x2000 image
        return 'fft'
    else:
        return 'scipy'


def rdf_focus_numpy_edgesafe(
    image: np.ndarray,
    inner_r: int,
    outer_r: int,
    *,
    power: int = 2,
    use_fft: bool = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute Ring-Difference focus with border correction.

    The focus response is computed as:
        mean_ring(x) - mean_disk(x)
    where means are computed with zero-padding but divided by actual coverage counts.

    Args:
        image: Input image (grayscale or RGB)
        inner_r: Inner radius (disk)
        outer_r: Outer radius (ring)
        power: Power to raise the response to (1=abs, 2=squared)
        use_fft: Use FFT-based convolution. If None, auto-detect best method
        eps: Small constant to avoid division by zero

    Returns:
        Focus response map (same shape as input image)
    """
    if image.ndim == 3:
        # Convert RGB to grayscale using luminance formula (optimized)
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    # Ensure C-contiguous for better cache performance
    img = np.ascontiguousarray(image, dtype=np.float32)

    disk, ring = _ring_disk_masks(inner_r, outer_r)

    # Auto-detect best convolution method if not specified
    if use_fft is None:
        kernel_size = 2 * outer_r + 1
        method = _auto_select_convolution_method(kernel_size, img.shape)
    elif use_fft:
        method = 'fft'
    else:
        method = 'opencv'  # Default to OpenCV for small kernels

    if method == 'fft':
        # FFT-based convolution (fast for large kernels)
        from scipy.signal import oaconvolve
        sum_ring = oaconvolve(img, ring, mode="same")
        sum_disk = oaconvolve(img, disk, mode="same")
        # Use cached coverage masks for FFT mode
        cnt_ring = oaconvolve(np.ones_like(img, np.float32), ring, mode="same")
        cnt_disk = oaconvolve(np.ones_like(img, np.float32), disk, mode="same")
    elif method == 'opencv':
        # OpenCV filter2D (fastest for small kernels)
        sum_ring = cv2.filter2D(img, -1, ring, borderType=cv2.BORDER_CONSTANT)
        sum_disk = cv2.filter2D(img, -1, disk, borderType=cv2.BORDER_CONSTANT)
        # Use cached coverage masks
        cnt_ring, cnt_disk = _get_coverage_masks(img.shape, inner_r, outer_r)
    else:  # scipy
        # Direct spatial convolution (medium kernels)
        from scipy.ndimage import convolve
        sum_ring = convolve(img, ring, mode="constant", cval=0.0)
        sum_disk = convolve(img, disk, mode="constant", cval=0.0)
        # Use cached coverage masks
        cnt_ring, cnt_disk = _get_coverage_masks(img.shape, inner_r, outer_r)

    # Vectorized division and subtraction
    mean_ring = sum_ring / (cnt_ring + eps)
    mean_disk = sum_disk / (cnt_disk + eps)
    resp = mean_ring - mean_disk

    # Optimized power operations
    if power == 1:
        return np.abs(resp, out=resp)  # In-place for memory efficiency
    elif power == 2:
        return np.square(resp, out=resp)  # Faster than resp * resp
    else:
        return np.power(np.abs(resp), float(power))


def rdf_multiscale(image: np.ndarray, radii, power: int = 2, use_numba=False, parallel: bool = True):
    """
    Compute RDF maps at multiple scales with optional parallel processing.

    Args:
        image: Input image
        radii: Iterable of (inner_r, outer_r) tuples
        power: Power to raise the response to
        use_numba: Use numba-accelerated version (if available)
        parallel: Use parallel processing for multiple scales (default: True)

    Returns:
        List of focus maps, one per scale
    """
    if use_numba:
        # Note: rdf_focus_numba not included in this refactor
        raise NotImplementedError("Numba version not available")

    radii_list = list(radii)

    # Use parallel processing if multiple scales and parallel=True
    if parallel and len(radii_list) > 1:
        max_workers = min(len(radii_list), multiprocessing.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(rdf_focus_numpy_edgesafe, image, rin, rout, power=power)
                for rin, rout in radii_list
            ]
            maps = [f.result() for f in futures]
    else:
        # Sequential processing
        maps = [rdf_focus_numpy_edgesafe(image, rin, rout, power=power) for rin, rout in radii_list]

    return maps


def fuse_rdf_sum(img, radii, *, power=2, use_numba=False, normalize="p99", parallel=True, downsample=None):
    """
    Sum-fuse multi-scale RDF maps with optional per-scale normalization.

    Args:
        img: Input image
        radii: Iterable of (inner_r, outer_r) tuples
        power: Power for RDF computation
        use_numba: Use numba-accelerated version
        normalize: Normalization method ('none', 'p99', or 'mad')
            - 'none': raw sum
            - 'p99': divide each map by its 99th percentile (default)
            - 'mad': robust z-score (>=0 after clamp)
        parallel: Use parallel processing for multi-scale computation
        downsample: Downsample factor for large images (e.g., 2 or 4).
                   Results are upscaled back to original size.
                   Useful for very large images to speed up processing.

    Returns:
        tuple: (fused_map, individual_maps)
    """
    original_shape = img.shape

    # Optional downsampling for large images
    if downsample is not None and downsample > 1:
        h, w = img.shape
        new_h, new_w = h // downsample, w // downsample
        img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Scale radii accordingly
        radii_scaled = [(max(1, r1 // downsample), max(2, r2 // downsample))
                        for r1, r2 in radii]
    else:
        img_proc = img
        radii_scaled = radii

    maps = rdf_multiscale(img_proc, radii_scaled, power=power, use_numba=use_numba, parallel=parallel)
    eps = 1e-8

    if normalize == "none":
        maps_n = maps
    elif normalize == "p99":
        # Batch percentile computation for efficiency
        maps_n = [m / (np.percentile(m, 99) + eps) for m in maps]
    elif normalize == "mad":
        # Optimized MAD computation using vectorized operations
        maps_n = []
        for m in maps:
            # Compute median and MAD in one pass using partition for speed
            flat = m.ravel()
            med = np.median(flat)
            mad = np.median(np.abs(flat - med))
            z = (m - med) / (1.4826 * mad + eps)
            maps_n.append(np.clip(z, 0, None))  # keep positive focus energy
    else:
        raise ValueError("normalize must be 'none', 'p99', or 'mad'.")

    fused = np.sum(maps_n, axis=0).astype(np.float32)

    # Upsample back to original size if downsampling was used
    if downsample is not None and downsample > 1:
        fused = cv2.resize(fused, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        maps = [cv2.resize(m, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_LINEAR) for m in maps]

    return fused, maps
