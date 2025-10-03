"""Ring Difference Filter (RDF) implementations for focus detection."""

import numpy as np
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import multiprocessing

# Pre-computed common kernels for fastest access
_PRECOMPUTED_KERNELS = {}


def _ring_disk_masks(inner_r: int, outer_r: int) -> Tuple[np.ndarray, np.ndarray]:
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
    # Explicitly use int64 to prevent overflow for large radii
    r2 = (xx.astype(np.int64)**2) + (yy.astype(np.int64)**2)
    inner_r2 = int(inner_r) * int(inner_r)
    outer_r2 = int(outer_r) * int(outer_r)
    disk = (r2 <= inner_r2)
    ring = (r2 <= outer_r2) & (~disk)
    if ring.sum() == 0 or disk.sum() == 0:
        raise ValueError("Degenerate kernel; increase outer_r or adjust inner_r.")

    result = (disk.astype(np.float32), ring.astype(np.float32))

    # Cache for future use
    _PRECOMPUTED_KERNELS[key] = result

    return result


# Pre-compute most common kernel sizes at module load time
def _precompute_common_kernels() -> None:
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
def _get_coverage_masks(img_shape: tuple, inner_r: int, outer_r: int) -> Tuple[np.ndarray, np.ndarray]:
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

    Thresholds based on empirical benchmarking:
    - OpenCV: Best for small kernels (<= 9x9) on any image size
    - scipy: Best for medium kernels (10x10 to 19x19) on small-medium images
    - FFT: Best for large kernels (>= 20x20) or when kernel/image ratio > 5%

    Args:
        kernel_size: Size of the kernel (2*outer_r + 1)
        image_size: Shape of the image (H, W)

    Returns:
        'opencv', 'scipy', or 'fft'
    """
    kernel_area = kernel_size * kernel_size
    image_area = image_size[0] * image_size[1]

    # Kernel represents significant portion of image - use FFT
    if kernel_area / image_area > 0.05:
        return 'fft'

    # Very small kernels - OpenCV is highly optimized
    if kernel_size <= 9:
        return 'opencv'

    # Very large kernels - FFT dominates
    if kernel_size >= 20:
        return 'fft'

    # Medium kernels on large images - prefer FFT
    if image_area > 4000000 and kernel_size > 15:
        return 'fft'

    # Default to scipy for medium cases
    return 'scipy'


def rdf_focus_numpy_edgesafe(
    image: np.ndarray,
    inner_r: int,
    outer_r: int,
    *,
    power: int = 2,
    use_fft: bool = None,
    eps: float = 1e-8,
    border_mode: str = 'reflect'
) -> np.ndarray:
    """
    Compute Ring-Difference focus with border correction.

    The focus response is computed as:
        mean_ring(x) - mean_disk(x)
    where means are computed with border extension and divided by actual coverage counts.

    Args:
        image: Input image (grayscale or RGB)
        inner_r: Inner radius (disk)
        outer_r: Outer radius (ring)
        power: Power to raise the response to (1=abs, 2=squared)
        use_fft: Use FFT-based convolution. If None, auto-detect best method
        eps: Small constant to avoid division by zero
        border_mode: Border extension method ('constant', 'reflect', 'replicate')
            - 'constant': Zero-padding (good for images with black backgrounds)
            - 'reflect': Mirror reflection (default, good for natural images)
            - 'replicate': Edge pixel repetition (good for uniform backgrounds)

    Returns:
        Focus response map (same shape as input image)

    Edge Handling:
        Uses border extension with coverage correction. Pixels within outer_r
        distance from image edges will have partial kernel coverage. Coverage
        correction normalizes by actual kernel support to reduce edge bias.
    """
    if image.ndim == 3:
        if image.shape[2] < 3:
            # Single or dual channel (shouldn't happen but handle gracefully)
            image = image[..., 0]
        else:
            # Convert RGB(A) to grayscale using luminance formula
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

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

    # Map border_mode to OpenCV constants
    cv2_border_map = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT_101,
        'replicate': cv2.BORDER_REPLICATE
    }

    # Map border_mode to scipy modes
    scipy_mode_map = {
        'constant': 'constant',
        'reflect': 'reflect',
        'replicate': 'nearest'
    }

    if border_mode not in cv2_border_map:
        raise ValueError(f"border_mode must be one of {list(cv2_border_map.keys())}, got '{border_mode}'")

    if method == 'fft':
        # FFT-based convolution (fast for large kernels)
        # Note: FFT always uses zero-padding (equivalent to 'constant' mode)
        from scipy.signal import oaconvolve
        if border_mode != 'constant':
            # Pre-pad image with border mode, then crop after convolution
            pad_size = outer_r
            if border_mode == 'reflect':
                img_padded = np.pad(img, pad_size, mode='reflect')
            elif border_mode == 'replicate':
                img_padded = np.pad(img, pad_size, mode='edge')
            sum_ring = oaconvolve(img_padded, ring, mode="same")
            sum_disk = oaconvolve(img_padded, disk, mode="same")
            # Crop back to original size
            sum_ring = sum_ring[pad_size:-pad_size, pad_size:-pad_size]
            sum_disk = sum_disk[pad_size:-pad_size, pad_size:-pad_size]
        else:
            sum_ring = oaconvolve(img, ring, mode="same")
            sum_disk = oaconvolve(img, disk, mode="same")
        # Use cached coverage masks (same for all convolution methods)
        cnt_ring, cnt_disk = _get_coverage_masks(img.shape, inner_r, outer_r)
    elif method == 'opencv':
        # OpenCV filter2D (fastest for small kernels)
        cv2_border = cv2_border_map[border_mode]
        sum_ring = cv2.filter2D(img, -1, ring, borderType=cv2_border)
        sum_disk = cv2.filter2D(img, -1, disk, borderType=cv2_border)
        # Use cached coverage masks
        cnt_ring, cnt_disk = _get_coverage_masks(img.shape, inner_r, outer_r)
    else:  # scipy
        # Direct spatial convolution (medium kernels)
        from scipy.ndimage import convolve
        scipy_mode = scipy_mode_map[border_mode]
        sum_ring = convolve(img, ring, mode=scipy_mode, cval=0.0)
        sum_disk = convolve(img, disk, mode=scipy_mode, cval=0.0)
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


def rdf_multiscale(
    image: np.ndarray,
    radii: List[Tuple[int, int]],
    power: int = 2,
    use_numba: bool = False,
    parallel: bool = True,
    border_mode: str = 'reflect'
) -> List[np.ndarray]:
    """
    Compute RDF maps at multiple scales with optional parallel processing.

    Args:
        image: Input image
        radii: Iterable of (inner_r, outer_r) tuples
        power: Power to raise the response to
        use_numba: Use numba-accelerated version (if available)
        parallel: Use parallel processing for multiple scales (default: True)
        border_mode: Border extension method ('constant', 'reflect', 'replicate')

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
                executor.submit(rdf_focus_numpy_edgesafe, image, rin, rout, power=power, border_mode=border_mode)
                for rin, rout in radii_list
            ]
            maps = [f.result() for f in futures]
    else:
        # Sequential processing
        maps = [rdf_focus_numpy_edgesafe(image, rin, rout, power=power, border_mode=border_mode) for rin, rout in radii_list]

    return maps


def fuse_rdf_sum(
    img: np.ndarray,
    radii: List[Tuple[int, int]],
    *,
    power: int = 2,
    use_numba: bool = False,
    normalize: str = "p99",
    parallel: bool = True,
    downsample: Optional[int] = None,
    border_mode: str = 'reflect'
) -> Tuple[np.ndarray, List[np.ndarray]]:
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
        border_mode: Border extension method ('constant', 'reflect', 'replicate')
            See rdf_focus_numpy_edgesafe for details.

    Returns:
        tuple: (fused_map, individual_maps)
    """
    original_shape = img.shape

    # Optional downsampling for large images
    if downsample is not None and downsample > 1:
        if not isinstance(downsample, int) or downsample <= 0:
            raise ValueError(f"downsample must be a positive integer, got {downsample}")

        # Handle both 2D and 3D images
        h, w = img.shape[:2]
        new_h, new_w = max(1, h // downsample), max(1, w // downsample)

        if new_h < 5 or new_w < 5:
            raise ValueError(f"Downsample factor {downsample} too large for image size {(h, w)}")

        img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Scale radii accordingly
        radii_scaled = [(max(1, r1 // downsample), max(2, r2 // downsample))
                        for r1, r2 in radii]
    else:
        img_proc = img
        radii_scaled = radii

    maps = rdf_multiscale(img_proc, radii_scaled, power=power, use_numba=use_numba, parallel=parallel, border_mode=border_mode)
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
            # Compute median and MAD efficiently
            med = np.median(m)  # No need to flatten
            mad = np.median(np.abs(m - med))
            # Avoid division if MAD is too small (degenerate case)
            if mad < eps:
                maps_n.append(np.zeros_like(m))
            else:
                z = (m - med) / (1.4826 * mad)
                maps_n.append(np.maximum(z, 0))  # Faster than clip for one-sided
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
