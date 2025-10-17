"""Batch processing utilities for focus detection on multiple images."""

import numpy as np
import cv2
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Callable
import multiprocessing
from time import time

try:
    from .filters import fuse_rdf_sum
    from .kde import kde_on_fused
except ImportError:
    from filters import fuse_rdf_sum
    from kde import kde_on_fused


def load_image(
    image_path: Union[str, Path], grayscale: bool = True
) -> Optional[np.ndarray]:
    """
    Load a single image.

    Args:
        image_path: Path to image file
        grayscale: Load as grayscale if True

    Returns:
        Image array or None if loading failed
    """
    try:
        if grayscale:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def load_images_parallel(
    image_paths: List[Union[str, Path]],
    grayscale: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load multiple images in parallel.

    Args:
        image_paths: List of paths to image files
        grayscale: Load as grayscale if True
        max_workers: Maximum number of parallel workers (default: CPU count)

    Returns:
        Dictionary mapping image path to loaded image array
    """
    if not image_paths:
        return {}

    if max_workers is None:
        max_workers = min(len(image_paths), multiprocessing.cpu_count())

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(load_image, path, grayscale): str(path)
            for path in image_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                img = future.result()
                if img is not None:
                    results[path] = img
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return results


def process_single_image(
    img: np.ndarray,
    radii: List[Tuple[int, int]],
    top_percent: float = 25.0,
    bandwidth_px: float = 10.0,
    power: int = 2,
    normalize: str = "p99",
    include_strength: bool = False,
    border_mode: str = "reflect",
) -> Dict:
    """
    Process a single image through the focus detection pipeline.

    Args:
        img: Input image array
        radii: List of (inner_r, outer_r) tuples for multi-scale RDF
        top_percent: Percentage of top focus pixels for KDE
        bandwidth_px: Gaussian bandwidth for KDE smoothing
        power: Power for RDF computation
        normalize: Normalization method ('none', 'p99', 'mad')
        include_strength: Weight KDE by focus intensity if True
        border_mode: Border extension method ('constant', 'reflect', 'replicate')

    Returns:
        Dictionary containing results
    """
    start = time()

    # RDF fusion
    fused, maps = fuse_rdf_sum(
        img,
        radii,
        power=power,
        use_numba=False,
        normalize=normalize,
        parallel=True,
        border_mode=border_mode,
    )
    fuse_time = time() - start

    # KDE
    start = time()
    density, thr = kde_on_fused(
        fused,
        top_percent=top_percent,
        bandwidth_px=bandwidth_px,
        include_strength=include_strength,
        clip_percentile=99.5,
        normalize=True,
    )
    kde_time = time() - start

    return {
        "fused": fused,
        "density": density,
        "threshold": thr,
        "individual_maps": maps,
        "fuse_time": fuse_time,
        "kde_time": kde_time,
        "total_time": fuse_time + kde_time,
    }


def batch_process_images(
    image_paths: List[Union[str, Path]],
    radii: List[Tuple[int, int]] = None,
    top_percent: float = 25.0,
    bandwidth_px: float = 10.0,
    power: int = 2,
    normalize: str = "p99",
    include_strength: bool = False,
    max_workers: Optional[int] = None,
    use_processes: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    batch_size: Optional[int] = None,
    output_folder: Optional[Union[str, Path]] = None,
    save_arrays: bool = True,
    save_visualizations: bool = False,
    border_mode: str = "reflect",
) -> Dict[str, Dict]:
    """
    Batch process multiple images in parallel.

    Args:
        image_paths: List of paths to image files
        radii: List of (inner_r, outer_r) tuples (default: [(1, 3)])
        top_percent: Percentage of top focus pixels for KDE (0-100)
        bandwidth_px: Gaussian bandwidth for KDE smoothing (>= 0)
        power: Power for RDF computation
        normalize: Normalization method ('none', 'p99', 'mad')
        include_strength: Weight KDE by focus intensity if True
        max_workers: Maximum number of parallel workers
        use_processes: Use ProcessPoolExecutor if True, ThreadPoolExecutor if False
        progress_callback: Optional callback function(completed, total, current_file)
        batch_size: Maximum number of images to process at once (default: None = all at once)
        output_folder: If specified, save results after each batch and clear from memory
        save_arrays: Save raw numpy arrays when output_folder is specified (default: True)
        save_visualizations: Save visualization images when output_folder is specified (default: False)
        border_mode: Border extension method ('constant', 'reflect', 'replicate')

    Returns:
        Dictionary mapping image path to processing results.
        Note: When output_folder is specified, results are saved to disk after each batch
        and NOT kept in memory. The returned dictionary will be empty to conserve memory.
        If you need results in memory, do not specify output_folder and use save_results()
        manually after processing completes.

    Raises:
        ValueError: If parameters are out of valid range
    """
    # Input validation
    if radii is None:
        radii = [(1, 3)]

    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not (0 <= top_percent <= 100):
        raise ValueError(f"top_percent must be in [0, 100], got {top_percent}")

    if bandwidth_px < 0:
        raise ValueError(f"bandwidth_px must be non-negative, got {bandwidth_px}")

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    total_images = len(image_paths)
    all_results = {}

    # Handle empty input
    if total_images == 0:
        print("No images to process!")
        return all_results

    # Setup output folder if specified
    if output_folder is not None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subfolders
        if save_arrays:
            filter_arrays_path = output_path / "filter_arrays"
            kde_arrays_path = output_path / "kde_arrays"
            filter_arrays_path.mkdir(exist_ok=True)
            kde_arrays_path.mkdir(exist_ok=True)

        if save_visualizations:
            filter_vis_path = output_path / "filter_vis"
            kde_vis_path = output_path / "kde_vis"
            filter_vis_path.mkdir(exist_ok=True)
            kde_vis_path.mkdir(exist_ok=True)

    # Process in batches if batch_size is specified
    if batch_size is None:
        batch_size = total_images

    num_batches = (total_images + batch_size - 1) // batch_size
    total_completed = 0
    overall_start = time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]

        print(f"\n{'='*60}")
        print(
            f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_paths)} images)"
        )
        print(f"{'='*60}")

        # Load batch of images (parallel I/O)
        print(f"Loading {len(batch_paths)} images...")
        load_start = time()
        images = load_images_parallel(
            batch_paths, grayscale=True, max_workers=max_workers
        )
        load_time = time() - load_start
        print(f"Loaded {len(images)} images in {load_time:.2f}s")

        if not images:
            print("No images loaded successfully in this batch!")
            continue

        # Process images in parallel
        print(f"Processing {len(images)} images with {max_workers} workers...")
        process_start = time()
        batch_results = {}

        # Select executor based on use_processes flag
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        # Note for ProcessPoolExecutor: requires image data to be pickled, which may be slower for large images
        # For very large images, consider thread-based parallelism instead
        with ExecutorClass(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    process_single_image,
                    img,
                    radii,
                    top_percent,
                    bandwidth_px,
                    power,
                    normalize,
                    include_strength,
                    border_mode,
                ): path
                for path, img in images.items()
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    batch_results[path] = result
                    total_completed += 1
                    if progress_callback:
                        progress_callback(total_completed, total_images, path)
                    else:
                        print(
                            f"Completed {total_completed}/{total_images}: {Path(path).name} "
                            f"(fuse: {result['fuse_time']:.2f}s, kde: {result['kde_time']:.2f}s)"
                        )
                except Exception as e:
                    print(f"Error processing {path}: {e}")

        process_time = time() - process_start
        print(f"Batch {batch_idx + 1} completed in {process_time:.2f}s")

        # Save batch results if output folder specified
        if output_folder is not None:
            print(f"Saving batch {batch_idx + 1} results...")
            for img_path, result in batch_results.items():
                basename = Path(img_path).stem

                # Save raw arrays (default)
                if save_arrays:
                    # Save filter output (fused RDF map)
                    filter_array_path = filter_arrays_path / f"{basename}_filter.npy"
                    np.save(str(filter_array_path), result["fused"])

                    # Save KDE output (density map)
                    kde_array_path = kde_arrays_path / f"{basename}_kde.npy"
                    np.save(str(kde_array_path), result["density"])

                # Save visualizations (optional)
                if save_visualizations:
                    # Save filter visualization (grayscale)
                    filter_vis_img_path = filter_vis_path / f"{basename}_filter.png"
                    fused_max = result["fused"].max()
                    if fused_max > 0:
                        fused_normalized = (result["fused"] / fused_max * 255).astype(
                            np.uint8
                        )
                    else:
                        fused_normalized = np.zeros_like(
                            result["fused"], dtype=np.uint8
                        )
                    cv2.imwrite(str(filter_vis_img_path), fused_normalized)

                    # Save KDE visualization (colored heatmap)
                    kde_vis_img_path = kde_vis_path / f"{basename}_kde.png"
                    density_normalized = (result["density"] * 255).astype(np.uint8)
                    density_color = cv2.applyColorMap(
                        density_normalized, cv2.COLORMAP_JET
                    )
                    cv2.imwrite(str(kde_vis_img_path), density_color)

            print(f"Batch {batch_idx + 1} results saved to {output_path}")
        else:
            # Keep results in memory only if not saving to disk
            all_results.update(batch_results)

        # Clear memory after each batch
        del images
        del batch_results
        gc.collect()

    overall_time = time() - overall_start
    print(f"\n{'='*60}")
    print(f"All batches completed in {overall_time:.2f}s")
    if output_folder is not None:
        print(f"Results saved to: {output_path}")
        avg_msg = (
            f"Average time per image: {overall_time / total_completed:.2f}s"
            if total_completed > 0
            else "No images processed"
        )
        print(avg_msg)
    else:
        avg_msg = (
            f"Average time per image: {overall_time / len(all_results):.2f}s"
            if all_results
            else "No images processed"
        )
        print(avg_msg)
    print(f"{'='*60}")

    return all_results


def get_image_files(
    folder_path: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"),
    recursive: bool = False,
) -> List[Path]:
    """
    Get all image files from a folder.

    Args:
        folder_path: Path to folder containing images
        extensions: Tuple of valid file extensions
        recursive: Search subdirectories if True

    Returns:
        List of Path objects for image files
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if recursive:
        image_files = []
        for ext in extensions:
            image_files.extend(folder.rglob(f"*{ext}"))
            image_files.extend(folder.rglob(f"*{ext.upper()}"))
    else:
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def save_results(
    results: Dict[str, Dict],
    output_folder: Union[str, Path],
    save_arrays: bool = True,
    save_visualizations: bool = False,
    clear_results: bool = False,
):
    """
    Save processing results to disk with organized folder structure.

    Args:
        results: Dictionary of processing results from batch_process_images
        output_folder: Base folder to save results
        save_arrays: Save raw numpy arrays (.npy) if True (default: True)
        save_visualizations: Save visualization images if True (default: False)
        clear_results: Clear results dict after saving to free memory (default: False)

    Folder structure:
        output_folder/
        ├── filter_arrays/      # Raw RDF fused maps (.npy)
        ├── kde_arrays/         # Raw KDE density maps (.npy)
        ├── filter_vis/         # RDF visualization images (optional)
        └── kde_vis/            # KDE visualization images (optional)
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders
    if save_arrays:
        filter_arrays_path = output_path / "filter_arrays"
        kde_arrays_path = output_path / "kde_arrays"
        filter_arrays_path.mkdir(exist_ok=True)
        kde_arrays_path.mkdir(exist_ok=True)

    if save_visualizations:
        filter_vis_path = output_path / "filter_vis"
        kde_vis_path = output_path / "kde_vis"
        filter_vis_path.mkdir(exist_ok=True)
        kde_vis_path.mkdir(exist_ok=True)

    print(f"Saving results to {output_path}...")

    for img_path, result in results.items():
        basename = Path(img_path).stem

        # Save raw arrays (default)
        if save_arrays:
            # Save filter output (fused RDF map)
            filter_array_path = filter_arrays_path / f"{basename}_filter.npy"
            np.save(str(filter_array_path), result["fused"])

            # Save KDE output (density map)
            kde_array_path = kde_arrays_path / f"{basename}_kde.npy"
            np.save(str(kde_array_path), result["density"])

        # Save visualizations (optional)
        if save_visualizations:
            # Save filter visualization (grayscale)
            filter_vis_img_path = filter_vis_path / f"{basename}_filter.png"
            fused_max = result["fused"].max()
            if fused_max > 0:
                fused_normalized = (result["fused"] / fused_max * 255).astype(np.uint8)
            else:
                fused_normalized = np.zeros_like(result["fused"], dtype=np.uint8)
            cv2.imwrite(str(filter_vis_img_path), fused_normalized)

            # Save KDE visualization (colored heatmap)
            kde_vis_img_path = kde_vis_path / f"{basename}_kde.png"
            density_normalized = (result["density"] * 255).astype(np.uint8)
            density_color = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
            cv2.imwrite(str(kde_vis_img_path), density_color)

    # Print summary
    print("\nResults saved:")
    if save_arrays:
        print(f"  - Filter arrays: {filter_arrays_path}")
        print(f"  - KDE arrays: {kde_arrays_path}")
    if save_visualizations:
        print(f"  - Filter visualizations: {filter_vis_path}")
        print(f"  - KDE visualizations: {kde_vis_path}")
    print(f"  - Total images: {len(results)}")

    # Clear results to free memory if requested
    if clear_results:
        results.clear()
