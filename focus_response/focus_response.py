"""
Focus Response Detection using Ring Difference Filter and KDE.

This module provides the main interface for detecting focus regions in images
using a multi-scale ring difference filter followed by kernel density estimation.
"""

import cv2
from time import time
import argparse
from pathlib import Path

try:
    from .filters import fuse_rdf_sum
    from .kde import kde_on_fused
    from .visualization import visualize_kde_density
    from .batch import batch_process_images, get_image_files, save_results
except ImportError:
    from filters import fuse_rdf_sum
    from kde import kde_on_fused
    from visualization import visualize_kde_density
    from batch import batch_process_images, get_image_files, save_results


def detect_focus_regions(
    image_path: str,
    radii=None,
    top_percent: float = 25.0,
    bandwidth_px: float = 10.0,
    power: int = 2,
    normalize: str = "p99",
    include_strength: bool = False,
    show_visualizations: bool = True,
):
    """
    Detect focus regions in an image using RDF and KDE.

    Args:
        image_path: Path to input image
        radii: List of (inner_r, outer_r) tuples for multi-scale RDF
        top_percent: Percentage of top focus pixels to use for KDE
        bandwidth_px: Gaussian bandwidth for KDE smoothing
        power: Power for RDF computation (1=abs, 2=squared)
        normalize: Normalization method ('none', 'p99', 'mad')
        include_strength: Weight KDE by focus intensity if True
        show_visualizations: Display visualization plots if True

    Returns:
        dict: Results containing 'fused', 'density', 'threshold', and timing info
    """
    if radii is None:
        radii = [(1, 3)]

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Build fused RDF focus map
    start = time()
    fused, maps = fuse_rdf_sum(
        img, radii, power=power, use_numba=False, normalize=normalize
    )
    fuse_time = time() - start

    # KDE over fused focus map
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

    # Visualize if requested
    if show_visualizations:
        visualize_kde_density(img, fused, density, show_on="image")
        visualize_kde_density(img, fused, density, show_on="focus")

    return {
        "fused": fused,
        "density": density,
        "threshold": thr,
        "individual_maps": maps,
        "fuse_time": fuse_time,
        "kde_time": kde_time,
        "total_time": fuse_time + kde_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Focus Response Detection - Process images using RDF and KDE"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input directory containing images or path to single image file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="1,3",
        help="RDF radii as comma-separated pairs (e.g., '1,3' or '1,3;2,5;3,7' for multi-scale)",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=25.0,
        help="Percentage of top focus pixels for KDE (default: 25.0)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=10.0,
        help="Gaussian bandwidth for KDE in pixels (default: 10.0)",
    )
    parser.add_argument(
        "--power",
        type=int,
        default=2,
        choices=[1, 2],
        help="Power for RDF computation (1=abs, 2=squared, default: 2)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="p99",
        choices=["none", "p99", "mad"],
        help="Normalization method (default: p99)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect CPU count)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images recursively in subdirectories",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualizations (default: only save arrays)",
    )
    parser.add_argument(
        "--no-arrays",
        action="store_true",
        help="Don't save arrays (only visualizations)",
    )

    args = parser.parse_args()

    # Parse radii
    try:
        if ";" in args.radii:
            # Multi-scale: "1,3;2,5;3,7"
            radii = [tuple(map(int, pair.split(","))) for pair in args.radii.split(";")]
        else:
            # Single scale: "1,3"
            radii = [tuple(map(int, args.radii.split(",")))]
    except Exception as e:
        print(f"Error parsing radii: {e}")
        print("Use format like '1,3' or '1,3;2,5;3,7' for multi-scale")
        exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Focus Response Detection")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Radii: {radii}")
    print(f"Top percent: {args.top_percent}%")
    print(f"Bandwidth: {args.bandwidth}px")
    print(f"Power: {args.power}")
    print(f"Normalize: {args.normalize}")
    print(f"Parallel: {'No' if args.no_parallel else 'Yes'}")
    print(f"Workers: {args.workers if args.workers else 'Auto'}")
    print("=" * 80)

    # Check if input is a file or directory
    if input_path.is_file():
        # Single file processing
        print(f"\nProcessing single image: {input_path.name}")
        results = detect_focus_regions(
            str(input_path),
            radii=radii,
            top_percent=args.top_percent,
            bandwidth_px=args.bandwidth,
            power=args.power,
            normalize=args.normalize,
            include_strength=False,
            show_visualizations=False,
        )

        # Save results
        import numpy as np

        basename = input_path.stem

        # Create subfolders
        if not args.no_arrays:
            filter_arrays_path = output_path / "filter_arrays"
            kde_arrays_path = output_path / "kde_arrays"
            filter_arrays_path.mkdir(exist_ok=True)
            kde_arrays_path.mkdir(exist_ok=True)

            np.save(
                str(filter_arrays_path / f"{basename}_filter.npy"), results["fused"]
            )
            np.save(str(kde_arrays_path / f"{basename}_kde.npy"), results["density"])

        if args.save_vis:
            filter_vis_path = output_path / "filter_vis"
            kde_vis_path = output_path / "kde_vis"
            filter_vis_path.mkdir(exist_ok=True)
            kde_vis_path.mkdir(exist_ok=True)

            fused_normalized = (results["fused"] / results["fused"].max() * 255).astype(
                np.uint8
            )
            cv2.imwrite(
                str(filter_vis_path / f"{basename}_filter.png"), fused_normalized
            )

            density_normalized = (results["density"] * 255).astype(np.uint8)
            density_color = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
            cv2.imwrite(str(kde_vis_path / f"{basename}_kde.png"), density_color)

        print("\nResults:")
        print(f"  Fuse time: {results['fuse_time']:.2f}s")
        print(f"  KDE time: {results['kde_time']:.2f}s")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Threshold: {results['threshold']:.4f}")
        print(f"\nResults saved to: {output_path}")

    elif input_path.is_dir():
        # Batch processing
        print(f"\nSearching for images in: {input_path}")
        image_files = get_image_files(input_path, recursive=args.recursive)

        if not image_files:
            print("No images found!")
            exit(1)

        print(f"Found {len(image_files)} images")
        print("\nProcessing images...")

        results = batch_process_images(
            image_files,
            radii=radii,
            top_percent=args.top_percent,
            bandwidth_px=args.bandwidth,
            power=args.power,
            normalize=args.normalize,
            include_strength=False,
            max_workers=args.workers,
            use_processes=False,
        )

        if results:
            save_results(
                results,
                output_path,
                save_arrays=not args.no_arrays,
                save_visualizations=args.save_vis,
            )

            # Print summary
            print("\n" + "=" * 80)
            print("Processing Summary")
            print("=" * 80)
            total_fuse = sum(r["fuse_time"] for r in results.values())
            total_kde = sum(r["kde_time"] for r in results.values())
            total_time = sum(r["total_time"] for r in results.values())

            print(f"Images processed: {len(results)}")
            print(f"Total fuse time: {total_fuse:.2f}s")
            print(f"Total KDE time: {total_kde:.2f}s")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average per image: {total_time / len(results):.2f}s")
            print("=" * 80)
        else:
            print("\nNo results to save!")
    else:
        print(f"Error: Input path not found: {input_path}")
        exit(1)
