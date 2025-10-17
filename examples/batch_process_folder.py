"""
Example script for batch processing all images in a folder.

This demonstrates how to use the batch processing capabilities to
process multiple images in parallel for focus detection.
"""

import sys
from pathlib import Path

from focus_response.batch import batch_process_images, get_image_files, save_results

# Add parent directory to path to import focus_response
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Configuration
    input_folder = r"D:\Repos\focus_response\test_data"
    output_folder = r"D:\Repos\focus_response\output"

    # Processing parameters
    radii = [(1, 3), (2, 5), (3, 7)]  # Multi-scale RDF
    top_percent = 25.0
    bandwidth_px = 10.0
    power = 2
    normalize = "p99"

    # Parallelization settings
    max_workers = None  # Use all CPU cores
    use_processes = (
        False  # Use threads (better for most cases due to GIL release in NumPy/OpenCV)
    )

    print("=" * 80)
    print("Batch Focus Response Processing")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Radii: {radii}")
    print(f"Parallel workers: {'Auto' if max_workers is None else max_workers}")
    print(f"Parallelization mode: {'Processes' if use_processes else 'Threads'}")
    print("=" * 80)

    # Get all image files
    try:
        image_files = get_image_files(input_folder, recursive=False)
        print(f"\nFound {len(image_files)} images in {input_folder}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not image_files:
        print("No images found!")
        return

    # Process all images in parallel
    results = batch_process_images(
        image_files,
        radii=radii,
        top_percent=top_percent,
        bandwidth_px=bandwidth_px,
        power=power,
        normalize=normalize,
        include_strength=False,
        max_workers=max_workers,
        use_processes=use_processes,
    )

    # Save results
    if results:
        print(f"\nSaving results to {output_folder}...")
        save_results(
            results, output_folder, save_arrays=True, save_visualizations=False
        )

        # Print summary statistics
        print("\n" + "=" * 80)
        print("Processing Summary")
        print("=" * 80)
        total_fuse = sum(r["fuse_time"] for r in results.values())
        total_kde = sum(r["kde_time"] for r in results.values())
        total_time = sum(r["total_time"] for r in results.values())

        print(f"Total images processed: {len(results)}")
        print(f"Total fuse time: {total_fuse:.2f}s")
        print(f"Total KDE time: {total_kde:.2f}s")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average per image: {total_time / len(results):.2f}s")
        print("=" * 80)
    else:
        print("\nNo results to save!")


if __name__ == "__main__":
    main()
