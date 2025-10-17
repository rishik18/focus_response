"""
Simple example of batch processing with minimal configuration.
"""

import sys
from pathlib import Path

from focus_response.batch import batch_process_images, get_image_files, save_results

sys.path.insert(0, str(Path(__file__).parent.parent))


# Get all images from a folder
image_folder = r"D:\Repos\focus_response\test_data"
images = get_image_files(image_folder)

print(f"Processing {len(images)} images...")

# Process all images in parallel (uses all CPU cores by default)
results = batch_process_images(
    images,
    radii=[(1, 3)],  # Single scale for speed
    max_workers=None,  # Auto-detect CPU count
    use_processes=False,  # Use threads (recommended)
)

# Save results
save_results(results, "output", save_arrays=True, save_visualizations=True)

print(f"\nProcessed {len(results)} images successfully!")
