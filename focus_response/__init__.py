"""
focus_response - This library provides functionality to measure focus levels in images

Author: Hrishikesh Kanade
Email: kanade.hrishikesh1994@gmail.com
"""

__version__ = "0.1.5"
__author__ = "Hrishikesh Kanade"
__email__ = "rishikanade@outlook.com"

# Import main functions for easy access
from .filters import rdf_focus_numpy_edgesafe, rdf_multiscale, fuse_rdf_sum

from .kde import kde_on_fused

from .visualization import visualize_kde_density

from .focus_response import detect_focus_regions

from .batch import (
    batch_process_images,
    get_image_files,
    load_images_parallel,
    save_results,
    process_single_image,
)

__all__ = [
    # Main API
    "detect_focus_regions",
    # Core functions
    "rdf_focus_numpy_edgesafe",
    "rdf_multiscale",
    "fuse_rdf_sum",
    "kde_on_fused",
    "visualize_kde_density",
    # Batch processing
    "batch_process_images",
    "get_image_files",
    "load_images_parallel",
    "save_results",
    "process_single_image",
]
