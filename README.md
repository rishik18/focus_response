# focus_response v0.1.4

A Python library for detecting focus regions in images using Ring Difference Filters (RDF) and Kernel Density Estimation (KDE).

<p align="center">
  <img src="https://github.com/user-attachments/assets/24a221e8-0771-4e69-b48d-2129dcb4d337" alt="frame_223" width="49%">
  <img width="49%" alt="output_2" src="https://github.com/user-attachments/assets/d140f9a2-661b-4aed-8903-56f54085f5e9" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/5937f609-c5fe-4aad-807e-d6e50727dfa2" width="49%">
  <img width="49%" alt="output_3" src="https://github.com/user-attachments/assets/858ccc8e-aff4-478c-94c0-d3b0cd315533" />
</p>

Above outputs were generated using: 
```python
detect_focus_regions(image_path, normalize="mad",top_percent=75, bandwidth_px =100)
```

## Installation

```bash
pip install focus_response
```

## Usage

### Single Image Processing

Process a single image to detect focus regions:

```python
from focus_response import detect_focus_regions

# Basic usage - detect focus regions in an image
results = detect_focus_regions(
    image_path="path/to/image.jpg",
    radii=[(1, 3)],              # Single scale: inner radius=1, outer radius=3
    top_percent=25.0,             # Use top 25% of focus pixels for KDE
    bandwidth_px=10.0,            # KDE smoothing bandwidth in pixels
    power=2,                      # Squared differences (2) or absolute (1)
    normalize="p99",              # Normalization: 'none', 'p99', or 'mad'
    include_strength=False,       # Weight KDE by focus intensity
    show_visualizations=True,     # Display visualization plots
    border_mode='reflect'         # Border handling: 'reflect', 'replicate', 'constant'
)

# Access results
fused_map = results['fused']              # RDF focus map (same size as input)
density_map = results['density']          # KDE density map (0-1 normalized)
threshold = results['threshold']          # Focus threshold value
individual_maps = results['individual_maps']  # List of RDF maps per scale
fuse_time = results['fuse_time']          # RDF computation time (seconds)
kde_time = results['kde_time']            # KDE computation time (seconds)
total_time = results['total_time']        # Total processing time (seconds)
```

**Output Format (Single Image):**
- `fused`: `np.ndarray` of shape `(H, W)` with dtype `float32` - Combined RDF focus map
- `density`: `np.ndarray` of shape `(H, W)` with dtype `float32` - KDE density map normalized to [0, 1]
- `threshold`: `float` - The focus value threshold used for selecting top pixels
- `individual_maps`: `list[np.ndarray]` - Individual RDF maps for each scale
- `fuse_time`: `float` - Time spent computing RDF (seconds)
- `kde_time`: `float` - Time spent computing KDE (seconds)
- `total_time`: `float` - Total processing time (seconds)

### Batch Processing

Process multiple images efficiently with parallel processing:

```python
from focus_response import batch_process_images, get_image_files, save_results

# Get all images from a folder
image_paths = get_image_files(
    folder_path="path/to/images",
    extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'),
    recursive=False  # Set True to search subdirectories
)

print(f"Found {len(image_paths)} images")

# Batch process all images
results = batch_process_images(
    image_paths=image_paths,
    radii=[(1, 3), (2, 5), (3, 7)],  # Multi-scale processing
    top_percent=25.0,
    bandwidth_px=10.0,
    power=2,
    normalize="p99",
    include_strength=False,
    max_workers=None,                 # Auto-detect CPU count
    use_processes=True,               # Use processes (default) or threads (False)
    progress_callback=None,           # Optional: callback(completed, total, path)
    batch_size=None,                  # Process all at once (None) or in batches
    output_folder=None,               # Set to save results incrementally (returns empty dict)
    save_arrays=True,                 # Save .npy arrays (when output_folder is set)
    save_visualizations=False,        # Save .png visualizations (when output_folder is set)
    border_mode='reflect'             # Border handling: 'reflect', 'replicate', 'constant'
)

# Save results manually (if output_folder not specified above)
save_results(
    results=results,
    output_folder="output",
    save_arrays=True,                 # Save raw .npy arrays
    save_visualizations=True,         # Save .png visualizations
    clear_results=False               # Clear results dict after saving to free memory
)

print(f"Processed {len(results)} images successfully!")
```

**Output Format (Batch Processing):**

The `batch_process_images` function returns a dictionary mapping image paths to their processing results.

**Note:** When `output_folder` is specified, results are saved to disk incrementally and the function returns an **empty dictionary** to conserve memory. Results are automatically saved after processing each batch.
```python
{
    'path/to/image1.jpg': {
        'fused': np.ndarray,              # Shape (H, W), dtype float32
        'density': np.ndarray,            # Shape (H, W), dtype float32, range [0, 1]
        'threshold': float,               # Focus threshold value
        'individual_maps': list[np.ndarray],  # Per-scale RDF maps
        'fuse_time': float,               # RDF time in seconds
        'kde_time': float,                # KDE time in seconds
        'total_time': float               # Total time in seconds
    },
    'path/to/image2.jpg': { ... },
    ...
}
```

**Saved Output Structure (when using `save_results` or `output_folder`):**
```
output/
├── filter_arrays/          # Raw RDF fused maps (.npy files)
│   ├── image1_filter.npy   # np.ndarray, shape (H, W), dtype float32
│   └── image2_filter.npy
├── kde_arrays/             # Raw KDE density maps (.npy files)
│   ├── image1_kde.npy      # np.ndarray, shape (H, W), dtype float32, range [0, 1]
│   └── image2_kde.npy
├── filter_vis/             # RDF visualizations (optional .png files)
│   ├── image1_filter.png   # Grayscale visualization (0-255)
│   └── image2_filter.png
└── kde_vis/                # KDE visualizations (optional .png files)
    ├── image1_kde.png      # Colored heatmap using COLORMAP_JET
    └── image2_kde.png
```

### Advanced: Using Individual Components

You can also use the individual processing components:

```python
import cv2
from focus_response import fuse_rdf_sum, kde_on_fused, visualize_kde_density

# Load image
img = cv2.imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)

# Step 1: Compute RDF focus map
fused, maps = fuse_rdf_sum(
    img=img,
    radii=[(1, 3), (2, 5)],
    power=2,
    use_numba=False,
    normalize="p99",
    parallel=True,
    downsample=None,                  # Optional: downsample factor (e.g., 2 or 4) for large images
    border_mode='reflect'             # Border handling: 'reflect', 'replicate', 'constant'
)

# Step 2: Apply KDE to get density map
density, threshold = kde_on_fused(
    fused=fused,
    top_percent=25.0,
    bandwidth_px=10.0,
    include_strength=False,
    clip_percentile=99.5,
    normalize=True
)

# Step 3: Visualize results
visualize_kde_density(img, fused, density, show_on="image")  # Overlay on original
visualize_kde_density(img, fused, density, show_on="focus")  # Overlay on focus map
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint
flake8
```

## References
1. Surh, J., Jeon, H. G., Park, Y., Im, S., Ha, H., & So Kweon, I. (2017). Noise robust depth from focus using a ring difference filter. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6328-6337).

## License

MIT License - see LICENSE file for details.
