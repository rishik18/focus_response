"""
Demonstrate intermediate outputs from filter and KDE stages.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from focus_response.filters import fuse_rdf_sum
from focus_response.kde import kde_on_fused


def visualize_stages(image_path: str):
    """Show outputs at each stage of the pipeline."""

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    print("=" * 80)
    print(f"Processing: {Path(image_path).name}")
    print("=" * 80)

    # Stage 1: Filter stage (RDF)
    print("\n[STAGE 1: Ring Difference Filter (RDF)]")
    radii = [(1, 3)]
    fused, individual_maps = fuse_rdf_sum(
        img, radii,
        power=2,
        normalize="p99"
    )

    print(f"Input image shape: {img.shape}")
    print(f"Input image dtype: {img.dtype}")
    print(f"Input image range: [{img.min()}, {img.max()}]")
    print(f"\nFused map shape: {fused.shape}")
    print(f"Fused map dtype: {fused.dtype}")
    print(f"Fused map range: [{fused.min():.4f}, {fused.max():.4f}]")
    print(f"Fused map mean: {fused.mean():.4f}")
    print(f"Fused map std: {fused.std():.4f}")
    print(f"\nNumber of individual scale maps: {len(individual_maps)}")

    # Stage 2: KDE stage
    print("\n[STAGE 2: Kernel Density Estimation (KDE)]")
    density, threshold = kde_on_fused(
        fused,
        top_percent=25.0,
        bandwidth_px=10.0,
        include_strength=False,
        clip_percentile=99.5,
        normalize=True
    )

    print(f"Threshold used: {threshold:.6f}")
    print(f"Pixels above threshold: {(fused >= threshold).sum()} ({(fused >= threshold).sum() / fused.size * 100:.1f}%)")
    print(f"\nDensity map shape: {density.shape}")
    print(f"Density map dtype: {density.dtype}")
    print(f"Density map range: [{density.min():.4f}, {density.max():.4f}]")
    print(f"Density map mean: {density.mean():.4f}")
    print(f"Density map std: {density.std():.4f}")

    # Visualization
    print("\n[VISUALIZATION]")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, RDF output, RDF thresholded
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fused, cmap='hot')
    axes[0, 1].set_title(f'RDF Output (Fused)\nRange: [{fused.min():.2f}, {fused.max():.2f}]')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)

    thresholded = (fused >= threshold).astype(np.float32)
    axes[0, 2].imshow(thresholded, cmap='gray')
    axes[0, 2].set_title(f'Thresholded RDF (top {25.0}%)\nThreshold: {threshold:.4f}')
    axes[0, 2].axis('off')

    # Row 2: Impulses, KDE density, KDE overlay
    impulses = np.where(fused >= threshold, 1.0, 0.0)
    axes[1, 0].imshow(impulses, cmap='gray')
    axes[1, 0].set_title(f'KDE Impulses\n{impulses.sum():.0f} points')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(density, cmap='jet')
    axes[1, 1].set_title(f'KDE Density Output\nRange: [{density.min():.2f}, {density.max():.2f}]')
    axes[1, 1].axis('off')
    plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046)

    # Overlay on original
    density_colored = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    density_rgb = cv2.cvtColor(density_colored, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(density_rgb, 0.5, img_rgb, 0.5, 0)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('KDE Overlay on Original')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('intermediate_outputs.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: intermediate_outputs.png")
    plt.show()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF OUTPUTS")
    print("=" * 80)
    print("\n1. FILTER STAGE (RDF) OUTPUT:")
    print("   - Type: Float32 2D array")
    print("   - Same size as input image")
    print("   - Values: Focus response strength (higher = more in-focus)")
    print("   - Normalized by 99th percentile")
    print(f"   - Actual range: [{fused.min():.4f}, {fused.max():.4f}]")

    print("\n2. KDE STAGE OUTPUT:")
    print("   - Type: Float32 2D array")
    print("   - Same size as input image")
    print("   - Values: Density of focus regions (0-1, normalized)")
    print("   - Smooth heatmap showing where focus is concentrated")
    print(f"   - Actual range: [{density.min():.4f}, {density.max():.4f}]")

    print("\n3. INTERPRETATION:")
    print("   - RDF output: Pixel-level focus measure (local edge contrast)")
    print("   - KDE output: Region-level focus probability (spatial density)")
    print("   - Higher values = More likely to be in-focus region")
    print("=" * 80)


if __name__ == "__main__":
    # Use default test image
    image_path = r"D:\Repos\focus_response\test_data\P7070077_DxO1200.jpg"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    visualize_stages(image_path)
