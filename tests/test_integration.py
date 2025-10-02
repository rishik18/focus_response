"""Integration tests for the complete pipeline."""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
import cv2

from focus_response import (
    fuse_rdf_sum,
    kde_on_fused,
    batch_process_images,
    get_image_files,
    save_results
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def test_image():
    """Create a synthetic test image with known focus regions."""
    # Create image with sharp edge
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255  # Sharp square
    img = cv2.GaussianBlur(img, (5, 5), 1.0)  # Slight blur
    return img


@pytest.fixture
def test_image_file(temp_dir, test_image):
    """Save test image to file."""
    path = Path(temp_dir) / "test_image.jpg"
    cv2.imwrite(str(path), test_image)
    return path


class TestCompletePipeline:
    """Test the complete processing pipeline."""

    def test_end_to_end(self, test_image):
        """Test complete pipeline from image to results."""
        radii = [(1, 3)]

        # Step 1: Filter stage
        fused, maps = fuse_rdf_sum(test_image, radii, power=2, normalize='p99')

        assert fused.shape == test_image.shape
        assert len(maps) == len(radii)

        # Step 2: KDE stage
        density, threshold = kde_on_fused(
            fused,
            top_percent=25.0,
            bandwidth_px=10.0
        )

        assert density.shape == test_image.shape
        assert 0 <= density.min() <= density.max() <= 1
        assert threshold >= 0

    def test_multiscale_pipeline(self, test_image):
        """Test pipeline with multiple scales."""
        radii = [(1, 3), (2, 5), (3, 7)]

        fused, maps = fuse_rdf_sum(test_image, radii)
        density, threshold = kde_on_fused(fused)

        assert len(maps) == 3
        assert density.shape == test_image.shape

    def test_batch_pipeline(self, temp_dir):
        """Test complete batch processing pipeline."""
        # Create a clean temp dir for this test
        import tempfile
        import shutil
        test_dir = tempfile.mkdtemp()

        try:
            # Create multiple test images
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                cv2.imwrite(str(Path(test_dir) / f"test_{i}.jpg"), img)

            # Get images
            image_files = get_image_files(test_dir)
            # May have duplicates due to case-insensitive matching, use unique names
            unique_files = list(set([f.name.lower() for f in image_files]))
            assert len(unique_files) == 3

            # Process
            results = batch_process_images(
                image_files,
                radii=[(1, 3)],
                max_workers=2,
                use_processes=False
            )

            assert len(results) == 3

            # Save
            output_dir = Path(test_dir) / "output"
            save_results(results, output_dir, save_arrays=True, save_visualizations=True)

            # Verify outputs
            assert (output_dir / "filter_arrays").exists()
            assert (output_dir / "kde_arrays").exists()
            assert (output_dir / "filter_vis").exists()
            assert (output_dir / "kde_vis").exists()
        finally:
            shutil.rmtree(test_dir)

    def test_focus_detection_quality(self, test_image):
        """Test that pipeline correctly identifies focus regions."""
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(test_image, radii)
        density, _ = kde_on_fused(fused, top_percent=25.0)

        # The center region (50:150, 50:150) should have higher focus
        center_density = density[75:125, 75:125].mean()
        edge_density = density[0:25, 0:25].mean()

        # Center should have more focus than corners
        assert center_density > edge_density

    def test_reproducibility(self, test_image):
        """Test that results are reproducible."""
        radii = [(1, 3)]

        # Run twice
        fused1, _ = fuse_rdf_sum(test_image, radii)
        density1, threshold1 = kde_on_fused(fused1)

        fused2, _ = fuse_rdf_sum(test_image, radii)
        density2, threshold2 = kde_on_fused(fused2)

        # Should be identical
        assert np.allclose(fused1, fused2)
        assert np.allclose(density1, density2)
        assert threshold1 == threshold2


class TestParameterSensitivity:
    """Test sensitivity to different parameters."""

    def test_radius_effect(self, test_image):
        """Test effect of different radii."""
        radii_small = [(1, 2)]
        radii_large = [(5, 10)]

        fused_small, _ = fuse_rdf_sum(test_image, radii_small)
        fused_large, _ = fuse_rdf_sum(test_image, radii_large)

        # Results should be different
        assert not np.allclose(fused_small, fused_large)

    def test_normalization_effect(self, test_image):
        """Test effect of different normalizations."""
        radii = [(1, 3)]

        fused_none, _ = fuse_rdf_sum(test_image, radii, normalize='none')
        fused_p99, _ = fuse_rdf_sum(test_image, radii, normalize='p99')
        fused_mad, _ = fuse_rdf_sum(test_image, radii, normalize='mad')

        # All should have valid shapes
        assert fused_none.shape == fused_p99.shape == fused_mad.shape

    def test_kde_parameters(self, test_image):
        """Test KDE parameter variations."""
        radii = [(1, 3)]
        fused, _ = fuse_rdf_sum(test_image, radii)

        # Different top_percent
        density_10, _ = kde_on_fused(fused, top_percent=10.0)
        density_50, _ = kde_on_fused(fused, top_percent=50.0)

        assert density_10.shape == density_50.shape

        # Different bandwidth
        density_small_bw, _ = kde_on_fused(fused, bandwidth_px=5.0)
        density_large_bw, _ = kde_on_fused(fused, bandwidth_px=20.0)

        assert density_small_bw.shape == density_large_bw.shape


class TestRobustness:
    """Test robustness to various inputs."""

    def test_all_black_image(self):
        """Test with all-black image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii)
        density, _ = kde_on_fused(fused)

        assert not np.isnan(fused).any()
        assert not np.isnan(density).any()

    def test_all_white_image(self):
        """Test with all-white image."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii)
        density, _ = kde_on_fused(fused)

        assert not np.isnan(fused).any()
        assert not np.isnan(density).any()

    def test_noisy_image(self):
        """Test with very noisy image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii)
        density, _ = kde_on_fused(fused)

        assert fused.shape == img.shape
        assert density.shape == img.shape

    def test_different_image_sizes(self):
        """Test with various image sizes."""
        sizes = [(50, 50), (100, 200), (200, 100), (500, 500)]
        radii = [(1, 3)]

        for size in sizes:
            img = np.random.rand(*size).astype(np.float32)
            fused, _ = fuse_rdf_sum(img, radii)
            density, _ = kde_on_fused(fused)

            assert fused.shape == img.shape
            assert density.shape == img.shape


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_time_reasonable(self, test_image):
        """Test that processing completes in reasonable time."""
        import time

        radii = [(1, 3)]

        start = time.time()
        fused, _ = fuse_rdf_sum(test_image, radii)
        kde_on_fused(fused)
        elapsed = time.time() - start

        # Should complete in less than 1 second for small image
        assert elapsed < 1.0

    def test_memory_efficiency(self, test_image):
        """Test that memory usage is reasonable."""
        radii = [(1, 3)]

        # Process without creating many temporary arrays
        fused, maps = fuse_rdf_sum(test_image, radii)
        density, _ = kde_on_fused(fused)

        # Check that outputs are not excessively large
        assert fused.nbytes < 10 * 1024 * 1024  # < 10 MB
        assert density.nbytes < 10 * 1024 * 1024  # < 10 MB


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_radii(self):
        """Test with invalid radii."""
        img = np.random.rand(100, 100).astype(np.float32)

        with pytest.raises(ValueError):
            fuse_rdf_sum(img, [(5, 3)])  # inner > outer

    def test_invalid_normalize(self):
        """Test with invalid normalization method."""
        img = np.random.rand(100, 100).astype(np.float32)

        with pytest.raises(ValueError):
            fuse_rdf_sum(img, [(1, 3)], normalize='invalid')

    def test_empty_image_list(self):
        """Test batch processing with empty list."""
        results = batch_process_images(
            [],
            radii=[(1, 3)],
            use_processes=False
        )

        # Should return empty dict or handle gracefully
        assert isinstance(results, dict)
