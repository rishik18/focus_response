"""Tests for filters module."""

import numpy as np
import pytest
from focus_response.filters import (
    _ring_disk_masks,
    _get_coverage_masks,
    _auto_select_convolution_method,
    rdf_focus_numpy_edgesafe,
    rdf_multiscale,
    fuse_rdf_sum,
    _PRECOMPUTED_KERNELS
)


class TestRingDiskMasks:
    """Test ring and disk mask generation."""

    def test_valid_masks(self):
        """Test that valid masks are created."""
        disk, ring = _ring_disk_masks(1, 3)

        assert disk.shape == ring.shape
        assert disk.dtype == np.float32
        assert ring.dtype == np.float32
        assert disk.sum() > 0
        assert ring.sum() > 0

    def test_mask_properties(self):
        """Test mathematical properties of masks."""
        disk, ring = _ring_disk_masks(2, 5)

        # Disk and ring should not overlap
        overlap = (disk > 0) & (ring > 0)
        assert overlap.sum() == 0

        # Ring should be larger than disk
        assert ring.sum() > disk.sum()

    def test_invalid_radii(self):
        """Test that invalid radii raise errors."""
        with pytest.raises(ValueError):
            _ring_disk_masks(-1, 3)  # Negative inner radius

        with pytest.raises(ValueError):
            _ring_disk_masks(5, 3)  # Inner >= outer

    def test_caching(self):
        """Test that masks are cached."""
        # Clear cache and check it works
        key = (1, 3)
        assert key in _PRECOMPUTED_KERNELS

        disk1, ring1 = _ring_disk_masks(1, 3)
        disk2, ring2 = _ring_disk_masks(1, 3)

        # Should return same objects (cached)
        assert disk1 is disk2
        assert ring1 is ring2


class TestCoverageMasks:
    """Test coverage mask generation."""

    def test_coverage_masks_shape(self):
        """Test coverage masks have correct shape."""
        img_shape = (100, 100)
        cnt_ring, cnt_disk = _get_coverage_masks(img_shape, 1, 3)

        assert cnt_ring.shape == img_shape
        assert cnt_disk.shape == img_shape

    def test_coverage_masks_values(self):
        """Test coverage masks have valid values."""
        img_shape = (100, 100)
        cnt_ring, cnt_disk = _get_coverage_masks(img_shape, 2, 5)

        # Center should have full coverage
        center_y, center_x = 50, 50
        assert cnt_ring[center_y, center_x] > 0
        assert cnt_disk[center_y, center_x] > 0


class TestAutoSelectConvolutionMethod:
    """Test automatic convolution method selection."""

    def test_small_kernel_opencv(self):
        """Small kernels should use OpenCV."""
        method = _auto_select_convolution_method(7, (1000, 1000))
        assert method == 'opencv'

    def test_medium_kernel_scipy(self):
        """Medium kernels should use scipy."""
        method = _auto_select_convolution_method(11, (1000, 1000))
        assert method == 'scipy'

    def test_large_kernel_fft(self):
        """Large kernels should use FFT."""
        method = _auto_select_convolution_method(20, (1000, 1000))
        assert method == 'fft'

    def test_large_image_fft(self):
        """Large images should use FFT."""
        method = _auto_select_convolution_method(10, (3000, 3000))
        # Large images use either scipy or fft
        assert method in ['scipy', 'fft']


class TestRDFFocus:
    """Test RDF focus computation."""

    def test_rdf_basic(self):
        """Test basic RDF computation."""
        # Create simple test image
        img = np.random.rand(100, 100).astype(np.float32)
        result = rdf_focus_numpy_edgesafe(img, 1, 3, power=2)

        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert not np.isnan(result).any()

    def test_rdf_grayscale(self):
        """Test RDF on grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = rdf_focus_numpy_edgesafe(img, 1, 3)

        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_rdf_rgb(self):
        """Test RDF on RGB image."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = rdf_focus_numpy_edgesafe(img, 1, 3)

        assert result.shape == img.shape[:2]
        assert result.dtype == np.float32

    def test_rdf_power_parameter(self):
        """Test different power parameters."""
        img = np.random.rand(100, 100).astype(np.float32)

        result_p1 = rdf_focus_numpy_edgesafe(img, 1, 3, power=1)
        result_p2 = rdf_focus_numpy_edgesafe(img, 1, 3, power=2)

        # Power 2 should generally give different values
        assert not np.allclose(result_p1, result_p2)

    def test_rdf_auto_method_selection(self):
        """Test auto method selection."""
        img = np.random.rand(100, 100).astype(np.float32)

        # Auto-detect (use_fft=None)
        result_auto = rdf_focus_numpy_edgesafe(img, 1, 3, use_fft=None)

        # OpenCV (use_fft=False)
        result_opencv = rdf_focus_numpy_edgesafe(img, 1, 3, use_fft=False)

        # Results should be similar
        assert np.allclose(result_auto, result_opencv, rtol=1e-5)


class TestMultiscaleRDF:
    """Test multi-scale RDF computation."""

    def test_multiscale_single(self):
        """Test multi-scale with single scale."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        maps = rdf_multiscale(img, radii)

        assert len(maps) == 1
        assert maps[0].shape == img.shape

    def test_multiscale_multiple(self):
        """Test multi-scale with multiple scales."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3), (2, 5), (3, 7)]

        maps = rdf_multiscale(img, radii)

        assert len(maps) == 3
        for m in maps:
            assert m.shape == img.shape

    def test_multiscale_parallel(self):
        """Test parallel processing."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3), (2, 5)]

        maps_parallel = rdf_multiscale(img, radii, parallel=True)
        maps_sequential = rdf_multiscale(img, radii, parallel=False)

        assert len(maps_parallel) == len(maps_sequential)
        for m1, m2 in zip(maps_parallel, maps_sequential):
            assert np.allclose(m1, m2, rtol=1e-5)


class TestFuseRDFSum:
    """Test RDF fusion."""

    def test_fuse_basic(self):
        """Test basic fusion."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        fused, maps = fuse_rdf_sum(img, radii)

        assert fused.shape == img.shape
        assert fused.dtype == np.float32
        assert len(maps) == 1

    def test_fuse_normalization_none(self):
        """Test no normalization."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii, normalize='none')
        assert fused.dtype == np.float32

    def test_fuse_normalization_p99(self):
        """Test p99 normalization."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii, normalize='p99')
        assert fused.dtype == np.float32

    def test_fuse_normalization_mad(self):
        """Test MAD normalization."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii, normalize='mad')
        assert fused.dtype == np.float32

    def test_fuse_invalid_normalize(self):
        """Test invalid normalization raises error."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        with pytest.raises(ValueError):
            fuse_rdf_sum(img, radii, normalize='invalid')

    def test_fuse_downsample(self):
        """Test downsampling."""
        img = np.random.rand(200, 200).astype(np.float32)
        radii = [(1, 3)]

        fused, _ = fuse_rdf_sum(img, radii, downsample=2)

        # Should return original size despite downsampling
        assert fused.shape == img.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_image(self):
        """Test with very small image."""
        img = np.random.rand(10, 10).astype(np.float32)
        result = rdf_focus_numpy_edgesafe(img, 1, 2)

        assert result.shape == img.shape

    def test_large_radii(self):
        """Test with large radii relative to image."""
        img = np.random.rand(50, 50).astype(np.float32)
        # This should work but have edge effects
        result = rdf_focus_numpy_edgesafe(img, 5, 15)

        assert result.shape == img.shape

    def test_constant_image(self):
        """Test with constant image."""
        img = np.ones((100, 100), dtype=np.float32)
        result = rdf_focus_numpy_edgesafe(img, 1, 3)

        # Constant image should give zero response
        assert np.abs(result).max() < 1e-5
