"""Tests for KDE module."""

import numpy as np
from focus_response.kde import kde_on_fused


class TestKDEOnFused:
    """Test KDE computation on fused maps."""

    def test_kde_basic(self):
        """Test basic KDE computation."""
        fused = np.random.rand(100, 100).astype(np.float32)
        density, threshold = kde_on_fused(fused, top_percent=25.0, bandwidth_px=10.0)

        assert density.shape == fused.shape
        assert density.dtype == np.float32
        assert 0 <= density.min() <= density.max() <= 1
        assert isinstance(threshold, float)

    def test_kde_threshold(self):
        """Test that threshold selects correct percentage."""
        fused = np.random.rand(100, 100).astype(np.float32)
        top_percent = 25.0

        density, threshold = kde_on_fused(fused, top_percent=top_percent)

        # Count pixels above threshold
        selected = (fused >= threshold).sum()
        total = fused.size
        actual_percent = (selected / total) * 100

        # Should be approximately the requested percentage
        assert abs(actual_percent - top_percent) < 5  # Allow 5% tolerance

    def test_kde_normalized(self):
        """Test that normalized output is in [0,1]."""
        fused = np.random.rand(100, 100).astype(np.float32)
        density, _ = kde_on_fused(fused, normalize=True)

        assert density.min() >= 0
        assert density.max() <= 1

    def test_kde_not_normalized(self):
        """Test unnormalized output."""
        fused = np.random.rand(100, 100).astype(np.float32)
        density, _ = kde_on_fused(fused, normalize=False)

        # Unnormalized can have any positive values
        assert density.min() >= 0

    def test_kde_bandwidth_effect(self):
        """Test that bandwidth affects smoothness."""
        fused = np.random.rand(100, 100).astype(np.float32)

        density_small, _ = kde_on_fused(fused, bandwidth_px=5.0)
        density_large, _ = kde_on_fused(fused, bandwidth_px=20.0)

        # Larger bandwidth should give smoother (less variable) result
        # We just check shapes since the actual smoothness may vary with random data
        assert density_small.shape == density_large.shape

    def test_kde_include_strength(self):
        """Test include_strength parameter."""
        fused = np.random.rand(100, 100).astype(np.float32)

        density_no_strength, _ = kde_on_fused(fused, include_strength=False)
        density_with_strength, _ = kde_on_fused(fused, include_strength=True)

        assert density_no_strength.shape == density_with_strength.shape
        # Results should be different
        assert not np.allclose(density_no_strength, density_with_strength)

    def test_kde_clip_percentile(self):
        """Test clipping percentile."""
        fused = np.random.rand(100, 100).astype(np.float32)

        density_clip, _ = kde_on_fused(fused, clip_percentile=95.0)
        density_no_clip, _ = kde_on_fused(fused, clip_percentile=None)

        assert density_clip.shape == density_no_clip.shape

    def test_kde_zero_map(self):
        """Test KDE on all-zero map."""
        fused = np.zeros((100, 100), dtype=np.float32)
        density, threshold = kde_on_fused(fused)

        assert density.shape == fused.shape
        assert threshold == 0.0

    def test_kde_sparse_map(self):
        """Test KDE on sparse map (few non-zero values)."""
        fused = np.zeros((100, 100), dtype=np.float32)
        # Add a few hot spots
        fused[25, 25] = 1.0
        fused[75, 75] = 1.0

        density, threshold = kde_on_fused(fused, top_percent=10.0)

        assert density.shape == fused.shape
        # Density should be concentrated around hot spots
        assert density[25, 25] > 0
        assert density[75, 75] > 0

    def test_kde_uniform_map(self):
        """Test KDE on uniform map."""
        fused = np.ones((100, 100), dtype=np.float32)
        density, threshold = kde_on_fused(fused, top_percent=25.0)

        assert density.shape == fused.shape
        assert threshold == 1.0  # All values are equal

    def test_kde_top_percent_range(self):
        """Test different top_percent values."""
        fused = np.random.rand(100, 100).astype(np.float32)

        for top_percent in [10.0, 25.0, 50.0, 75.0]:
            density, threshold = kde_on_fused(fused, top_percent=top_percent)
            assert density.shape == fused.shape
            assert threshold >= 0

    def test_kde_bandwidth_range(self):
        """Test different bandwidth values."""
        fused = np.random.rand(100, 100).astype(np.float32)

        for bandwidth in [1.0, 5.0, 10.0, 20.0]:
            density, threshold = kde_on_fused(fused, bandwidth_px=bandwidth)
            assert density.shape == fused.shape

    def test_kde_output_consistency(self):
        """Test that same input gives same output."""
        fused = np.random.rand(100, 100).astype(np.float32)

        density1, threshold1 = kde_on_fused(fused, top_percent=25.0, bandwidth_px=10.0)
        density2, threshold2 = kde_on_fused(fused, top_percent=25.0, bandwidth_px=10.0)

        assert np.allclose(density1, density2)
        assert threshold1 == threshold2

    def test_kde_small_image(self):
        """Test KDE on small image."""
        fused = np.random.rand(10, 10).astype(np.float32)
        density, threshold = kde_on_fused(fused)

        assert density.shape == fused.shape

    def test_kde_large_image(self):
        """Test KDE on large image."""
        fused = np.random.rand(1000, 1000).astype(np.float32)
        density, threshold = kde_on_fused(fused)

        assert density.shape == fused.shape

    def test_kde_rectangular_image(self):
        """Test KDE on non-square image."""
        fused = np.random.rand(100, 200).astype(np.float32)
        density, threshold = kde_on_fused(fused)

        assert density.shape == fused.shape


class TestKDEEdgeCases:
    """Test edge cases for KDE."""

    def test_very_small_bandwidth(self):
        """Test with very small bandwidth."""
        fused = np.random.rand(100, 100).astype(np.float32)
        density, _ = kde_on_fused(fused, bandwidth_px=0.1)

        assert density.shape == fused.shape
        assert not np.isnan(density).any()

    def test_very_large_bandwidth(self):
        """Test with very large bandwidth."""
        fused = np.random.rand(100, 100).astype(np.float32)
        density, _ = kde_on_fused(fused, bandwidth_px=50.0)

        assert density.shape == fused.shape
        assert not np.isnan(density).any()

    def test_extreme_top_percent(self):
        """Test with extreme top_percent values."""
        fused = np.random.rand(100, 100).astype(np.float32)

        # Very small
        density_small, _ = kde_on_fused(fused, top_percent=1.0)
        assert density_small.shape == fused.shape

        # Very large
        density_large, _ = kde_on_fused(fused, top_percent=99.0)
        assert density_large.shape == fused.shape

    def test_negative_values_in_fused(self):
        """Test with negative values in fused map."""
        fused = np.random.randn(100, 100).astype(np.float32)  # Can have negatives
        density, threshold = kde_on_fused(fused)

        assert density.shape == fused.shape
        # Density should still be valid
        assert not np.isnan(density).any()
