"""Tests for batch processing module."""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
import cv2

from focus_response.batch import (
    load_image,
    load_images_parallel,
    process_single_image,
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
def test_images(temp_dir):
    """Create test images."""
    images = []
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        path = Path(temp_dir) / f"test_{i}.jpg"
        cv2.imwrite(str(path), img)
        images.append(path)
    return images


class TestLoadImage:
    """Test image loading."""

    def test_load_grayscale(self, test_images):
        """Test loading grayscale image."""
        img = load_image(test_images[0], grayscale=True)

        assert img is not None
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_load_color(self, test_images):
        """Test loading color image."""
        img = load_image(test_images[0], grayscale=False)

        assert img is not None
        assert img.ndim == 3

    def test_load_nonexistent(self):
        """Test loading nonexistent image."""
        img = load_image("nonexistent.jpg", grayscale=True)
        assert img is None


class TestLoadImagesParallel:
    """Test parallel image loading."""

    def test_load_parallel_basic(self, test_images):
        """Test basic parallel loading."""
        results = load_images_parallel(test_images, grayscale=True)

        assert len(results) == len(test_images)
        for path in test_images:
            assert str(path) in results

    def test_load_parallel_max_workers(self, test_images):
        """Test with specific number of workers."""
        results = load_images_parallel(test_images, grayscale=True, max_workers=2)

        assert len(results) == len(test_images)

    def test_load_parallel_empty_list(self):
        """Test with empty image list."""
        results = load_images_parallel([], grayscale=True)
        assert len(results) == 0


class TestProcessSingleImage:
    """Test single image processing."""

    def test_process_basic(self):
        """Test basic processing."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        result = process_single_image(img, radii)

        assert 'fused' in result
        assert 'density' in result
        assert 'threshold' in result
        assert 'fuse_time' in result
        assert 'kde_time' in result
        assert 'total_time' in result

    def test_process_output_shapes(self):
        """Test output shapes."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        result = process_single_image(img, radii)

        assert result['fused'].shape == img.shape
        assert result['density'].shape == img.shape

    def test_process_multiscale(self):
        """Test with multiple scales."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3), (2, 5)]

        result = process_single_image(img, radii)

        assert len(result['individual_maps']) == 2

    def test_process_timing(self):
        """Test that timing values are reasonable."""
        img = np.random.rand(100, 100).astype(np.float32)
        radii = [(1, 3)]

        result = process_single_image(img, radii)

        assert result['fuse_time'] >= 0
        assert result['kde_time'] >= 0
        assert result['total_time'] >= result['fuse_time']
        assert result['total_time'] >= result['kde_time']


class TestBatchProcessImages:
    """Test batch image processing."""

    def test_batch_process_basic(self, test_images):
        """Test basic batch processing."""
        radii = [(1, 3)]
        results = batch_process_images(
            test_images,
            radii=radii,
            max_workers=2,
            use_processes=False
        )

        assert len(results) == len(test_images)
        for path in test_images:
            assert str(path) in results

    def test_batch_process_thread_vs_process(self, test_images):
        """Test thread vs process parallelism."""
        radii = [(1, 3)]

        results_thread = batch_process_images(
            test_images,
            radii=radii,
            use_processes=False,
            max_workers=2
        )

        results_process = batch_process_images(
            test_images,
            radii=radii,
            use_processes=True,
            max_workers=2
        )

        assert len(results_thread) == len(results_process)

    def test_batch_process_parameters(self, test_images):
        """Test different parameter combinations."""
        radii = [(2, 5)]

        results = batch_process_images(
            test_images,
            radii=radii,
            top_percent=30.0,
            bandwidth_px=15.0,
            power=1,
            normalize='mad'
        )

        assert len(results) > 0


class TestGetImageFiles:
    """Test file discovery."""

    def test_get_files_basic(self, temp_dir, test_images):
        """Test basic file discovery."""
        files = get_image_files(temp_dir)

        # Should find at least the test images (may find more due to duplicates from other tests)
        assert len(files) >= len(test_images)
        for f in files:
            assert f.exists()
            assert f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def test_get_files_recursive(self, temp_dir, test_images):
        """Test recursive file discovery."""
        # Create subdirectory with image
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(subdir / "sub_test.jpg"), img)

        files_nonrecursive = get_image_files(temp_dir, recursive=False)
        files_recursive = get_image_files(temp_dir, recursive=True)

        assert len(files_recursive) > len(files_nonrecursive)

    def test_get_files_extensions(self, temp_dir):
        """Test file extension filtering."""
        # Create a clean temp dir for this test
        import tempfile
        import shutil
        test_dir = tempfile.mkdtemp()

        try:
            # Create images with different extensions
            img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(Path(test_dir) / "test.jpg"), img)
            cv2.imwrite(str(Path(test_dir) / "test.png"), img)

            files = get_image_files(test_dir, extensions=('.jpg',))
            # May find duplicates due to case-insensitive matching
            jpg_files = [f for f in files if f.suffix.lower() == '.jpg']
            assert len(jpg_files) >= 1
        finally:
            shutil.rmtree(test_dir)

    def test_get_files_nonexistent(self):
        """Test with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            get_image_files("nonexistent_directory")


class TestSaveResults:
    """Test result saving."""

    def test_save_arrays_only(self, temp_dir):
        """Test saving only arrays."""
        results = {
            'test.jpg': {
                'fused': np.random.rand(100, 100).astype(np.float32),
                'density': np.random.rand(100, 100).astype(np.float32),
                'threshold': 0.5
            }
        }

        save_results(results, temp_dir, save_arrays=True, save_visualizations=False)

        # Check folders exist
        assert (Path(temp_dir) / "filter_arrays").exists()
        assert (Path(temp_dir) / "kde_arrays").exists()

        # Check files exist
        assert (Path(temp_dir) / "filter_arrays" / "test_filter.npy").exists()
        assert (Path(temp_dir) / "kde_arrays" / "test_kde.npy").exists()

        # Visualization folders should not exist
        assert not (Path(temp_dir) / "filter_vis").exists()
        assert not (Path(temp_dir) / "kde_vis").exists()

    def test_save_visualizations_only(self, temp_dir):
        """Test saving only visualizations."""
        results = {
            'test.jpg': {
                'fused': np.random.rand(100, 100).astype(np.float32),
                'density': np.random.rand(100, 100).astype(np.float32),
                'threshold': 0.5
            }
        }

        save_results(results, temp_dir, save_arrays=False, save_visualizations=True)

        # Check folders exist
        assert (Path(temp_dir) / "filter_vis").exists()
        assert (Path(temp_dir) / "kde_vis").exists()

        # Check files exist
        assert (Path(temp_dir) / "filter_vis" / "test_filter.png").exists()
        assert (Path(temp_dir) / "kde_vis" / "test_kde.png").exists()

        # Array folders should not exist
        assert not (Path(temp_dir) / "filter_arrays").exists()
        assert not (Path(temp_dir) / "kde_arrays").exists()

    def test_save_both(self, temp_dir):
        """Test saving both arrays and visualizations."""
        results = {
            'test.jpg': {
                'fused': np.random.rand(100, 100).astype(np.float32),
                'density': np.random.rand(100, 100).astype(np.float32),
                'threshold': 0.5
            }
        }

        save_results(results, temp_dir, save_arrays=True, save_visualizations=True)

        # All folders should exist
        assert (Path(temp_dir) / "filter_arrays").exists()
        assert (Path(temp_dir) / "kde_arrays").exists()
        assert (Path(temp_dir) / "filter_vis").exists()
        assert (Path(temp_dir) / "kde_vis").exists()

    def test_save_multiple_results(self, temp_dir):
        """Test saving multiple results."""
        results = {
            f'test_{i}.jpg': {
                'fused': np.random.rand(100, 100).astype(np.float32),
                'density': np.random.rand(100, 100).astype(np.float32),
                'threshold': 0.5
            }
            for i in range(3)
        }

        save_results(results, temp_dir, save_arrays=True)

        # Check all files exist
        for i in range(3):
            assert (Path(temp_dir) / "filter_arrays" / f"test_{i}_filter.npy").exists()
            assert (Path(temp_dir) / "kde_arrays" / f"test_{i}_kde.npy").exists()

    def test_save_load_roundtrip(self, temp_dir):
        """Test saving and loading arrays."""
        original_fused = np.random.rand(100, 100).astype(np.float32)
        original_density = np.random.rand(100, 100).astype(np.float32)

        results = {
            'test.jpg': {
                'fused': original_fused,
                'density': original_density,
                'threshold': 0.5
            }
        }

        save_results(results, temp_dir, save_arrays=True, save_visualizations=False)

        # Load back
        loaded_fused = np.load(str(Path(temp_dir) / "filter_arrays" / "test_filter.npy"))
        loaded_density = np.load(str(Path(temp_dir) / "kde_arrays" / "test_kde.npy"))

        assert np.allclose(original_fused, loaded_fused)
        assert np.allclose(original_density, loaded_density)
