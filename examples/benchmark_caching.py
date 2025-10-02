"""Benchmark to demonstrate kernel caching improvement."""

import time
import numpy as np

# Test kernel access speed
from focus_response.filters import _ring_disk_masks

print("=" * 60)
print("Kernel Caching Benchmark")
print("=" * 60)

# Test 1: Pre-cached kernel (1,3) - the default
print("\nTest 1: Default kernel (1,3) - Pre-cached")
times = []
for i in range(100):
    start = time.perf_counter()
    disk, ring = _ring_disk_masks(1, 3)
    times.append(time.perf_counter() - start)

avg_time_cached = np.mean(times[1:]) * 1e6  # Convert to microseconds
print(f"Average access time: {avg_time_cached:.2f} µs")
print(f"Min: {np.min(times[1:]) * 1e6:.2f} µs, Max: {np.max(times[1:]) * 1e6:.2f} µs")

# Test 2: First-time computation (non-cached)
print("\nTest 2: New kernel (5,9) - First access (not pre-cached)")
start = time.perf_counter()
disk, ring = _ring_disk_masks(5, 9)
first_time = (time.perf_counter() - start) * 1e6
print(f"First access time: {first_time:.2f} µs")

# Test 3: Cached access after first computation
print("\nTest 3: Same kernel (5,9) - Cached after first access")
times = []
for i in range(100):
    start = time.perf_counter()
    disk, ring = _ring_disk_masks(5, 9)
    times.append(time.perf_counter() - start)

avg_time_after = np.mean(times) * 1e6
print(f"Average access time: {avg_time_after:.2f} µs")
print(f"Speedup vs first access: {first_time / avg_time_after:.1f}x")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Pre-cached kernels: 6 common sizes")
print(f"Cache lookup time: ~{avg_time_cached:.1f} µs (essentially instant)")
print(f"First computation: ~{first_time:.1f} µs")
print(f"Cache speedup: ~{first_time / avg_time_cached:.0f}x faster")
print("\nConclusion: Pre-cached kernels provide near-instant access!")
print("=" * 60)
