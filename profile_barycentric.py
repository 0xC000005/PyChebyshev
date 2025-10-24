"""Profile barycentric implementation to compare old vs new performance."""

import numpy as np
import time
from chebyshev_barycentric import ChebyshevApproximation

def dummy_func(x, _):
    return sum(x)

# Build 5D approximation (like in our tests)
print("Building 5D approximation...")
cheb = ChebyshevApproximation(
    dummy_func, 5,
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    [11, 11, 11, 11, 11],
    max_derivative_order=0
)
cheb.build()

# Test point (avoid coinciding with Chebyshev nodes at 0.5)
point = [0.52, 0.48, 0.53, 0.47, 0.51]
n_evals = 10

print(f"\n{'='*70}")
print("Performance Comparison: eval() vs fast_eval()")
print(f"{'='*70}")

# Warm up JIT compilation
for _ in range(100):
    cheb.fast_eval(point, [0, 0, 0, 0, 0])

# Test original eval()
print("\nTesting eval() (with validation)...")
start = time.time()
for _ in range(n_evals):
    result = cheb.eval(point, [0, 0, 0, 0, 0])
elapsed_eval = time.time() - start
eval_time_us = elapsed_eval / n_evals * 1e6

print(f"  Time per eval: {eval_time_us:.2f} µs")
print(f"  Throughput: {n_evals/elapsed_eval:.0f} evals/sec")

# Test fast_eval()
print("\nTesting fast_eval() (JIT + pre-allocated cache)...")
start = time.time()
for _ in range(n_evals):
    result_fast = cheb.fast_eval(point, [0, 0, 0, 0, 0])
elapsed_fast = time.time() - start
fast_time_us = elapsed_fast / n_evals * 1e6

print(f"  Time per eval: {fast_time_us:.2f} µs")
print(f"  Throughput: {n_evals/elapsed_fast:.0f} evals/sec")

# Speedup
speedup = elapsed_eval / elapsed_fast
print(f"\n{'='*70}")
print(f"SPEEDUP: {speedup:.1f}× faster")
print(f"{'='*70}")

# Verify results match
print(f"\nVerifying accuracy...")
print(f"  eval() result:      {result:.10f}")
print(f"  fast_eval() result: {result_fast:.10f}")
print(f"  Difference:         {abs(result - result_fast):.2e}")
if abs(result - result_fast) < 1e-12:
    print("  ✓ Results match!")
else:
    print("  ✗ Results differ!")
