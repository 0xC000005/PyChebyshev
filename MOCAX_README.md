# MoCaX Suite - Quick Start

Multi-dimensional Chebyshev approximation for accelerating option pricing and risk calculations.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Extract and setup (one-time)
cd MoCaXSuite-1.2.0/MoCaXSuite-1.2.0/MoCaX/Linux/gmake/64bit/Python/MoCaXLibrary
unzip -q mocaxpy-4.3.1.linux-x86_64.zip
mkdir -p mocax_lib
cp -r usr/lib/python2.7/site-packages/mocaxpy mocax_lib/
cp libmocaxc.so mocax_lib/
cd /path/to/FinRegressor
cp -r [path-to-mocax_lib] .

# 2. Test installation
./run_mocax_test.sh
```

Expected output: ✅ 3 tests PASSED in ~2 minutes

## 📚 Documentation

### Essential Reading

1. **[MOCAX_SETUP_GUIDE.md](MOCAX_SETUP_GUIDE.md)** (942 lines) - **START HERE**
   - Complete installation guide
   - Step-by-step troubleshooting
   - Usage examples and code templates
   - Performance tuning guide
   - Quick reference card

2. **[MOCAX_5D_RESULTS.md](MOCAX_5D_RESULTS.md)** (299 lines)
   - Comprehensive 5D Black-Scholes test results
   - Spectral accuracy demonstration (0.000% price error)
   - Comparison: MoCaX vs linear interpolation
   - Greeks accuracy and performance benchmarks

3. **[CLAUDE.md](CLAUDE.md)** - Project documentation
   - MoCaX setup section (lines 54-104)
   - Test results overview
   - Integration with project architecture

### Reference Material

- **MoCaX User Manual**: `MoCaXSuite-1.2.0/MoCaX/MoCaXDocumentationPDF/`
- **Example Code**: `mocax_test.py` (590 lines with 3 comprehensive tests)
- **Helper Script**: `run_mocax_test.sh` (automatic environment setup)

## 🎯 What You Get

### Test 1: Simple 3D Function
- Validates basic Chebyshev approximation
- 0.024% error, 1e-7 derivative accuracy
- Build time: ~1ms

### Test 2: Black-Scholes 3D (S, T, σ)
- 3D parametric pricing
- 1,800 evaluations in 10ms
- <0.01% price error, ~2% Vega error

### Test 3: 5D Parametric Black-Scholes ⭐ (S, K, T, σ, r)
**The showcase test** - same challenging case from our tensor demo

| Metric | Result |
|--------|--------|
| **Build** | 161,051 evals in ~1.5s |
| **Price accuracy** | **0.000%** (spectral!) |
| **Delta/Gamma/Rho** | 0.000% error |
| **Vega** | 1.98% vs 7.60% (linear interp) |
| **Method** | Analytical derivatives |

**Key insight**: True Chebyshev polynomial evaluation vs piecewise linear interpolation gives 3-4× better accuracy on Greeks.

## 🔍 Key Comparisons

### Our Tensor Approach (`chebyshev_tensor_demo.py`)
- ✅ Smart sampling (Chebyshev nodes)
- ✅ Compression (CP decomposition)
- ❌ Linear interpolation (loses smoothness)
- ❌ Finite difference Greeks (numerical errors)
- **Result**: 3.22% mean Vega error, 7.60% max

### MoCaX Approach
- ✅ Smart sampling (Chebyshev nodes)
- ✅ Compression (coefficient representation)
- ✅ Polynomial interpolation (spectral accuracy)
- ✅ Analytical Greeks (Chebyshev differentiation)
- **Result**: 1.98% Vega error, 0.00% Rho error

## 💡 When to Use MoCaX

### ✅ Use MoCaX When:
- No analytical formula exists (exotic options, early exercise)
- Expensive PDE solvers (FDM: 100ms-10s per solve)
- Monte Carlo simulations (thousands of paths)
- Need many queries (>100) to amortize build cost
- Need Greeks across multiple parameters
- Want <1% accuracy on derivatives

### ❌ Don't Use When:
- Fast analytical formula exists (Black-Scholes: <1μs)
- Single or few queries
- Can tolerate 5-10% error
- Simple implementation more important than speed

## 📖 Learning Path

### Beginner (30 minutes)
1. Read: MOCAX_SETUP_GUIDE.md "Quick Test" section
2. Run: `./run_mocax_test.sh`
3. Study: Test 1 output (simple 3D function)

### Intermediate (1 hour)
1. Read: MOCAX_SETUP_GUIDE.md "Usage Examples"
2. Modify: `mocax_test.py` Test 2 parameters
3. Run: Custom 3D Black-Scholes test

### Advanced (2-3 hours)
1. Read: MOCAX_5D_RESULTS.md (detailed analysis)
2. Study: Test 3 implementation (5D parametric)
3. Compare: Results vs tensor interpolation demo
4. Implement: Your own 5D pricing function

### Expert (1 week)
1. Read: MoCaX User Manual (all chapters)
2. Study: CHEBYSHEV_ACCELERATION.md (190 pages of research)
3. Explore: Tensor Train methods (MoCaXExtend)
4. Build: Production pricing system

## 🎓 Key Concepts

### Chebyshev Nodes
Optimal sampling points for polynomial interpolation:
```
xᵢ = cos(iπ/n)  mapped to [a, b]
```
- Clustered near boundaries
- Minimize interpolation error (Runge phenomenon)

### Analytical Derivatives
```
∂V/∂xᵢ = Σⱼ cⱼ · ∂Tⱼ(xᵢ)/∂xᵢ

Since T'ₙ(x) = n·Uₙ₋₁(x) is known, derivatives are exact
```
- No finite difference errors
- No need to re-evaluate function
- Mixed derivatives: ∂²V/∂xᵢ∂xⱼ directly available

### Spectral Accuracy
```
Error ≈ C · ρ⁻ⁿ  (exponential convergence)
```
- For smooth functions, error decreases exponentially with nodes
- Black-Scholes: 0.000% error with 11 nodes per dimension
- Much faster convergence than finite differences (O(h²))

## 🚦 Common Issues

| Issue | Solution |
|-------|----------|
| "cannot open shared object" | `export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"` |
| "ModuleNotFoundError" | `sys.path.insert(0, 'mocax_lib')` before import |
| "'module' object is not callable" | Use `BlackScholesCall(...)` not `bs.call(...)` |
| Slow build times | Reduce nodes: `[11,11,9,9,7]` instead of `[15,15,15,15,15]` |
| Low accuracy | Increase nodes or check domain boundaries |

See MOCAX_SETUP_GUIDE.md "Troubleshooting" for detailed solutions.

## 📊 Performance Benchmarks

### Build Phase (Offline)
| Dimensions | Nodes | Evaluations | Time |
|------------|-------|-------------|------|
| 3D | 15×12×10 | 1,800 | 10ms |
| 5D | 11×11×11×11×11 | 161,051 | 1.5s |
| 5D (adaptive) | 11×11×9×9×7 | 68,607 | 0.4s |

### Query Phase (Online)
| Method | Time per Call | Use Case |
|--------|---------------|----------|
| MoCaX | 220μs | When no formula exists |
| Analytical | 1μs | When formula is simple |
| FDM | 500ms | Exotic/American options |

**Break-even**: ~3 queries if FDM alternative, Never if analytical exists

### Speedups from Literature
- Parametric option pricing: **10,000×** (arXiv:1505.04648)
- XVA calculations: **40,000×** (arXiv:1805.00898)
- SIMM initial margin: **20,000×** (arXiv:1808.08221)

## 🔗 Resources

### Project Files
- `mocax_test.py` - Comprehensive test suite (3 tests)
- `run_mocax_test.sh` - One-command testing
- `chebyshev_tensor_demo.py` - Our tensor approach (for comparison)

### Documentation
- `MOCAX_SETUP_GUIDE.md` - Installation and usage (942 lines)
- `MOCAX_5D_RESULTS.md` - Detailed test results (299 lines)
- `TENSOR_COMPARISON_SUMMARY.md` - Methods comparison
- `CHEBYSHEV_ACCELERATION.md` - Research survey (190+ pages)

### External
- Website: https://mocaxintelligence.org
- Textbook: "Machine Learning for Risk Calculations" (Ruiz & Zeron, Wiley)
- YouTube: youtube.com/mocax
- Papers: 7+ key arXiv papers analyzed in CHEBYSHEV_ACCELERATION.md

## 🎯 Quick Code Template

```python
import sys
sys.path.insert(0, 'mocax_lib')
import mocaxpy

# Your expensive function
def my_function(x, additional_data):
    """x = [param1, param2, ...]"""
    return expensive_calculation(x[0], x[1], ...)

# Setup
domain = mocaxpy.MocaxDomain([[a1, b1], [a2, b2], ...])
ns = mocaxpy.MocaxNs([n1, n2, ...])  # Nodes per dimension

# Build (offline phase)
mocax = mocaxpy.Mocax(
    my_function,
    num_dimensions,
    domain,
    None,  # error_threshold
    ns,
    max_derivative_order=2
)

# Query (online phase)
point = [100.0, 1.0, 0.25]

# Function value
deriv_id = mocax.get_derivative_id([0, 0, 0])
price = mocax.eval(point, deriv_id)

# First derivative
deriv_id = mocax.get_derivative_id([1, 0, 0])
delta = mocax.eval(point, deriv_id)

# Second derivative
deriv_id = mocax.get_derivative_id([2, 0, 0])
gamma = mocax.eval(point, deriv_id)

# Clean up
del mocax
```

## 📝 Citation

If using MoCaX in academic work:

```bibtex
@book{ruiz2020machine,
  title={Machine Learning for Risk Calculations},
  author={Ruiz, Ignacio and Zeron, Mariano},
  year={2020},
  publisher={Wiley}
}
```

## ✅ Next Steps

1. **Install**: Follow MOCAX_SETUP_GUIDE.md steps 1-5 (10 minutes)
2. **Verify**: Run `./run_mocax_test.sh` (2 minutes)
3. **Learn**: Study the 3 test implementations in `mocax_test.py`
4. **Analyze**: Read MOCAX_5D_RESULTS.md for detailed insights
5. **Build**: Create your own approximation for your pricing function

---

**Version**: 1.0
**Date**: 2025-10-22
**Status**: ✅ All tests passing

For detailed documentation, see [MOCAX_SETUP_GUIDE.md](MOCAX_SETUP_GUIDE.md)
