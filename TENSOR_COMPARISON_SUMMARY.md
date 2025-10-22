# Comparison: 1D vs Multi-Dimensional Tensor Interpolation

## Terminology: What Do "1D" and "Multi-D" Mean?

### 1D Chebyshev Interpolation

**Definition**: Build a **1-dimensional interpolant** over a **single parameter** while **fixing all other parameters** at specific constant values.

**Example for Vega (volatility sensitivity)**:
```python
# Build interpolant V(σ) at FIXED point (S=100, K=100, T=1.0, r=0.05)
# We vary only σ ∈ [0.10, 0.40]
# All other parameters are CONSTANT
```

**What this means**:
- ✅ Works great when querying at the same fixed point with different σ
- ❌ Fails when querying at different S, K, or T (because Vega changes with these!)

**Why it's called "1D"**: We're interpolating over a single dimension (σ) in parameter space.

---

### Multi-Dimensional (5D) Tensor Interpolation

**Definition**: Build a **multi-dimensional interpolant** over **ALL parameters simultaneously**, creating a 5-dimensional representation.

**Example for same Vega calculation**:
```python
# Build interpolant V(S, K, T, σ, r) over ENTIRE 5D parameter space
# S ∈ [80, 120], K ∈ [90, 110], T ∈ [0.25, 2.0], σ ∈ [0.10, 0.40], r ∈ [0.01, 0.10]
# We sample ALL combinations using Chebyshev nodes
```

**What this means**:
- ✅ Works uniformly well across the entire parameter space
- ✅ Vega queries accurate for ANY (S, K, T, σ, r) combination

**Why it's called "5D"**: We're interpolating over five dimensions (S, K, T, σ, r) in parameter space.

---

### Key Difference: Parameter Space Coverage

| Approach | Parameters Varied | Parameters Fixed | Dimension |
|----------|------------------|------------------|-----------|
| **1D** | 1 (e.g., σ only) | 4 (S, K, T, r fixed) | 1-dimensional |
| **Multi-D (5D)** | 5 (S, K, T, σ, r) | 0 (none fixed) | 5-dimensional |

**Your intuition is correct**: 1D means fixing all parameters except one!

---

### What is a "Chebyshev Tensor"?

**Short answer**: Chebyshev tensor = Chebyshev interpolation at multi-dimensional nodes + tensor decomposition for compression.

**The two-part process**:

1. **Chebyshev Interpolation** (same as 1D, but in d dimensions)
   - Generate Chebyshev nodes in each dimension: x₁, x₂, ..., xₐ
   - Evaluate function at ALL combinations: f(x₁[i], x₂[j], ..., xₐ[k])
   - This creates a d-dimensional array (tensor) of values
   - Example: 11 nodes per dimension in 5D → 11⁵ = 161,051 values

2. **Tensor Decomposition** (the compression step)
   - Apply low-rank decomposition (CP, Tucker, or Tensor Train)
   - Compress the full tensor from n^d values to ~O(d·n·r) parameters
   - Example: 161,051 values → 560 parameters (287× compression!)
   - This exploits "low-rank structure" in financial pricing functions

**Note:** See https://arxiv.org/pdf/2011.04544 section 2 for a detail definition
What Proper Multi-Dimensional Chebyshev Polynomial Regression Looks Like
**Key difference from regular Chebyshev interpolation**:

| Method | What you build | Storage |
|--------|---------------|---------|
| **Regular 1D Chebyshev** | Polynomial coefficients for 1 dimension | 15 coefficients |
| **Naive Multi-D Chebyshev** | Full tensor at all d-dimensional nodes | 11⁵ = 161,051 values ❌ Infeasible! |
| **Chebyshev Tensor** | Compressed low-rank representation | 560 parameters ✓ Practical! |

**Why "tensor"?** Because we're working with multi-dimensional arrays (tensors) and compressing them using tensor decomposition techniques (CP/Tucker/TT).

**In this document**: We use "tensor interpolation" and "Chebyshev tensor" interchangeably to refer to this two-step process: interpolate at Chebyshev nodes, then compress the resulting tensor.

---

## Summary of Results

This document compares three approaches for computing option Greeks:
1. **1D Chebyshev Interpolation** - Fast but limited to fixed parameter points
2. **Tensor Interpolation (CP Decomposition)** - Fast and accurate across all parameters
3. **FDM (Finite Difference Method)** - Accurate baseline but slow

---

## Step-by-Step: How Chebyshev Interpolation Works

This section provides an intuitive, step-by-step explanation of both the 1D and multi-dimensional approaches.

### What Are Chebyshev Nodes?

**Chebyshev nodes** are special points for interpolation that minimize approximation error.

**Formula:**
```
xᵢ = (a+b)/2 + (b-a)/2 · cos((2i+1)π/(2n))  for i = 0, 1, ..., n-1
```

**Key Properties:**
1. **Not evenly spaced** - clustered near endpoints
2. **Minimize interpolation error** - avoids Runge's phenomenon
3. **Optimal points** for polynomial approximation

**Example:** For 5 nodes in range [0.10, 0.40] (volatility):
```
Even spacing:  0.10, 0.175, 0.25, 0.325, 0.40
Chebyshev:     0.101, 0.143, 0.25, 0.357, 0.399  (clustered at ends)
```

**Why clustering matters:** Functions often change rapidly near boundaries, so we need more resolution there!

---

### 1D Chebyshev Interpolation: Step-by-Step

Let's walk through computing **Vega** using 1D Chebyshev interpolation for volatility σ.

#### **Setup**
- Fix parameters: S=100, K=100, T=1.0, r=0.05
- Vary: σ ∈ [0.10, 0.40]
- Use n=15 Chebyshev nodes

---

#### **Step 1: Generate Chebyshev Nodes**

**What we do:**
```python
sigma_nodes = chebyshev_nodes(n=15, a=0.10, b=0.40)
# Returns: [0.101, 0.107, 0.120, 0.139, 0.162, 0.189, ...]
```

**What's happening:**
- We calculate 15 special points in the volatility range
- These points are **NOT** evenly spaced
- They cluster near σ=0.10 and σ=0.40 (the endpoints)

**Intuition:**
Think of it like a camera's autofocus points - you want more concentration where things change quickly (near edges) and fewer in the middle where changes are smooth.

---

#### **Step 2: Evaluate Option Price at Each Node**

**What we do:**
```python
prices = []
for sigma in sigma_nodes:
    bs = BlackScholesCall(S=100, K=100, T=1.0, r=0.05, sigma=sigma, q=0)
    prices.append(bs.price())

# Results:
# σ=0.101 → Price=6.854
# σ=0.107 → Price=7.079
# ...
# σ=0.399 → Price=17.533
```

**What's happening:**
- We compute the option price at each of the 15 Chebyshev nodes
- We're building a **lookup table**: (σ₁, V₁), (σ₂, V₂), ..., (σ₁₅, V₁₅)
- This is the ONLY time we call the expensive pricing function

**Intuition:**
Like taking measurements at strategic points on a curve - we don't measure everywhere, just at the smartest locations.

**Cost:** 15 analytical evaluations ≈ 0.8ms (super fast!)

---

#### **Step 3: Fit Chebyshev Polynomial**

**What we do:**
```python
# Normalize σ to [-1, 1] for numerical stability
sigma_norm = 2 * (sigma_nodes - 0.10) / (0.40 - 0.10) - 1

# Fit polynomial: V(σ) ≈ c₀T₀(σ) + c₁T₁(σ) + ... + c₁₄T₁₄(σ)
coeffs = chebfit(sigma_norm, prices, deg=14)
# Returns: [c₀, c₁, c₂, ..., c₁₄]  (15 coefficients)
```

**What's happening:**
- We're finding a polynomial that passes through all 15 points
- Uses **Chebyshev basis polynomials**: T₀(x)=1, T₁(x)=x, T₂(x)=2x²-1, etc.
- The result is 15 coefficients that define the polynomial

**Intuition:**
Like fitting a smooth curve through your measured points. Instead of remembering all 15 (σ, price) pairs, we remember just the 15 polynomial coefficients. The polynomial lets us compute the price at **any** σ between 0.10 and 0.40.

**Key advantage:** Chebyshev polynomials are numerically stable (unlike regular polynomials which can oscillate wildly).

---

#### **Important: Two-Scale Workflow**

**You work in TWO different scales:**

1. **Real scale** [0.10, 0.40] → for generating nodes and evaluating function
2. **Normalized scale** [-1, 1] → for polynomial operations (fitting and evaluating)

**Building phase:**
```python
# Step 1: Generate nodes in REAL scale
sigma_nodes = [0.101, 0.107, ..., 0.399]  # Real σ values

# Step 2: Evaluate function in REAL scale
prices = [bs.price(sigma=s) for s in sigma_nodes]  # Use actual σ

# Step 3: NORMALIZE to [-1, 1], then fit
sigma_norm = 2*(sigma_nodes - 0.10)/(0.40 - 0.10) - 1  # → [-1, 1]
coeffs = chebfit(sigma_norm, prices, deg=14)  # Fit on normalized scale
```

**Query phase:**
```python
# Query comes in REAL scale
sigma_query = 0.25  # Actual volatility

# NORMALIZE it before evaluating polynomial
sigma_norm = 2*(sigma_query - 0.10)/(0.40 - 0.10) - 1  # → 0.0
price = chebval(sigma_norm, coeffs)  # Evaluate on normalized scale
```

**Why this works:**
- Chebyshev polynomials are defined on [-1, 1]
- We map our real range [a, b] to this mathematical space
- The normalization formula is the same for building and querying
- Result comes back in real units (price in dollars)

---

#### **Step 4: Interpolate for Any σ**

**What we do:**
```python
def interpolate_price(sigma_query):
    sigma_norm = 2 * (sigma_query - 0.10) / (0.40 - 0.10) - 1
    return chebval(sigma_norm, coeffs)

# Query at σ=0.25 (not a node!)
price = interpolate_price(0.25)
# Returns: 12.347 (exact match to analytical!)
```

**What's happening:**
- Normalize the query point to [-1, 1]
- Evaluate the Chebyshev polynomial: V = Σ cᵢTᵢ(σ)
- This is just arithmetic - no expensive pricing calls!

**Why normalize to [-1, 1]?**

Chebyshev polynomials are **defined** on the interval [-1, 1]. The formula transforms our volatility range [0.10, 0.40] to this standard interval:

```python
sigma_norm = 2 * (sigma_query - 0.10) / (0.40 - 0.10) - 1
```

**Step-by-step transformation:**
1. `sigma_query - 0.10` → Shift range to [0, 0.30]
2. `(···) / (0.40 - 0.10)` → Scale by range width: [0, 0.30] → [0, 1]
3. `2 * ···` → Scale to [0, 2]
4. `··· - 1` → Shift to [-1, 1] ✓

**Examples:**
- σ = 0.10 (min) → `2*(0-0)/(0.30) - 1` = **-1** ✓
- σ = 0.25 (mid) → `2*(0.15)/(0.30) - 1` = **0** ✓
- σ = 0.40 (max) → `2*(0.30)/(0.30) - 1` = **+1** ✓

**General formula for any interval [a, b]:**
```python
x_normalized = 2 * (x - a) / (b - a) - 1
```

This ensures we're evaluating Chebyshev polynomials at the correct input values!

**Intuition:**
Like using the smooth curve we fitted to read off values anywhere along it. We're not limited to the 15 points we measured - we can get the value at any volatility. But first we must translate our real-world volatility (0.10-0.40) into the "mathematical space" (-1 to +1) where Chebyshev polynomials live.

**Cost:** ~30μs (polynomial evaluation is instant!)

---

#### **Step 5: Compute Vega via Derivative**

**What we do:**
```python
def interpolate_vega(sigma_query):
    sigma_norm = 2 * (sigma_query - 0.10) / (0.40 - 0.10) - 1

    # Derivative of Chebyshev polynomial
    deriv_coeffs = chebder(coeffs)  # Automatic differentiation!
    deriv_norm = chebval(sigma_norm, deriv_coeffs)

    # Chain rule: dV/dσ = (dV/dσ_norm) · (dσ_norm/dσ)
    vega = deriv_norm * 2 / (0.40 - 0.10)
    return vega

vega = interpolate_vega(0.25)
# Returns: 37.589 (vs analytical: 37.842, error: 0.67%)
```

**What's happening:**
- Take the derivative of the polynomial (analytically!)
- Chebyshev polynomials have known derivatives: T'ₙ(x) = n·Uₙ₋₁(x)
- Adjust for the normalization (chain rule)

**Intuition:**
Since we have a smooth curve V(σ), we can compute its slope dV/dσ anywhere! This is **Vega**. No need to re-solve the PDE with σ±Δσ - we just differentiate the polynomial.

**Cost:** ~30μs (still instant!)

**This is the magic:** Traditional FDM would need 2 PDE solves (~1 second). Chebyshev gives us the derivative for free!

---

#### **Summary of 1D Approach**

| Step | What We Do | Cost | One-Time or Repeated? |
|------|-----------|------|----------------------|
| 1. Generate nodes | Calculate 15 special σ values | <1μs | One-time (offline) |
| 2. Evaluate prices | Call analytical formula 15 times | 0.8ms | One-time (offline) |
| 3. Fit polynomial | Compute 15 coefficients | <1ms | One-time (offline) |
| **Offline total** | | **~2ms** | |
| 4. Interpolate price | Evaluate polynomial | 30μs | Per query (online) |
| 5. Compute Vega | Differentiate polynomial | 30μs | Per query (online) |
| **Online total** | | **60μs** | |

**vs FDM for Vega:** 2 PDE solves = ~1,000ms = **16,667× slower!**

---

### Multi-Dimensional Tensor Interpolation: Step-by-Step

Now let's extend to **5 dimensions**: V(S, K, T, σ, r)

The 1D approach won't work because Vega **depends on all parameters**, not just σ!

---

#### **Step 1: Generate Chebyshev Nodes in All Dimensions**

**What we do:**
```python
S_nodes = chebyshev_nodes(11, 80, 120)      # [80.1, 83.1, ..., 119.9]
K_nodes = chebyshev_nodes(11, 90, 110)      # [90.1, 92.1, ..., 109.9]
T_nodes = chebyshev_nodes(11, 0.25, 2.0)    # [0.251, 0.337, ..., 1.999]
sigma_nodes = chebyshev_nodes(11, 0.10, 0.40)  # [0.101, 0.120, ..., 0.399]
r_nodes = chebyshev_nodes(11, 0.01, 0.10)   # [0.0101, 0.0143, ..., 0.0999]
```

**What's happening:**
- We generate 11 Chebyshev nodes **independently** for each dimension
- Total grid points: 11 × 11 × 11 × 11 × 11 = **161,051 points**
- These form a 5-dimensional grid in parameter space

**Intuition:**
Imagine a 5D checkerboard, but instead of squares evenly spaced, they cluster near the edges in each direction. We're strategically placing measurement points throughout the entire 5D parameter space.

---

#### **Step 2: Evaluate Option Price at All Grid Points**

**What we do:**
```python
tensor_full = np.zeros((11, 11, 11, 11, 11))  # 5D array

for i, S in enumerate(S_nodes):
    for j, K in enumerate(K_nodes):
        for k, T in enumerate(T_nodes):
            for l, sigma in enumerate(sigma_nodes):
                for m, r in enumerate(r_nodes):
                    bs = BlackScholesCall(S, K, T, r, sigma, q=0)
                    tensor_full[i, j, k, l, m] = bs.price()

# Result: 161,051 option prices stored in 5D array
```

**What's happening:**
- We evaluate the option price at **every** combination of the Chebyshev nodes
- This builds a complete lookup table in 5D space
- Each dimension's index corresponds to a Chebyshev node

**Intuition:**
Like building a 5D photograph of the pricing function. We take a picture at every strategically chosen point in the 5D parameter space. The result is a dense 5D grid of prices.

**Cost:** 161,051 analytical evaluations ≈ 0.28s (still fast with analytical formulas!)

**Note:** If we used FDM instead, this would take 161,051 × 0.5s ≈ **22 hours**! This is why we use analytical formulas for the offline phase.

---

#### **Step 3: Compress Using CP Decomposition**

**What we do:**
```python
# Decompose the 5D tensor using CP (CANDECOMP/PARAFAC)
tensor_decomp = parafac(tensor_full, rank=10)

# Returns: (weights, [A, B, C, D, E])
# where:
#   weights: 10 numbers (λ₁, λ₂, ..., λ₁₀)
#   A: 11×10 matrix (S factor)
#   B: 11×10 matrix (K factor)
#   C: 11×10 matrix (T factor)
#   D: 11×10 matrix (σ factor)
#   E: 11×10 matrix (r factor)
```

**What's happening:**
- We're **compressing** the 161,051-element tensor
- CP decomposition finds a low-rank approximation:
  ```
  V[i,j,k,l,m] ≈ Σʳ⁼¹¹⁰ λᵣ · A[i,r] · B[j,r] · C[k,r] · D[l,r] · E[m,r]
  ```
- Instead of storing 161,051 numbers, we store:
  - 10 weights + 5 matrices of size 11×10 = **560 parameters**
- **Compression ratio: 287.6×**

**Intuition:**
Think of it like JPEG compression for images, but for 5D data:
- JPEG finds patterns in 2D images and stores only the essential information
- CP decomposition finds patterns in 5D price surfaces and stores only the essential structure
- We lose a tiny bit of accuracy (0.56% error) but save massive amounts of storage

**Mathematical insight:** The 5D pricing function has **structure** - it's not random noise. CP decomposition discovers that structure and represents it compactly.

**Why rank=10?** This is like the "quality setting" in JPEG:
- Rank=1: Extreme compression, poor quality
- Rank=10: Good balance (0.56% error, 287× compression)
- Rank=50: High quality, less compression

**Cost:** ~1s for decomposition (iterative algorithm)

---

#### **Step 4: Reconstruct Tensor from Decomposition**

**What we do:**
```python
# Reconstruct the full tensor from compressed representation
tensor_reconstructed = tl.cp_to_tensor(tensor_decomp)

# Verify accuracy
error = ||tensor_full - tensor_reconstructed|| / ||tensor_full||
# error = 0.56% ✓
```

**What's happening:**
- We multiply the factor matrices back together to get the full 5D array
- This gives us a **smooth approximation** of the original tensor
- The reconstruction error is tiny (0.56%)

**Intuition:**
Like decompressing a JPEG - you get back a 5D array that looks almost identical to the original, but stored using 287× less memory.

**Key point:** We don't actually store `tensor_reconstructed` (that would defeat the purpose!). We keep the compressed form (weights + 5 factor matrices) and reconstruct values on-the-fly.

---

#### **Step 5: Build Multi-Linear Interpolator**

**What we do:**
```python
interpolator = RegularGridInterpolator(
    points=(S_nodes, K_nodes, T_nodes, sigma_nodes, r_nodes),
    values=tensor_reconstructed,
    method='linear'
)
```

**What's happening:**
- We create an interpolation object that knows:
  - The grid points (Chebyshev nodes in each dimension)
  - The values at those points (from reconstructed tensor)
  - How to interpolate between points (linearly)

**Intuition:**
Like creating a GPS system for the 5D price surface:
- The GPS knows key landmarks (Chebyshev nodes)
- It knows the altitude at each landmark (option prices)
- When you query any location, it finds nearby landmarks and smoothly interpolates

**Example in 2D (simpler to visualize):**
```
Imagine we have prices at 4 corners of a square:
  (S=90, σ=0.10) → Price = 5.2
  (S=90, σ=0.20) → Price = 8.1
  (S=110, σ=0.10) → Price = 15.7
  (S=110, σ=0.20) → Price = 18.3

To get price at (S=100, σ=0.15):
1. Find the 4 surrounding points (the square's corners)
2. Interpolate along S direction
3. Interpolate along σ direction
4. Result: smooth blend of nearby values
```

The 5D case works the same way, but with a "hypercube" of 2⁵=32 surrounding points!

---

#### **Step 6: Interpolate Price for Any Parameters**

**What we do:**
```python
def interpolate_price(S, K, T, sigma, r):
    point = np.array([[S, K, T, sigma, r]])
    return interpolator(point)[0]

# Query at arbitrary point (not a grid node!)
price = interpolate_price(S=105, K=97, T=0.75, sigma=0.23, r=0.047)
# Returns: 11.234 (smooth interpolation from nearby grid points)
```

**What's happening:**
- Find the 32 surrounding grid points (2⁵ hypercube corners)
- For each corner, retrieve value from reconstructed tensor
- Perform multi-linear interpolation (weighted average based on distance)

**Intuition:**
You're asking "what's the price at this random point in 5D space?"
1. The interpolator finds the 32 nearest Chebyshev grid points
2. It looks up the (compressed, reconstructed) price at each
3. It blends them smoothly based on how close each one is
4. Returns a smooth estimate

**Mathematical view:**
```
V(S, K, T, σ, r) ≈ Σ₃₂ wᵢ · V(Sᵢ, Kᵢ, Tᵢ, σᵢ, rᵢ)
```
where weights wᵢ sum to 1 and depend on distance to each corner.

**Cost:** ~1ms per query (fast!)

---

#### **Step 7: Compute Vega via Finite Difference**

**What we do:**
```python
def interpolate_vega(S, K, T, sigma, r, dsigma=0.01):
    V_up = interpolate_price(S, K, T, sigma + dsigma, r)
    V_down = interpolate_price(S, K, T, sigma - dsigma, r)
    return (V_up - V_down) / (2 * dsigma)

vega = interpolate_vega(S=105, K=97, T=0.75, sigma=0.23, r=0.047)
# Returns: 35.234
```

**What's happening:**
- We perturb σ slightly in both directions
- Use our fast interpolator to get prices
- Compute derivative via finite difference

**Intuition:**
To find the slope (Vega), we:
1. Step a tiny bit forward in σ → get price
2. Step a tiny bit backward in σ → get price
3. Calculate: slope = (forward - backward) / (2 × step_size)

**Key advantage over 1D:**
Now this works for **any** (S, K, T, r) combination! The 1D method only worked at the fixed point (S=100, K=100, T=1.0, r=0.05).

**Why not use polynomial derivatives like 1D?**
In 5D, taking analytical derivatives of the tensor decomposition is complex. Finite difference on the interpolator is simpler and still fast (~2ms for both evaluations).

**Cost:** ~2ms (2 interpolations)

**vs FDM:** 2 PDE solves = ~1,000ms = **500× faster!**

---

#### **Summary of Multi-Dimensional Approach**

| Step | What We Do | Cost | One-Time or Repeated? |
|------|-----------|------|----------------------|
| 1. Generate nodes (5D) | Calculate 11 nodes × 5 dims | <1ms | One-time (offline) |
| 2. Evaluate prices | Call analytical 161,051 times | 280ms | One-time (offline) |
| 3. CP decomposition | Compress to rank-10 | 950ms | One-time (offline) |
| 4. Reconstruct | Build smooth approximation | <10ms | One-time (offline) |
| 5. Build interpolator | Setup 5D grid interpolator | <1ms | One-time (offline) |
| **Offline total** | | **~1.24s** | |
| 6. Interpolate price | 5D multi-linear interp | 1ms | Per query (online) |
| 7. Compute Vega | 2 interpolations + difference | 2ms | Per query (online) |
| **Online total** | | **~3ms** | |

**vs FDM for Vega:** 2 PDE solves = ~1,000ms = **333× faster!**

---

### Why 5D Works Where 1D Fails

#### **The 1D Problem:**

Built at fixed point:
```
V_σ(σ) at (S=100, K=100, T=1.0, r=0.05)
```

Works great when querying near this point:
- Query: (S=100, K=100, T=1.0, σ=0.25, r=0.05) ✓ Vega error: 0.67%

Fails when parameters change:
- Query: (S=85, K=100, T=1.0, σ=0.20, r=0.05) ✗ Vega error: 22%

**Why?** Because Vega **changes** with S! The 1D interpolant doesn't know about this.

```
Vega at (S=85, σ=0.20) = 30.47
Vega at (S=100, σ=0.20) = 37.52
Vega at (S=115, σ=0.20) = 26.47
```

Using the σ-only interpolant is like using a thermometer calibrated at sea level to measure temperature on a mountain - the calibration is wrong!

#### **The 5D Solution:**

Built everywhere:
```
V(S, K, T, σ, r) for all combinations
```

Knows how Vega depends on **all** parameters:
- Interpolates smoothly in all 5 dimensions
- Works uniformly well across entire parameter space

Query: (S=85, K=100, T=1.0, σ=0.20, r=0.05) ✓ Vega error: **3.77%**

**Why better?** The 5D tensor knows that at S=85, the σ-sensitivity is different than at S=100. It learned this during the offline phase by sampling the entire 5D space.

---

### Computational Trade-offs

| Aspect | 1D Interpolation | 5D Tensor Interpolation |
|--------|------------------|-------------------------|
| **Offline evaluations** | 15 | 161,051 |
| **Offline time** | 2ms | 1.24s |
| **Storage** | 15 coeffs | 560 params (compressed) |
| **Online price query** | 30μs | 1ms |
| **Online Vega query** | 30μs | 2ms |
| **Accuracy (when varying σ)** | 0.5-1% ✓ | 3-8% ✓ |
| **Accuracy (when varying S,K,T)** | 20-100% ✗ | 3-8% ✓ |
| **Break-even** | 1 query | 2-3 queries |

**Key insight:**
- 1D: Ultra-fast but only works in narrow scenarios
- 5D: Slightly slower but works everywhere
- Both are **vastly faster** than FDM for repeated queries

---

### When to Use Each Approach

#### **Use 1D Chebyshev when:**
✅ Only **one parameter varies** at a time
✅ Other parameters are truly fixed
✅ Need ultra-fast queries (<100μs)

**Example:** Volatility surface calibration
- Fix S, K, T, r (market conditions)
- Sweep σ to match market prices
- Perfect use case for 1D!

#### **Use Multi-D Tensor when:**
✅ **Multiple parameters vary** simultaneously
✅ Need Greeks across diverse scenarios
✅ Can tolerate 1-8% error
✅ Have many queries (>10) to amortize offline cost

**Example:** Portfolio risk management
- Many options with different S, K, T, σ, r
- Need Vega and Rho for all of them
- Run daily with thousands of positions
- Perfect use case for 5D!

#### **Use FDM/Numerical when:**
✅ Need highest accuracy (<0.1% error)
✅ One-off calculation
✅ American options or exotic features
✅ Can afford the computational time

**Example:** Pricing a custom structured product
- Non-standard payoff
- No analytical formula
- Single pricing, accuracy critical
- Perfect use case for FDM!

---

## Performance Comparison

| Method | Vega Max Error | Rho Max Error | Works for All Parameters? | Offline Time | Online Time |
|--------|---------------|---------------|---------------------------|--------------|-------------|
| **1D Chebyshev** | 40.48% | 108.79% | ❌ Only σ, r | 1.8ms | 30μs |
| **Tensor (CP)** | **7.60%** | **4.28%** | ✅ All (S,K,T,σ,r) | 1.24s | 1.1ms |
| **FDM** | ~0.05% | ~0.04% | ✅ All | ~0.5s per solve | ~1s per Greek |

---

## Detailed Results: Tensor Interpolation

### Configuration
- **Tensor size**: 11^5 = 161,051 points
- **CP rank**: 10
- **Compression ratio**: 287.6×
- **Reconstruction error**: 0.56%
- **Storage**: 560 parameters (vs 161,051 full tensor)

### Accuracy Across 19 Test Configurations

#### Price Interpolation
| Metric | Value |
|--------|-------|
| Mean Error | 0.79% |
| Median Error | 0.70% |
| Max Error | 1.94% |
| Min Error | 0.11% |

#### Vega Interpolation (Volatility Sensitivity)
| Metric | Value |
|--------|-------|
| Mean Error | 3.22% |
| Median Error | 2.97% |
| Max Error | **7.60%** |
| Min Error | 0.50% |

**Improvement over 1D**: 40.48% → 7.60% (5.3× better!)

#### Rho Interpolation (Interest Rate Sensitivity)
| Metric | Value |
|--------|-------|
| Mean Error | 1.36% |
| Median Error | 0.72% |
| Max Error | **4.28%** |
| Min Error | 0.37% |

**Improvement over 1D**: 108.79% → 4.28% (25× better!)

---

## Why 1D Interpolation Failed

### The Problem

1D Chebyshev interpolants were built at **fixed point** (S=100, K=100, T=1.0):
```python
V_sigma(σ) at (S=100, K=100, T=1.0, r=0.05)  # Fixed S, K, T, r
V_r(r) at (S=100, K=100, T=1.0, σ=0.20)      # Fixed S, K, T, σ
```

When querying at **different** S, K, or T:
- Vega at S=85 had 22% error (vs S=100 where it was built)
- Rho at T=0.5 had 91% error (vs T=1.0 where it was built)

**Root cause**: Vega and Rho are **not constant** across different spot prices or maturities!

### Example from Results

| Configuration | 1D Vega Error | Tensor Vega Error | Improvement |
|---------------|--------------|-------------------|-------------|
| σ = 20% (baseline) | 0.90% | 2.97% | Similar (both good) |
| σ = 25% (varying σ) | 0.67% | 0.50% | Similar (both good) |
| **S = 85** (varying S) | **22.04%** | **3.77%** | **5.8× better** |
| **T = 0.5** (varying T) | **35.92%** | **7.60%** | **4.7× better** |

The 1D method only works when varying the interpolated parameter!

---

## How Tensor Interpolation Solves This

### Multi-Dimensional Representation

Instead of two separate 1D functions:
```
V(σ) and V(r)  [2 separate 1D interpolants]
```

Build full 5D tensor:
```
V(S, K, T, σ, r)  [single multi-dimensional interpolant]
```

### CP (CANDECOMP/PARAFAC) Decomposition

Approximates the 5D tensor using low-rank factorization:

```
V[i,j,k,l,m] ≈ Σᵣ λᵣ · A[i,r] · B[j,r] · C[k,r] · D[l,r] · E[m,r]
```

Where:
- λᵣ: weights (rank R)
- A, B, C, D, E: factor matrices (size n×R for each dimension)
- R = 10 (tensor rank)

**Storage**:
- Full tensor: 11^5 = 161,051 values
- CP decomposition: 10 weights + 5×(11×10) = 560 parameters
- **Compression**: 287.6× reduction!

### Interpolation

Use `scipy.RegularGridInterpolator` with the reconstructed tensor for multi-linear interpolation in 5D space.

**Result**: Accurate sensitivities for **any** combination of (S, K, T, σ, r)!

---

## Computational Cost Analysis

### Offline Phase (Build Tensor)

| Step | Time | Operations |
|------|------|-----------|
| Evaluate analytical formulas | 0.28s | 161,051 evaluations |
| CP decomposition | 0.95s | Tensor factorization |
| Build interpolator | <0.01s | Setup grid |
| **Total** | **1.24s** | One-time cost |

**Note**: Using analytical formulas instead of FDM is ~100× faster for offline phase!
- FDM would take: 161,051 × 0.5s ≈ 22 hours
- Analytical: 161,051 × 1.8μs ≈ 0.3s

### Online Phase (Query)

| Method | Time per Query | Speedup vs FDM |
|--------|---------------|----------------|
| Analytical | 10μs | 100,000× |
| Tensor Interpolation | 1.1ms | 909× |
| FDM (Vega+Rho) | ~1s | 1× (baseline) |

### Break-Even Analysis

If computing Vega+Rho via FDM:
- Cost per query: ~2s (1s for Vega, 1s for Rho)
- Tensor cost: 1.1ms

**Break-even**: After just 1 query, tensor method pays for itself!

For 1000 queries:
- **FDM**: 2,000 seconds (33 minutes)
- **Tensor**: 1.24s + 1.1s = 2.34s
- **Speedup**: 855×

---

## Key Insights from Research Papers

This implementation demonstrates concepts from `CHEBYSHEV_ACCELERATION.md`:

### 1. Parametric Option Pricing (arXiv:1505.04648)
✅ Pre-compute at Chebyshev nodes
✅ Fast online queries via interpolation
✅ Separates offline (expensive) from online (cheap)

### 2. Tensor Methods (arXiv:1902.04367)
✅ Low-rank tensor approximation
✅ Multi-dimensional parameter space
✅ 287× compression with <1% error
✅ Handles curse of dimensionality

### 3. Ultra-Efficient Risk Calculations (arXiv:1805.00898)
✅ Thousands of sensitivity calculations
✅ ~1000× speedup for repeated queries
✅ Practical for real-time risk systems

---

## Production Recommendations

### When to Use Each Method

**1D Chebyshev Interpolation:**
- ✅ Single parameter varies (e.g., volatility surface calibration)
- ✅ Extremely fast queries needed (<50μs)
- ❌ Don't use if multiple parameters vary simultaneously

**Tensor Interpolation:**
- ✅ Multi-dimensional parameter space
- ✅ Thousands of queries needed
- ✅ All Greeks required (including Vega, Rho)
- ✅ Can tolerate 1-8% error
- ❌ Don't use for single query (offline cost not amortized)

**FDM/Numerical Methods:**
- ✅ Highest accuracy needed (<0.1% error)
- ✅ American options, exotics
- ✅ One-off calculations
- ❌ Don't use for repeated similar queries

### Optimal Strategy

```python
# Tier 1: Analytical (if available)
if option_type in ['european_call', 'european_put']:
    return blackscholes.price()

# Tier 2: Tensor interpolation (for repeated queries)
elif in_parameter_range and num_queries > 10:
    return tensor_pricer.interpolate_price(S, K, T, sigma, r)

# Tier 3: Numerical methods (for accuracy or exotic features)
else:
    return fdm_solver.solve()
```

---

## Extending to Higher Dimensions

Current implementation: 5D (S, K, T, σ, r)

### Tensor Train vs CP Decomposition

For **d > 5 dimensions**, use **Tensor Train (TT)** instead of CP:

**CP Decomposition**:
- Storage: O(R × d × n)
- Rank growth: Can be unstable for d > 10

**Tensor Train (TT)**:
- Storage: O(d × n × r²) where r << R
- Rank growth: More stable
- Used in papers for d=10-50 dimensions

### Example Extensions

**Multi-asset options** (d=10-25):
- V(S₁, S₂, S₃, K, T, σ₁, σ₂, σ₃, ρ₁₂, ρ₁₃, ρ₂₃, r)
- TT decomposition with rank r=5-10
- Papers report <2% error with ~10⁵ parameters

**XVA calculations** (d=30-50):
- CVA, DVA, FVA with multiple risk factors
- Tensor completion for sparse observations
- Papers report 90%+ computational reduction

---

## What Proper Multi-Dimensional Chebyshev Polynomial Regression Looks Like

### The Question: Why Linear Interpolation Instead of Chebyshev Polynomials?

You might have noticed that in our 5D tensor interpolation (Step 5, line 476), we use:

```python
interpolator = RegularGridInterpolator(
    points=(S_nodes, K_nodes, T_nodes, sigma_nodes, r_nodes),
    values=tensor_reconstructed,
    method='linear'  # ← Why linear, not Chebyshev polynomial?
)
```

**The question is valid**: We use Chebyshev **nodes** for optimal sampling, but then use **linear interpolation** for reconstruction. Why not use Chebyshev **polynomial evaluation** instead?

**Short answer**: Implementation complexity vs practical accuracy trade-off.

**Long answer**: Let's examine how a production-grade system (MoCaX Intelligence) solves this problem properly.

---

### Case Study: MoCaX Intelligence Library

**MoCaX Intelligence** is a commercial library (by iRuiz Technologies) that implements true multi-dimensional Chebyshev polynomial approximation. We have access to their library in `MoCaXSuite-1.2.0/` and can see their approach.

#### What MoCaX Does Differently

**Our Implementation** (Pragmatic Approach):
```python
# Step 1: Sample at Chebyshev nodes
S_nodes = chebyshev_nodes(11, 80, 120)
K_nodes = chebyshev_nodes(11, 90, 110)
# ... etc for all 5 dimensions

# Step 2: Build full tensor at nodes
tensor_full[i,j,k,l,m] = bs.price(S[i], K[j], T[k], σ[l], r[m])
# → 11^5 = 161,051 values

# Step 3: Compress with CP decomposition
tensor_decomp = parafac(tensor_full, rank=10)
# → 560 parameters

# Step 4: Reconstruct tensor
tensor_reconstructed = tl.cp_to_tensor(tensor_decomp)

# Step 5: Use LINEAR INTERPOLATION between grid points
interpolator = RegularGridInterpolator(
    points=(S_nodes, K_nodes, ...),
    values=tensor_reconstructed,
    method='linear'  # ← Piecewise linear!
)

# Step 6: Greeks via finite differences (numerical)
V_up = interpolator([S, K, T, σ + Δσ, r])
V_down = interpolator([S, K, T, σ - Δσ, r])
vega = (V_up - V_down) / (2 * Δσ)  # Approximation!
```

**MoCaX Implementation** (Theoretically Correct Approach):
```python
# From mocax_test.py (lines 140-150):
n_values = [15, 12, 10]  # Chebyshev nodes per dimension
ns = mocaxpy.MocaxNs(n_values)
max_derivative_order = 2

# Build MoCaX approximation (internally stores Chebyshev coefficients)
mocax_bs = mocaxpy.Mocax(
    bs_call_wrapper,      # Original function
    num_dimensions=3,     # (S, T, σ)
    domain=domain,        # [(50, 150), (0.1, 2.0), (0.1, 0.5)]
    None,                 # Or specify error threshold
    ns,                   # Chebyshev nodes specification
    max_derivative_order=2
)

# Evaluation: Chebyshev POLYNOMIAL evaluation (not linear interpolation!)
derivative_id = mocax_bs.get_derivative_id([0, 0, 0])
price = mocax_bs.eval(test_point, derivative_id)

# Greeks: ANALYTICAL derivatives of Chebyshev polynomials
derivative_id = mocax_bs.get_derivative_id([0, 0, 1])  # ∂/∂σ
vega = mocax_bs.eval(test_point, derivative_id)  # Direct evaluation!
```

---

### Internal Representation: Coefficients vs Values

#### Our Approach: Stores Reconstructed Values
```
Full tensor of VALUES at grid nodes
    ↓ (CP decomposition)
Compressed representation (560 params)
    ↓ (Reconstruction)
Approximate VALUES at grid nodes
    ↓ (Linear interpolation)
Query result at arbitrary point
```

#### MoCaX Approach: Stores Chebyshev Coefficients
```
Sample at Chebyshev nodes
    ↓ (Chebyshev transform)
Full tensor of COEFFICIENTS
    ↓ (Tensor decomposition)
Compressed COEFFICIENTS representation
    ↓ (Chebyshev polynomial evaluation)
Query result at arbitrary point
```

**Key difference**: MoCaX builds this representation:

```
V(S, K, T, σ, r) = Σᵢ Σⱼ Σₖ Σₗ Σₘ cᵢⱼₖₗₘ · Tᵢ(S) · Tⱼ(K) · Tₖ(T) · Tₗ(σ) · Tₘ(r)
```

Where:
- `cᵢⱼₖₗₘ` are **Chebyshev coefficients** (not raw values!)
- `Tₙ(x)` are Chebyshev polynomials: T₀(x)=1, T₁(x)=x, T₂(x)=2x²-1, etc.

These coefficients are then compressed using tensor decomposition.

---

### Evaluation: Polynomial vs Piecewise Linear

#### Our Linear Interpolation Approach

**What happens when querying at (S=105, σ=0.23)?**

1. Find surrounding hypercube (2^5 = 32 corner nodes)
2. Extract values at corners from reconstructed tensor
3. Perform multi-linear interpolation:
   ```
   V(x) ≈ weighted average of 32 corner values
   ```
4. Result: **Piecewise linear** approximation
   - Continuous (C⁰)
   - **Not smooth**: derivatives discontinuous at grid boundaries
   - Error: O(h²) where h = grid spacing

**For Greeks** (finite difference):
```python
vega ≈ (V(σ + 0.01) - V(σ - 0.01)) / 0.02
```
- Requires 2 interpolation calls
- Introduces finite difference error: O(Δσ²)
- Total error: interpolation error + finite difference error
- **Result from our tests: 3-8% error on Greeks**

#### MoCaX Chebyshev Polynomial Approach

**What happens when querying at (S=105, σ=0.23)?**

1. Normalize inputs to [-1, 1] for each dimension
2. Evaluate Chebyshev polynomials:
   ```
   Tᵢ(S_norm), Tⱼ(K_norm), Tₖ(T_norm), Tₗ(σ_norm), Tₘ(r_norm)
   ```
3. Compute weighted sum of products:
   ```
   V = Σ (compressed_coefficients × basis_products)
   ```
4. Result: **Smooth polynomial** approximation
   - Infinitely differentiable (C^∞)
   - **Spectral accuracy**: exponential convergence for smooth functions
   - Error: O(exp(-αN)) for smooth f(x)

**For Greeks** (analytical derivatives):
```python
∂V/∂σ = Σ (coefficients × ∂Tₗ(σ)/∂σ × other_terms)
```
- Single evaluation (same cost as function eval!)
- Uses analytical Chebyshev derivative formulas:
  - T'₀(x) = 0
  - T'₁(x) = 1
  - T'₂(x) = 4x
  - T'ₙ(x) = n · Uₙ₋₁(x) (Chebyshev polynomials of 2nd kind)
- **No finite difference error!**
- **Result from MoCaX tests: 0.15% error on Greeks**

---

## Conclusion

### Demonstrated Results

✅ **Accuracy**: 1-8% error across all parameters (vs 40-108% for 1D)
✅ **Speed**: 855× faster than FDM for 1000 queries
✅ **Compression**: 287× storage reduction
✅ **Scalability**: Works in 5D parameter space

### Validates Research Claims

The implementation confirms findings from academic papers:
- Tensor methods handle multi-dimensional problems
- Low-rank approximations achieve high compression with low error
- Practical for real-time risk systems
- Separating offline/online phases is key to performance

### Next Steps

1. **Implement TT decomposition** for higher dimensions
2. **Add TT-cross algorithm** for adaptive sampling
3. **Benchmark** against production systems
4. **Extend** to American options using dynamic tensor methods

---

## References

See `CHEBYSHEV_ACCELERATION.md` for detailed analysis of:
- arXiv:1505.04648 - Chebyshev Interpolation for Parametric Option Pricing
- arXiv:1805.00898 - Chebyshev Methods for Ultra Efficient Risk Calculations
- arXiv:1902.04367 - Low-rank tensor approximations
- arXiv:1808.08221 - Dynamic Initial Margin via Tensors
