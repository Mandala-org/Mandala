# Performance Optimization Plan for Minimal E(3)-GNN Study

**Date:** January 27, 2026
**Target:** `studies/minimal_overfit_study/` codebase
**Goal:** Identify and implement performance optimizations focusing on torch.jit/torchscript and computational graph improvements

---

## Executive Summary

The current implementation has **significant performance bottlenecks** due to:
1. **Python overhead in hot loops** (data structure construction)
2. **Repeated irrep-to-block conversions** during training
3. **Verbose print statements** in forward pass
4. **No torch.jit compilation** for core modules
5. **Inefficient scatter operations** without optimization
6. **WandB logging overhead** during training

**Estimated speedup potential:** **3-10x** with comprehensive optimizations.

---

## Current Performance Bottlenecks (Ranked by Impact)

### 🔴 Critical (High Impact)

#### 1. **IrrepsBlockData Construction Overhead** (Training Loop)
**Location:** `overfit_water_minimal.py` lines 480-503

```python
for key, payload in pred_raw.items():
    pair_vec_H[key] = payload["vectors"]
    pair_edges_dict[key] = payload["edges"]

    for idx, edge_5d in enumerate(payload["edges"].t()):  # ← Python loop!
        sx, sy, sz, i, j = edge_5d.tolist()  # ← GPU → CPU transfer!
        lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)
```

**Problem:**
- Python loop iterating over edges (30-100+ edges per snapshot)
- `.tolist()` triggers GPU→CPU data transfer **every edge, every forward pass**
- Dictionary construction in pure Python

**Impact:** ~40-60% of per-epoch time
**Solution:** Pre-compute lookup dict once, or move to vectorized operations

---

#### 2. **Print Statements in Forward Pass**
**Location:** Throughout `common.py` (MinimalNetwork, all modules)

```python
def forward(self, ...):
    print("    [Forward] Starting forward pass...")  # ← I/O blocking!
    print(f"      [EdgeEncoder.forward] Radial: {edge_length_emb.shape}...")
```

**Problem:**
- Print I/O blocks GPU computation
- String formatting overhead
- Prevents torch.jit compilation

**Impact:** ~20-30% slowdown when enabled
**Solution:** Replace with conditional logging or remove for production

---

#### 3. **Lack of torch.jit Compilation**
**Location:** All network modules in `common.py`

**Problem:**
- No modules use `@torch.jit.script` or `.jit.trace()`
- Python interpreter overhead for every operation
- No operator fusion (e.g., `matmul + bias + activation` → single kernel)

**Impact:** 2-3x slowdown vs. JIT compiled code
**Compatibility Issues:**
- `e3nn` modules (TP, Gate) have **limited JIT support**
- Dynamic control flow (if statements) complicates scripting
- String-based irreps require workarounds

---

#### 4. **Irrep-to-Block Conversion on Every Iteration**
**Location:** Training loop after forward pass

```python
pred_H_matrix = pred_H_irreps.to_blocks(mapper)  # ← Expensive matmul @ Q^T
```

**Problem:**
- `blocks_to_vectors` / `vectors_to_blocks` involves matrix multiply with change-of-basis `Q`
- Done **every forward pass** even though Q is constant
- Not fused with downstream operations

**Impact:** ~10-15% of forward pass time
**Solution:**
- Compute loss directly on irrep vectors (avoid conversion)
- Cache Q^T to avoid repeated transposes
- Fuse loss computation with conversion

---

### 🟡 Moderate (Medium Impact)

#### 5. **e3LayerNorm Scatter Operations**
**Location:** `common.py` lines 57-95

```python
mean = scatter(field, batch, dim=0, dim_size=batch_size, reduce="add").mean(...)
norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size, reduce="mean").mean(...)
```

**Problem:**
- Two separate scatter operations
- Not using CUDA-optimized paths for single-graph batches
- Intermediate `.mean()` allocations

**Impact:** ~5-10% of layer norm time
**Solution:** Fuse operations, use specialized single-batch path

---

#### 6. **WandB Logging in Training Loop**
**Location:** `overfit_water_minimal.py` lines 522, 559-572

```python
wandb.log({"train/loss_step": loss.item(), "epoch": epoch})  # ← Every step!
wandb.log({...7 metrics...})  # ← Every log_interval
```

**Problem:**
- `.item()` forces GPU synchronization
- Network I/O overhead
- Blocks GPU work queue

**Impact:** ~2-5% per step, ~10-15% during log intervals
**Solution:** Batch logs, async logging, reduce frequency

---

#### 7. **Radial Basis Function Computation**
**Location:** `overfit_water_minimal.py` lines 304-310

```python
edge_length_emb = soft_one_hot_linspace(
    edge_dist,
    start=0.0,
    end=CONFIG["cutoff_radius"],
    number=CONFIG["n_radial"],
    basis="gaussian",
    cutoff=False,
)
```

**Problem:**
- Computed every epoch even though edges are static for single-molecule study
- `soft_one_hot_linspace` not JIT-compiled

**Impact:** ~5% overhead (only for static datasets)
**Solution:** Pre-compute and cache for single-snapshot overfitting

---

### 🟢 Minor (Low Impact)

#### 8. **F.one_hot in Edge Encoder**
**Location:** `common.py` line 199

```python
edge_type_onehot = F.one_hot(edge_type_idx, num_classes=self.num_edge_types).float()
```

**Impact:** <2%
**Solution:** Pre-compute for static graphs

---

#### 9. **Activation Magnitude Logging**
**Location:** `common.py` `log_activation_magnitudes` function

**Impact:** ~2-3% when enabled
**Solution:** Make truly optional with no-op when disabled

---

## Optimization Strategies

### Phase 1: Low-Hanging Fruit (Easy Wins)
**Estimated speedup: 2-3x, Implementation time: 2-4 hours**

1. **Remove/Disable Print Statements**
   - Add `verbose` flag (default `False`)
   - Use `logging` module with levels
   - Only print during debug mode

2. **Pre-compute Static Graph Features**
   - Cache `edge_length_emb`, `edge_sh`, `edge_type_onehot`
   - Store in dataset or as model buffers
   - Only for single-snapshot overfitting

3. **Optimize IrrepsBlockData Construction**
   - Pre-build lookup dict once before training
   - Use tensor operations instead of Python loops
   - Avoid `.tolist()` calls

4. **Batch WandB Logging**
   - Accumulate metrics in lists
   - Log in batches every N steps
   - Use async logging if available

---

### Phase 2: torch.jit Selective Compilation (Moderate Complexity)
**Estimated speedup: 1.5-2x additional, Implementation time: 1-2 days**

**Target modules for JIT compilation:**

#### ✅ **Good JIT Candidates:**
- `MinimalNodeEncoder` (simple embedding lookup + linear)
- `MinimalHead.projections` (linear layers only)
- Radial projection (`self.radial_proj`)
- Loss computation functions

#### ⚠️ **Partial JIT Candidates (with modifications):**
- `e3LayerNorm` (requires rewriting to avoid dynamic irreps iteration)
- Message update aggregations (if e3nn TPs are frozen)

#### ❌ **Poor JIT Candidates:**
- Full `MinimalNetwork` (too many e3nn components)
- `Gate` module (e3nn internal complexity)
- `FullyConnectedTensorProduct` (dynamic path selection)

**Implementation approach:**
```python
@torch.jit.script
class FastNodeEncoder(nn.Module):
    def __init__(self, num_elements: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, out_dim)

    def forward(self, node_type_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(node_type_idx)
```

**JIT-Compatible Loss Function:**
```python
@torch.jit.script
def compute_block_mse(pred_blocks: Dict[str, torch.Tensor],
                      targ_blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=pred_blocks['H-H'].device)
    for key in targ_blocks.keys():
        if key in pred_blocks:
            pred = pred_blocks[key]
            targ = targ_blocks[key]
            min_n = min(pred.size(0), targ.size(0))
            loss = loss + F.mse_loss(pred[:min_n], targ[:min_n])
    return loss
```

---

### Phase 3: Advanced Optimizations (High Complexity)
**Estimated speedup: 1.3-1.5x additional, Implementation time: 3-5 days**

#### 1. **Fused Kernels for e3LayerNorm**
Replace scatter operations with custom CUDA kernel:

```python
# Pseudo-code for fused kernel
@torch.jit.script
def fused_layer_norm_single_batch(x: torch.Tensor,
                                    irreps_slices: List[Tuple[int, int]],
                                    weight: torch.Tensor,
                                    bias: torch.Tensor,
                                    eps: float) -> torch.Tensor:
    # Single graph batch optimization
    out_chunks = []
    for start, end in irreps_slices:
        field = x[:, start:end]
        # Fused mean + variance + normalize + affine
        normalized = torch.nn.functional.layer_norm(
            field, (field.size(-1),), None, None, eps
        )
        out_chunks.append(normalized * weight[...] + bias[...])
    return torch.cat(out_chunks, dim=1)
```

#### 2. **Compute Loss on Irrep Vectors Directly**
Avoid irrep→block conversion:

```python
# Define loss in irrep space
def irrep_space_loss(pred_vecs: Dict[str, torch.Tensor],
                      targ_vecs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Loss computed on irrep vectors, no block conversion needed."""
    loss = 0.0
    for key in targ_vecs.keys():
        if key in pred_vecs:
            loss += F.mse_loss(pred_vecs[key], targ_vecs[key])
    return loss
```

**Advantage:**
- Skip Q matrix multiplication
- ~10-15% faster
- Mathematically equivalent (Q is orthogonal)

#### 3. **Mixed Precision Training**
```python
# Add to training script
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    with autocast():  # FP16 forward pass
        pred = network(...)
        loss = compute_loss(pred, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 1.5-2x faster on modern GPUs (Ampere, Ada)
- 50% memory reduction
- Minimal accuracy loss for large models

#### 4. **Operator Fusion via torch.compile (PyTorch 2.0+)**
```python
# Experimental: PyTorch 2.0 compile
network = torch.compile(network, mode="reduce-overhead")
```

**Pros:** Automatic kernel fusion, graph optimization
**Cons:** Limited e3nn compatibility, may not work out-of-box

---

## Recommended Implementation Order

### Week 1: Quick Wins ✅
1. **Day 1:** Remove print statements, add verbose flag
2. **Day 2:** Pre-compute static features (edges, radials, SH)
3. **Day 3:** Optimize IrrepsBlockData construction (vectorize lookup)
4. **Day 4:** Batch WandB logging, reduce synchronization
5. **Day 5:** Benchmark and validate (expect 2-3x speedup)

### Week 2: Selective JIT 🔧
1. **Days 6-7:** JIT-compile NodeEncoder, Head, loss functions
2. **Days 8-9:** Rewrite e3LayerNorm for single-batch fast path
3. **Day 10:** Profile and fix JIT errors

### Week 3: Advanced (Optional) 🚀
1. **Days 11-13:** Implement irrep-space loss computation
2. **Days 14-15:** Mixed precision training
3. **Day 16+:** Explore torch.compile for compatible modules

---

## Profiling Strategy

### Tools to Use:
1. **PyTorch Profiler:**
   ```python
   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       for epoch in range(10):
           network(...)

   prof.export_chrome_trace("trace.json")  # View in chrome://tracing
   ```

2. **Line Profiler:**
   ```bash
   pip install line_profiler
   kernprof -l -v overfit_water_minimal.py
   ```

3. **NVIDIA Nsight Systems:**
   ```bash
   nsys profile -o profile.qdrep python overfit_water_minimal.py
   ```

### Key Metrics to Track:
- **Forward pass time** (target: <10ms for single H2O)
- **Backward pass time** (target: <20ms)
- **Python overhead** (target: <5% of total time)
- **GPU utilization** (target: >80%)
- **Memory bandwidth utilization** (target: >60%)

---

## E3NN-Specific Considerations

### torch.jit Compatibility Issues:
1. **Irreps as strings:** e3nn uses string parsing → not JIT friendly
   - **Solution:** Pre-build irreps, pass as module attributes

2. **FullyConnectedTensorProduct dynamic paths:**
   - Different code paths for different irreps
   - **Solution:** Use `@torch.jit.ignore` for TP modules, only JIT wrapping layers

3. **Gate module internals:**
   - Complex activation routing
   - **Solution:** Keep Gate as eager mode, JIT pre/post layers

### Equivariance-Preserving Optimizations:
- ✅ **Safe:** JIT compilation, operator fusion, mixed precision
- ✅ **Safe:** Pre-computing static features, caching
- ⚠️ **Risky:** Custom CUDA kernels (must maintain equivariance)
- ❌ **Unsafe:** Approximate spherical harmonics, pruning irreps arbitrarily

---

## Expected Performance Gains (H2O Single Molecule)

| Optimization | Speedup | Cumulative | Effort |
|--------------|---------|-----------|--------|
| Baseline     | 1.0x    | 1.0x      | -      |
| Remove prints| 1.3x    | 1.3x      | 30 min |
| Pre-compute features | 1.5x | 2.0x | 2 hrs |
| Optimize IrrepsBlockData | 1.5x | 3.0x | 4 hrs |
| Batch WandB logging | 1.1x | 3.3x | 1 hr |
| JIT node encoder + head | 1.3x | 4.3x | 1 day |
| JIT loss functions | 1.1x | 4.7x | 4 hrs |
| Irrep-space loss | 1.2x | 5.6x | 2 days |
| Mixed precision | 1.5x | 8.4x | 1 day |
| torch.compile (if compatible) | 1.2x | 10x | 1 day |

---

## Code Examples

### 1. Verbose Flag Implementation

**File:** `common.py`
```python
class MinimalNetwork(nn.Module):
    def __init__(self, ..., verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        ...

    def forward(self, ...):
        if self.verbose:
            print("    [Forward] Starting forward pass...")
        ...
```

---

### 2. Pre-computed Features

**File:** `overfit_water_minimal.py`
```python
# Before training loop
print("\n[OPTIMIZATION] Pre-computing static features...")
with torch.no_grad():
    static_features = {
        'edge_length_emb': edge_length_emb.detach(),
        'edge_sh': edge_sh.detach(),
        'edge_type_onehot': F.one_hot(edge_type_idx, num_edge_types).float(),
    }

# Register as buffers to move with model
network.register_buffer('edge_length_emb', static_features['edge_length_emb'])
network.register_buffer('edge_sh', static_features['edge_sh'])
network.register_buffer('edge_type_onehot', static_features['edge_type_onehot'])

# In training loop: no recomputation needed
for epoch in range(num_epochs):
    pred_raw = network(
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        network.edge_length_emb,  # ← pre-computed
        network.edge_sh,           # ← pre-computed
        batch_node,
        batch_edge,
    )
```

---

### 3. Vectorized Lookup Dict Construction

**File:** Training loop
```python
# Build lookup once before training
print("\n[OPTIMIZATION] Building edge lookup table...")
edge_lookup = {}
for key, edges_5d in target_H_matrix.pair_edges.items():
    for idx in range(edges_5d.shape[1]):
        sx, sy, sz, i, j = edges_5d[:, idx].tolist()
        edge_lookup[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

# No need to rebuild in training loop - edges are static!
```

---

## Testing & Validation

### Correctness Tests:
```python
# Test that optimizations don't break equivariance
def test_optimization_correctness():
    network_original = MinimalNetwork(...)
    network_optimized = MinimalNetworkOptimized(...)

    # Same weights
    network_optimized.load_state_dict(network_original.state_dict())

    # Test forward pass
    out_orig = network_original(test_input)
    out_opt = network_optimized(test_input)

    assert torch.allclose(out_orig, out_opt, atol=1e-5)
```

### Performance Benchmarks:
```python
import time

def benchmark_forward_pass(network, inputs, num_iterations=100):
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = network(*inputs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations
```

---

## Risk Assessment

| Optimization | Risk | Mitigation |
|--------------|------|------------|
| Remove prints | ⚠️ Low | Keep verbose flag for debugging |
| Pre-compute features | ⚠️ Low | Only for static graphs, validate outputs |
| torch.jit | 🔴 High | Extensive testing, gradual rollout |
| Custom CUDA kernels | 🔴 High | Expert implementation, thorough validation |
| Mixed precision | ⚠️ Medium | Monitor for NaN/Inf, use loss scaling |
| Irrep-space loss | ⚠️ Medium | Validate mathematical equivalence |

---

## Conclusion

**Priority 1 (Do First):**
1. Remove print statements
2. Pre-compute static features
3. Vectorize IrrepsBlockData construction
4. Batch WandB logging

**Priority 2 (After validation):**
5. JIT-compile simple modules (encoder, head, loss)
6. Optimize e3LayerNorm for single-batch case
7. Implement irrep-space loss

**Priority 3 (Advanced):**
8. Mixed precision training
9. Explore torch.compile
10. Custom CUDA kernels (if needed)

**Expected total speedup:** **5-10x** for end-to-end training on single-molecule overfitting.

For larger datasets/models, gains will be even more substantial as Python overhead becomes more dominant.

---

**Next Steps:**
1. Profile current code to validate bottleneck analysis
2. Implement Phase 1 optimizations (2-3x speedup, low risk)
3. Measure and iterate
4. Proceed to Phase 2 only after Phase 1 is validated
