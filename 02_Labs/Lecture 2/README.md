# **Experiment brief - Experiment 1**

This lab benchmarks element-wise vector addition `c[i] = a[i] + b[i]`) across vector sizes from 1K to 100M, comparing three implementations: a plain C++CPU loop, a hand-written CUDA kernel with explicit host↔device copies, and PyTorch on GPU. The CPU run times only the add loop; the CUDA run splits timing into host-to-device copy, kernel execution, and device-to-host copy; PyTorch times the add on tensors already allocated on the GPU (no PCIe transfers in that number). Running `./run_experiments.sh` compiles the C++/CUDA binaries, sweeps all sizes, and writes `experiment_results.csv`. The goal is to see when the GPU actually wins: at small sizes fixed launch and transfer overhead dominate, but at large sizes the kernel alone can be much faster than CPU—while end-to-end CUDA still loses if you pay full PCIe round-trips every run.

# How to run the experiment

Just run `./run_experiments.sh` on a GPU Runtime. You will see a CSV Generated of the experiment results

# Observations

## What each column means


| Column              | What it measures                                                        |
| ------------------- | ----------------------------------------------------------------------- |
| `CPU_Total_ms`      | Pure CPU add loop only (alloc/init not timed)                           |
| `CUDA_Copy_H2D_ms`  | Host → device copy of `a` and `b`                                       |
| `CUDA_Kernel_ms`    | GPU kernel only                                                         |
| `CUDA_Copy_D2H_ms`  | Device → host copy of `c`                                               |
| `CUDA_Total_ms`     | Sum of the three CUDA phases                                            |
| `PyTorch_Kernel_ms` | `a + b` on **already-GPU-resident** tensors (no H2D/D2H in this number) |


So **CUDA_Total** is end-to-end GPU workflow; **PyTorch_Kernel** is compute-only, closer to **CUDA_Kernel**.

---

## Main takeaways

### 1. GPU loses end-to-end for most sizes here — memory transfer dominates

For raw CUDA, copies dwarf the kernel until very large N:


| Vector size | Kernel  | H2D + D2H | Total    | Transfer share of total |
| ----------- | ------- | --------- | -------- | ----------------------- |
| 1M          | 0.32 ms | ~3.0 ms   | 3.31 ms  | ~91%                    |
| 10M         | 0.71 ms | ~25.9 ms  | 26.56 ms | ~97%                    |
| 100M        | 5.23 ms | ~256 ms   | 261 ms   | ~98%                    |


At **100M**, the kernel is ~~5 ms but **PCIe round-trip** (~~171 ms H2D + ~~85 ms D2H) makes **CUDA_Total (~~261 ms) slower than CPU (~98 ms)**.

**Lesson:** GPU speedups only show up when data already lives on the device, you amortize transfers over many kernels, or compute is heavy relative to I/O. Vector add is the classic “transfer-bound” case.

---

### 2. Kernel-only: GPU wins big at large N

Compare CPU vs **CUDA_Kernel** (fairer compute comparison):


| Vector size | CPU     | CUDA kernel | Speedup |
| ----------- | ------- | ----------- | ------- |
| 1M          | 0.99 ms | 0.32 ms     | ~3×     |
| 10M         | 9.98 ms | 0.71 ms     | ~14×    |
| 100M        | 98.5 ms | 5.23 ms     | ~19×    |


That ~19× at 100M is what you expect once compute dominates on-device.

---

### 3. Small N: kernel time is mostly launch/overhead, not math

From 1K → 1M, **CUDA_Kernel** stays roughly **0.24–0.32 ms** while N grows 1000×. The GPU isn’t doing the work faster per element — fixed **kernel launch + synchronization** dominates.

Same pattern for CPU at **N=1000 → `0.000` ms**: sub-millisecond, rounded to 3 decimals in the script.

**Lesson:** GPUs need enough parallel work to hide fixed costs.

---

### 4. PyTorch vs hand-written CUDA kernel — not a perfect apples-to-apples compare

From the source:

- CUDA uses `float32`
- PyTorch uses `float64`

So PyTorch moves **2× the bytes** per element. At 100M:

- CUDA kernel: **5.23 ms**
- PyTorch kernel: **17.34 ms** (~3.3× slower)

Part of that gap is dtype, not “PyTorch is slow.” PyTorch also uses fused/optimized kernels and includes `c.copy_(a + b)` in the timed region.

At small N, PyTorch looks faster (e.g. 1K: 0.126 vs 0.28 ms) — again overhead noise and different setup, not meaningful compute throughput.

---

### 5. CPU scaling is roughly linear (as expected)


| N    | CPU (ms) | ms per 1M elements |
| ---- | -------- | ------------------ |
| 1M   | 0.985    | ~1.0               |
| 10M  | 9.977    | ~1.0               |
| 100M | 98.485   | ~0.98              |


Simple O(n) loop on CPU — good sanity check that timing is sensible.

---

### 6. Transfer bandwidth (back-of-envelope at 100M)

At N=100M, float32:

- H2D: 2 vectors × 400 MB ≈ **800 MB** in **171 ms** → ~**4.7 GB/s**
- D2H: 1 vector ≈ **400 MB** in **85 ms** → ~**4.7 GB/s**

Plausible for PCIe; confirms the story that **you’re measuring the bus, not the ALUs**.

---

```text
Small N     → CPU often wins (or GPU wins only if you ignore copies)
Medium N    → GPU kernel faster, but total CUDA still loses to CPU
Large N     → GPU kernel ~20× faster; end-to-end CUDA still loses without reuse
PyTorch col → kernel-only timing; good model for “training step” when tensors stay on GPU
```

The core GPUMODE lesson: **profile the full pipeline** (H2D → kernel → D2H), not just the kernel. A fast kernel doesn’t help if you pay 256 ms moving data for 5 ms of math.

---

## Kernel vs CPU (from your CSV)

**Small N (1K–100K)**

- CPU wins: **0.000–0.032 ms** vs kernel **~0.24–0.29 ms**
- GPU kernel time barely changes with size → **launch overhead**, not real compute
- CPU at 1K shows **0.000** (sub-ms, rounded)

**Medium N (1M)**

- CPU **0.99 ms** vs CUDA kernel **0.32 ms** → GPU ~**3×** faster (compute only)
- Still not a big win; fixed GPU cost still matters

**Large N (10M–100M)**

- CPU scales linearly: **10 ms → 98 ms**
- Kernel scales slowly: **0.71 ms → 5.2 ms**
- At **100M**: CPU **98.5 ms** vs kernel **5.2 ms** → ~**19×** GPU speedup

**Takeaway**

- **Small data:** CPU is faster (or tied) — GPU overhead dominates
- **Large data:** GPU kernel pulls ahead sharply once work amortizes launch cost
- Compare **CPU_Total** vs **CUDA_Kernel** for fair compute-only; **CUDA_Total** includes copies and is much slower end-to-end

---

## Caveats when interpreting this sheet

1. **Single run, no warmup** for CPU/CUDA (PyTorch does warm up) — numbers can jitter run-to-run.
2. **CPU `0.000` at 1K** is rounding, not zero work.
3. **PyTorch vs CUDA kernel** differs in dtype (`float64` vs `float32`) — normalize before comparing.

---

# Experiment Brief - Experiment 2

Experiment 2 fixes the same vector-add CUDA kernel as Experiment 1 but sweeps **block size** (32, 64, 128, 256, 512, 1024 threads per block) across vector sizes from 1K to 100M, using `vector_add_var_block_cuda.cu` and `./run_experiments.sh` (or the second half of that script). For each `(N, block_size)` pair it records blocks per grid and the usual timings: H2D copy, kernel execution, D2H copy, and total end-to-end time, writing results to `block_observations.csv`. The goal is to see how launch configuration affects **kernel time** once N is large enough — while PCIe copy times stay roughly flat for a given N — and whether any block size gets meaningfully closer to the GPU’s memory-bandwidth limit. At small N, kernel time is mostly fixed overhead and block size barely matters; at 100M elements, kernel time drops from ~9.8 ms (block 32) to ~4.7 ms (64–1024), after which block size has little effect because the workload is memory-bound, not compute-bound.

# How to run the experiment

Just run `./run_experiments.sh` on a GPU Runtime. You will see a CSV Generated of the experiment results.

# Observations

### Copies (H2D / D2H) — block size doesn’t matter much

- For a fixed vector size, **copy times are nearly the same** across all block sizes.
- Copies depend on **how many bytes** you move, not how you launch the kernel.
- At **100M**: H2D ~**172 ms**, D2H ~**85–88 ms** regardless of block size.
- **Total time is still transfer-dominated** at large N (~260–270 ms total vs ~5 ms kernel).

---

### Small vectors (1K–100K) — block size barely helps

- Kernel time stays around **~0.10–0.20 ms** for all block sizes.
- Too little work → **launch overhead** dominates, not memory bandwidth.
- Differences between block 32 and 1024 are mostly noise.

---

### Medium vectors (1M) — still flat

- Kernel ~**0.15–0.16 ms** for every block size.
- End-to-end ~~**3.1 ms** — still mostly **PCIe copies** (~~3 ms transfer vs ~0.15 ms kernel).

---

### Large vectors (10M+) — block size starts to matter

- **Block 32 is slowest** on kernel time:
  - 10M: **0.68 ms** (block 32) vs **~0.57–0.64 ms** (others)
  - 100M: **9.77 ms** (block 32) vs **~4.7–4.8 ms** (64–1024)
- **~2× kernel speedup** at 100M just by moving from block 32 → 64+.

---

### Very large N (100M) — block size plateaus after 64

- Best kernel times cluster around **~4.7–4.8 ms** for blocks **64, 128, 256, 512, 1024**.
- Going from 64 → 1024 **doesn’t help much** — you’re **memory-bound**, not launch-bound.
- Too many tiny blocks (32 → 3.1M blocks) adds scheduling overhead.

---

### Rule of thumb from your data

| Vector size | Block size effect |

|-------------|-------------------|

| **≤ 100K** | Ignore — overhead dominates |

| **1M** | Ignore — copies dominate total |

| **10M** | Small effect (~15–20% kernel spread) |

| **100M** | Big effect at 32; **64+ is fine** |

---

### Bottom line

- **Tuning block size only matters when N is large enough** that the kernel does real work.
- **Block 32 is a bad choice at 100M** for this naive kernel.
- **Block 64–256 is a sweet spot** in your results — good occupancy without millions of blocks.
- Even with a tuned block size, **PCIe transfers still dominate end-to-end time** for a one-shot vector add.

