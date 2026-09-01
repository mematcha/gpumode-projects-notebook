# Understanding Image Blur Kernels

**Date:** June 21, 2026  
**Topic:** Experiment 4 — mean filter with and without shared memory (Socratic walkthrough)

## Overview

This thread built conceptual understanding for **Experiment 4**: implement two CUDA mean-blur kernels (naive global-memory reads vs tiled shared-memory), verify they match with `torch.allclose`, and benchmark them. We progressed from “what is a mean blur?” through neighborhood overlap, the **halo** region, global-read accounting, boundary policy, block/tile sizing, cooperative loading, and the four-phase tiled kernel pipeline. No kernel code was implemented yet; stub files exist under `02_Labs/Lecture 2/experiment 4/`.

## Glossary

| Term | Definition |
|------|------------|
| **Mean blur** | Image filter where each output pixel is the **average** of a square neighborhood of input pixels centered on that location. |
| **Radius (R)** | Half-width of the blur window in pixels. Window side length is **2R + 1**; area is **(2R + 1)²**. |
| **Neighborhood / window** | The **(2R + 1) × (2R + 1)** input pixels one output thread must read to compute its average. |
| **Naive kernel** | Each thread independently loads its full neighborhood from **global memory** every time — simple but redundant across neighbors. |
| **Tiled kernel** | Each block cooperatively loads one **tile + halo** patch into **shared memory**; threads then blur from shared memory instead of re-fetching from global. |
| **Halo** | Extra **R** pixels loaded on every side of the block’s output tile — inputs required for edge outputs in the tile but **not** outputs this block writes. |
| **Output tile** | The **TILE × TILE** region of output pixels one block is responsible for writing (e.g. 16×16). |
| **Shared tile** | The **(TILE + 2R) × (TILE + 2R)** array in shared memory holding the output tile plus halo. |
| **Global read** | One load of **one input pixel** (one float at one `(row, col)`) from global memory. |
| **Cooperative / strided load** | All threads in a block participate in filling shared memory: thread `t` loads indices `t`, `t + blockSize`, `t + 2×blockSize`, … until all shared cells are filled. |
| **Boundary / OOB rule** | How out-of-bounds neighbors are handled at the **image** edge (zero-fill, edge clamp, or average only valid pixels). Both kernels must use the **same** rule for `torch.allclose` to pass. |
| **`__syncthreads()`** | Block barrier between load and compute phases so no thread reads shared memory before all loads finish. |

## FAQ

### Q: What's a mean blur?

**A:** A mean blur replaces each pixel with the **average brightness of nearby pixels**. For a 3×3 patch (nine values), the output is their sum divided by 9. Visually this smooths sharp edges and reduces noise. It is the spatial analogue of Experiment 3’s grayscale step: instead of mixing **R, G, B** at one location, you mix **neighboring pixels** at one location.

### Q: For radius R = 2, how many input pixels does one output need?

**A:** Radius **R = 2** gives a **5 × 5** window, so **25** input pixels per output. In general the count is **(2R + 1)²**. This is the per-thread read count in the **naive** kernel (one global read per window element).

### Q: Why does the naive kernel repeat work across neighboring threads?

**A:** Horizontally adjacent outputs have heavily overlapping windows. For R = 2, two side-by-side 5×5 windows share **20 of 25** pixels (4 overlapping columns × 5 rows). In the naive kernel each thread loads all 25 independently, so the same global addresses are fetched twice. That redundancy is what shared memory is meant to eliminate.

### Q: What does “global read” mean in this experiment?

**A:** One **global read** = one load of **one input pixel** from global memory. It is **not** “one read per output pixel.” In the naive kernel each thread performs **(2R + 1)²** global reads. Many of those loads hit the same memory locations as neighboring threads, but each thread still issues them unless you share via shared memory (or rely on cache effects, which the naive code model does not assume).

### Q: For one 16×16 block with R = 2, how many global reads do naive vs tiled issue?

**A:** **Naive:** 256 threads × 25 reads = **6,400** global loads per block. **Tiled:** one cooperative load of a **20 × 20** shared tile = **400** global loads per block. Ratio: **6,400 ÷ 400 = 16×** fewer global loads in the tiled version. The tiled kernel still does **400** loads, not 16 — **16×** is a **reduction factor**, not the absolute load count.

### Q: Why can't a block load only its 16×16 output region into shared memory?

**A:** Outputs on the **edge of the tile** still need neighbors **outside** that 16×16 write region. Example: if the block writes output rows 100–115, the output at row 100 needs input rows 98 and 99. Row 99 is an **input** this block must fetch, not an output it writes (that row belongs to the block above). The extra ring of **R** pixels on each side is the **halo**.

### Q: How big is the shared-memory array for a 16×16 output tile and R = 2?

**A:** Side length = **TILE + 2R = 16 + 4 = 20**, so **20 × 20 = 400** floats (~1.6 KB). General formula: **(TILE + 2R)²**. This is separate from the **256** output threads — shared cells (400) and thread count (256) are different numbers.

### Q: What are the four phases of the tiled kernel?

**A:** (1) **Cooperative load** — strided GMEM → shared fill of the **(TILE + 2R)²** patch (with correct boundary handling when mapping global coordinates). (2) **`__syncthreads()`** — wait until all shared cells are loaded. (3) **Blur math** — each thread sums its **(2R + 1)²** window from **shared memory** and divides by **(2R + 1)²** (or your chosen denominator rule). (4) **Write** — each thread writes one output pixel to global memory.

### Q: How do 256 threads load 400 shared cells?

**A:** Use a **strided cooperative load**: thread `t` loads shared indices `t`, `t + 256`, `t + 512`, … while index < 400. First pass fills indices 0–255; **144** cells remain. Threads **0–143** each load one more cell (indices 256–399). Threads **144–255** load only once. General pattern: start at thread ID, step by block size until all shared indices are covered.

### Q: Why must naive and tiled kernels use the same boundary rule?

**A:** `torch.allclose` compares outputs element-wise. At image corners, **zero-fill** (missing neighbors = 0) and **edge clamp** (repeat edge pixel) produce **different averages** even with the same denominator **(2R + 1)²**. Example: top-left pixel `(0, 0)` with R = 2 — zeros pull the sum down; clamping repeats the corner value. Pick one OOB policy, implement it identically in both kernels, then compare.

### Q: Why is 16×16 a sensible starting tile size?

**A:** **256 threads** = one thread per output pixel (simple mapping, same spirit as Experiment 3). Well under the usual **1024 threads/block** CUDA limit (32×32 outputs would hit that ceiling). Shared memory for 20×20 floats (~1.6 KB) is tiny compared to typical **~48 KB/block** limits — shared memory is **not** the bottleneck here. Doubling to 32×32 quadruples threads (256 → 1024) and grows shared array from 20² to 36² (1,296 cells) — a bit **less** than 4× because the fixed **2R** halo does not scale with tile area the same way.

### Q: Which kernel should be implemented first?

**A:** The **naive** kernel first. It is simpler, defines the correct blur math and boundary behavior, and serves as the **reference** for `torch.allclose` when the tiled kernel is added. After both match, benchmark kernel-only time on GPU-resident tensors (Experiment 3 pattern).

### Q: How should correctness be verified?

**A:** Run both custom kernels (via PyTorch C++/CUDA bindings, as in Experiment 3) on the **same input tensor** and check **`torch.allclose(naive_out, tiled_out)`** with chosen `atol`/`rtol`. Timing comparisons are only meaningful after correctness passes.

## Decisions & Actions

- **Decision:** Use **16×16** output tiles and **R = 2** as the lab’s concrete numbers for reasoning (20×20 shared tile, 256 threads, 16× global-traffic reduction vs naive block).
- **Decision:** Treat **(2R + 1)²** as the denominator consistently (full window area), but enforce the **same OOB neighbor policy** in both kernels.
- **Decision:** Implement **naive first**, then tiled; validate with `torch.allclose` before benchmarking.
- **Action:** Socratic Q&A session — no kernel implementation in this thread.
- **Action:** Stub files present: `02_Labs/Lecture 2/experiment 4/image_blur.cu`, `image_blur_ext.cpp`, `experiment4.ipynb` (problem notes only).

## Outcome

Conceptual understanding is complete for naive vs tiled mean-filter kernels: blur math, redundant global reads, halo sizing, cooperative loading, synchronization, boundary matching, and tile-size tradeoffs. Code is not yet implemented.

## Next Steps

- Implement and verify the **naive** kernel in `image_blur.cu` + PyTorch bindings (mirror Experiment 3 structure).
- Implement the **tiled** kernel with matching OOB rule, `__shared__` tile, strided load, and `__syncthreads()`.
- Benchmark both on GPU-resident tensors; record results in `experiment4.ipynb`.
