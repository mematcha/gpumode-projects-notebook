# OOB Padding Threads and Grid Guard

**Date:** June 21, 2026  
**Topic:** Out-of-bounds errors when CUDA grid threads exceed image dimensions (Experiment 4 handoff 2)

## Overview

While defining launch geometry for the naive blur kernel, we worked through why **`ceil` grid sizing** spawns extra “padding” threads past the right and bottom edges of an `(H, W)` image. Without an in-kernel bounds guard, those threads read and write past the allocated tensor buffers — undefined behavior, not a friendly PyTorch error. This note captures that failure mode and the fix.

## Glossary

| Term | Definition |
|------|------------|
| **Padding threads** | Threads launched because the grid is rounded up with `ceil`; their `(row, col)` fall outside `0..H-1` or `0..W-1`. |
| **Grid guard** | In-kernel check `if (row < H && col < W)` (or equivalent) so padding threads exit before read/write. |
| **`gridDim` / `blockDim`** | CUDA launch dimensions: number of blocks and threads per block. Sized with `ceil` so the image is fully covered. |
| **OOB write** | Writing `output[row * width + col]` when `row ≥ H` or `col ≥ W` — index exceeds the `(H × W)` buffer. |
| **Undefined behavior** | No guaranteed error message; possible corruption, wrong pixels, or GPU fault. |

## FAQ

### Q: Why do we launch more threads than `H × W` pixels?

**A:** Blocks are fixed size (e.g. `16×16`). Grid dimensions use **`ceil(width / Bx)`** and **`ceil(height / By)`** so every real pixel gets at least one thread. When `H` or `W` is not a multiple of the block size, the last block row/column includes threads that map to coordinates **outside** the image. Those are padding threads.

### Q: Example — `H = 100`, `W = 200`, block `16×16`. What is `gridDim`?

**A:** `gridDim.x = ⌈200 / 16⌉ = 13`, `gridDim.y = ⌈100 / 16⌉ = 7`. Threads address columns `0..207` and rows `0..111`, but valid pixels are only `col < 200` and `row < 100`.

### Q: How many “extra” slots exist along each axis?

**A:** `13×16 − 200 = **8**` extra column indices (cols 200–207) and `7×16 − 100 = **12**` extra row indices (rows 100–111). Any thread with `col ≥ 200` or `row ≥ 100` is a padding thread.

### Q: What goes wrong if padding threads run the blur without a guard?

**A:** Wrong intuition: “the thread errors out cleanly.” **Correct:** the thread still executes. It computes `output[row * width + col]` and reads a `(2R+1)²` neighborhood from `input`. Indices past `H-1` or `W-1` are **out of bounds** — illegal writes to `output` and illegal reads from `input`. That is **undefined behavior** (corruption, nonsense values, or crash), not a catchable Python exception.

### Q: What is the fix?

**A:** At the start of the kernel body, compute `row` and `col` from block/thread indices, then:

```text
if (row >= height || col >= width) return;
```

Only in-bounds threads perform the blur. Padding threads become no-ops. This matches the Experiment 3 pattern of `if (tid < num_pixels)` for 1-D launches.

### Q: Is this related to zero-fill at the **image** border?

**A:** No — different layer. **Grid guard** handles threads that should not run at all. **Zero-fill OOB** handles blur windows that extend past the **image edge** for valid pixels (e.g. neighbor at `col = -1` → use `0`). Both kernels need the same image-edge rule for `torch.allclose`; only the grid guard prevents buffer overruns from launch geometry.

### Q: Who computes `gridDim` — binding or `.cu` launch wrapper?

**A:** The **launch wrapper in `.cu`** (same as Experiment 3’s `launch_rgb_grayscale_hwc`). The binding passes `H`, `W`, and `thread_block_size`; the wrapper computes `gridDim`/`blockDim` and issues `<<<>>>`. The guard lives in the **`__global__` kernel**.

## Decisions & Actions

- **Decision:** Use `ceil` grid sizing for 2-D thread-to-pixel mapping; always guard with `row < H && col < W`.
- **Decision:** Distinguish grid padding OOB (must guard) from image-edge zero-fill (blur policy).
- **Action:** Launch geometry discussed during handoff 2; kernels in `experiment 4/image_blur.cu` still stubs.

## Outcome

Padding-thread OOB failure mode is understood with a concrete numeric example (`13×7` grid, `8`/`12` overshoot). Next: complete tiled handoff 2 (`tile_size` → shared array size), then implement naive kernel with guard + zero-fill neighbor reads.

## Next Steps

- Finish `launch_image_blur_naive` grid math and kernel guard in `image_blur.cu`.
- Spell tiled launch contract: extra `tile_size` arg; shared patch `(tile_size + 2R)²`.
- Verify on small tensor in notebook before benchmarking.
