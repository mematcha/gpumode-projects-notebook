# Data Contract — Experiment 4

**Date:** June 21, 2026  
**Topic:** What a data contract means for the mean-blur CUDA experiment (Socratic coding session)

## Overview

During Experiment 4 planning, we defined **done** criteria and a full aspect map before writing kernel code. One early build dependency is locking the **data contract**: the agreed specification for what crosses each boundary between notebook, PyTorch bindings, C++ launch layer, and CUDA kernels. This note captures only that slice of the thread — not tiling, kernels, or benchmarks.

## Glossary

| Term | Definition |
|------|------------|
| **Data contract** | The fixed agreement about what data crosses each stage boundary before implementation: dtypes, shapes, layout, device, parameters, and shared rules (e.g. boundary handling). It is the “handshake” spec, not the blur algorithm itself. |
| **Boundary / OOB rule** | How missing neighbors are treated when the blur window extends past the image edge. For this experiment: **zero-fill** — use `0` for out-of-bounds reads. Both kernels must use the same rule or `torch.allclose` will fail. |
| **Handoff** | One crossing point where one layer passes data or parameters to the next (e.g. notebook → Python binding). Each handoff has its own contract slice. |

## FAQ

### Q: What does “data contract” mean in this experiment?

**A:** It is the agreed spec for what is transferred at each stage — not only **data types**, but also **tensor shape**, **memory layout** (contiguous row-major), **device** (CUDA), **parameters** (blur radius, tile size for the tiled kernel), and **semantic rules** both sides must follow (zero-fill at edges). If any layer expects something different, you get integration bugs that look like kernel bugs.

### Q: Is it just “what dtypes touch each other through a function”?

**A:** Types are one part. The fuller picture is: *this stage receives X and must produce Y*. Example: the binding layer receives a PyTorch tensor and must extract raw `float*` + `height` + `width` + `radius` that match what the launch wrapper and kernel expect. Wrong shape or mismatched OOB policy breaks the contract even when dtypes match.

### Q: What is a concrete data contract for Experiment 4?

**A:** **Input:** `float32`, contiguous, CUDA tensor of shape `(H, W)` — single-channel grayscale. **Parameters:** blur **radius** `R` (window side `(2R+1)`). **Output:** `float32` CUDA tensor of shape `(H, W)`. **OOB rule:** zero-fill on both kernels. **Comparison:** same input tensor passed to naive and tiled paths; `torch.allclose` checks they agree.

### Q: Why lock the data contract before writing kernel logic?

**A:** Build order treats it as the first dependency: notebook, bindings, launch code, and kernels must all agree on the handshake. If the naive kernel assumes `(H, W, 3)` but the notebook passes `(H, W)`, or one kernel zero-fills and another clamps edges, verification fails for the wrong reason. Fixing the contract first makes `torch.allclose` a fair correctness check.

### Q: What are the three handoffs we still need to spell out?

**A:** (Open from the session — not yet written by the user.)

1. **Notebook → Python binding** — what arguments the notebook passes (tensor, radius, tile size for tiled path).
2. **Python binding → CUDA launch** — what C++ extracts (`data_ptr`, dimensions, launch config inputs).
3. **Kernel → output** — what each thread writes (one output pixel per thread, same `(H, W)` indexing as input).

Defining these three lines completes the contract before coding.

## Decisions & Actions

- **Decision:** Out-of-bounds neighbors use **zero-fill** in **both** kernels.
- **Decision:** Launch config (grid/block/tile, shared array size) owned by the **C++** layer, not the notebook.
- **Action:** No files written yet; `image_blur.cu` and `image_blur_ext.cpp` still empty.

## Outcome

Conceptual understanding of **data contract** is in place for Experiment 4. The three per-handoff contract lines are the next Socratic step before implementation.

## Next Steps

- Write the three handoffs (notebook → binding → kernel → output) in one line each.
- Validate contract with `TORCH_CHECK` in bindings (pattern from Experiment 3) before kernel logic.
- Implement naive kernel first under this contract, then tiled + shared memory using the **same** contract.