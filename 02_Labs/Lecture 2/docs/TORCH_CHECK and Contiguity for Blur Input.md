# TORCH_CHECK and Contiguity for Blur Input

**Date:** June 21, 2026  
**Topic:** Input validation (`TORCH_CHECK`) and contiguity requirements for Experiment 4 blur bindings

## Overview

During Experiment 4 Socratic coding (handoff 1 — notebook → binding), we locked Python-facing signatures and clarified what the C++ binding must validate on `input_image` before launch. The user initially mixed kernel boundary policy and output-shape checks into `TORCH_CHECK`; we separated those layers. We also explained why **`is_contiguous()`** is required when the kernel indexes with `row * width + col`. Stub forward declarations exist in `image_blur_ext.cpp`; binding validation logic is not yet implemented.

## Glossary

| Term | Definition |
|------|------------|
| **`TORCH_CHECK`** | PyTorch C++ macro that throws a clear error if a precondition fails (e.g. wrong device or dtype). Used in the binding layer before kernel launch. |
| **Input contract** | Agreed constraints on `input_image`: CUDA, float32, 2-D `(H, W)`, H/W > 0, contiguous. Enforced at the binding, not inside the kernel. |
| **Contiguous tensor** | Elements stored in one unbroken row-major block so `index = row * width + col` matches physical memory order. |
| **Non-contiguous tensor** | Logical shape `(H, W)` is valid in Python, but memory layout has gaps or reordering (e.g. after `.t()`). Simple row-major indexing reads wrong pixels. |
| **Boundary / OOB policy** | How blur handles neighbors outside the image (zero-fill in this experiment). Lives in **kernel logic**, not in input `TORCH_CHECK`. |
| **Binding-allocated output** | Binding creates output with same `(H, W)` as input; no caller-passed output to validate. |

## FAQ

### Q: What should I `TORCH_CHECK` on `input_image` before launch?

**A:** On the **input tensor** only:

- **`input.is_cuda()`** — prevents launching a kernel on host memory (crash or garbage).
- **`scalar_type() == torch::kFloat32`** — kernel reads raw `float*`; wrong dtype misinterprets bytes.
- **`input.dim() == 2`** — blur expects `(H, W)`; wrong rank makes `size(0)`/`size(1)` mean the wrong thing.
- **`height > 0` and `width > 0`** — empty image yields invalid grid/launch or no work.
- **`input.is_contiguous()`** — kernel uses `row * width + col`; non-contiguous layout breaks that formula.

This mirrors Experiment 3’s `check_float32_cuda_contiguous` in `rgb_grayscale_ext.cpp`, but with **`dim() == 2`** instead of 3.

### Q: The user listed zero-fill at edges and same input/output shape — are those `TORCH_CHECK`?

**A:** No — those belong elsewhere:

- **Zero-fill at edges** → **kernel / cross-cutting policy** when the blur window extends past the image border. It is not a property of the input tensor you validate at the door.
- **Input and output same shape** → **binding allocation**. The binding creates `output` with the same `(H, W)` as the validated input. There is no caller-passed output to check (see `Binding Creates Output Tensor.md`).

### Q: What checks belong on parameters instead of `input_image`?

**A:** Validate launch-related scalars separately:

- **`radius >= 0`** — negative radius breaks window size `(2R + 1)`.
- **`thread_block_size > 0`** (and typically ≤ 1024) — invalid block size breaks launch.
- **`tile_size > 0`** (tiled path only) — invalid tile breaks shared-memory sizing.

### Q: Why does contiguity matter if shape and dtype look correct?

**A:** The kernel formula `index = row * width + col` assumes **row-major contiguous** storage: all of row 0’s `W` floats, then row 1’s, and so on. A non-contiguous tensor (common after **transpose** `.t()`) still displays as `(H, W)` in Python, but logical `(row, col)` is not at that linear index in memory. The kernel reads the wrong slots → scrambled blur or failed `torch.allclose` even when blur math is correct.

### Q: Does requiring contiguity mean the index formula “expects” contiguity?

**A:** Yes. Choosing `row * width + col` is a **contract**: you promise memory is one tight row-major block. `TORCH_CHECK(input.is_contiguous())` enforces that contract at the binding. Without it, you would need **strides** (skip distances per row/column) in the kernel instead of simple linear indexing.

### Q: What is non-contiguity in plain terms?

**A:** **Contiguous** = memory order matches your row-by-row index formula. **Non-contiguous** = PyTorch’s view of rows/columns and the actual buffer order diverged (e.g. transpose swapped semantics without moving data). You can still process non-contiguous tensors, but not with a naive `row * width + col` pointer offset alone.

### Q: What are the locked Python signatures (handoff 1)?

**A:**

- `image_blur_naive(input_image, radius, thread_block_size) -> torch.Tensor`
- `image_blur_tiled(input_image, radius, thread_block_size, tile_size) -> torch.Tensor`

Return: new float32 CUDA tensor, same `(H, W)` as input.

## Decisions & Actions

- **Decision:** Input validation list = CUDA · float32 · 2-D · H,W > 0 · contiguous.
- **Decision:** OOB zero-fill is kernel policy, not `TORCH_CHECK` on input.
- **Decision:** Output shape enforced by binding allocation, not dual tensor checks.
- **Action:** Forward declarations added in `experiment 4/image_blur_ext.cpp` (incomplete `PYBIND11_MODULE` stub).
- **Action:** Related doc: `Binding Creates Output Tensor.md`.

## Outcome

Handoff 1 input contract is defined. User understands why contiguity pairs with row-major indexing. Binding `TORCH_CHECK` helper and handoff 2 (binding → launch: `data_ptr`, dimensions, launch config) are next.

## Next Steps

- Implement `check_blur_input`-style helper with the five input checks (+ parameter checks).
- Complete handoff 2 contract: what crosses from binding to `launch_*` wrappers.
- Allocate output with `torch::empty({H, W}, input.options())` after checks pass.
- Implement naive kernel first under zero-fill OOB policy.
