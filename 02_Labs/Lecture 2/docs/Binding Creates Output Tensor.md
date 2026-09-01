# Binding Creates Output Tensor

**Date:** June 21, 2026  
**Topic:** Why the PyTorch C++ binding allocates output (Experiment 4 handoff 1)

## Overview

During Experiment 4 Socratic coding, we defined Python-facing blur signatures and clarified how the **C++ binding layer** handles output memory. Unlike an API where the notebook passes a pre-allocated output buffer, this project follows Experiment 3: the binding validates **input only**, allocates a matching output tensor internally, launches the kernel, and **returns** the result. This note captures that handoff and how it relates to `TORCH_CHECK`.

## Glossary

| Term | Definition |
|------|------------|
| **Binding** | The C++ layer exposed to Python via pybind11 (e.g. `image_blur_naive`). It sits between the notebook and CUDA launch code. |
| **Handoff 1** | Notebook → Python binding: what arguments the notebook passes and what the binding returns. |
| **Caller-passed output** | An API design where Python provides an existing output tensor (e.g. `blur(input, output)`). The binding must validate both tensors match. |
| **Binding-allocated output** | The binding creates the output with `torch::empty(...)` (or equivalent) sized to match the input, then returns it. |
| **Input contract** | The agreed constraints on `input_image`: float32, CUDA, contiguous, shape `(H, W)`. Enforced with `TORCH_CHECK` in C++. |

## FAQ

### Q: What does “the binding creates the output tensor” mean?

**A:** The C++ function called from Python (e.g. `image_blur_naive`) does not take an output argument. Inside that function, it allocates a new GPU buffer—typically with something like `torch::empty({H, W}, input.options())`—passes its pointer to the kernel, and returns the filled tensor to Python. The notebook only supplies `input_image` and parameters (`radius`, `thread_block_size`, etc.).

### Q: Why don’t we validate a caller-passed output?

**A:** Because there is no output argument. When the notebook calls `output = image_blur_ext.image_blur_naive(input, ...)`, the output tensor does not exist until the binding creates it. You cannot `TORCH_CHECK` a tensor the caller never passed. Output shape, dtype, and device are guaranteed because **you** construct the tensor to mirror the validated input.

### Q: How is that different from `blur(input, output)`?

**A:** In a two-argument design, the caller might pass a CPU tensor, wrong shape `(H, W, 3)` instead of `(H, W)`, or non-contiguous memory. The binding would need checks on **both** sides. In the allocate-and-return pattern (Experiment 3 style), input validation plus controlled allocation removes an entire class of output mismatches.

### Q: What do we `TORCH_CHECK` on input for Experiment 4?

**A:** Same family as Experiment 3’s `check_float32_cuda_contiguous`: input must be on **CUDA**, **float32**, **contiguous**, and **2-D** with shape `(H, W)`. Each check prevents a specific failure—e.g. CPU tensor → kernel reads wrong memory; wrong rank → height/width extraction is wrong; non-contiguous → row-major indexing in the kernel misaligns.

### Q: What are the locked Python signatures (handoff 1)?

**A:**

- `image_blur_naive(input_image, radius, thread_block_size) -> torch.Tensor`
- `image_blur_tiled(input_image, radius, thread_block_size, tile_size) -> torch.Tensor`

Both return a new float32 CUDA tensor of the same `(H, W)` shape as `input_image`. `radius` sets blur window size `(2R+1)²`; `tile_size` applies only to the tiled path.

### Q: Where does launch config live?

**A:** The **C++ binding / launch layer** computes grid, block, and (for tiled) shared-memory dimensions from `thread_block_size`, `tile_size`, and image size. The notebook passes parameters; it does not compute CUDA launch geometry.

## Decisions & Actions

- **Decision:** Follow Experiment 3’s allocate-and-return pattern, not caller-passed output.
- **Decision:** Validate input with `TORCH_CHECK`; output inherits input’s `(H, W)` float32 CUDA layout by construction.
- **Decision:** Both blur functions share `radius`; tiled adds `tile_size`.
- **Action:** Handoff 1 signatures recorded in `experiment4.ipynb`; `image_blur.cu` and `image_blur_ext.cpp` still empty.

## Outcome

Handoff 1 (notebook → binding) is conceptually locked. Next Socratic step: handoff 2 (binding → CUDA launch — what raw pointers and scalars cross that boundary).

## Next Steps

- Spell out handoff 2: `data_ptr`, `height`, `width`, `radius`, launch dimensions passed to `launch_*` wrappers.
- Implement binding stubs with input `TORCH_CHECK` and empty output allocation before kernel logic.
- Build naive kernel under the same contract, then tiled + shared memory.
