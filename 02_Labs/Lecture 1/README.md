# Lecture 1 — Understanding CUDA Profiling

Lab notebook: [`Understanding_CUDA_Profiling.ipynb`](Understanding_CUDA_Profiling.ipynb)  
Chrome trace: [`trace.json`](trace.json) (open in [chrome://tracing](chrome://tracing))

**Hardware:** NVIDIA L4 · PyTorch 2.8+cu128

## What we did

Profiled a simple `torch.square` on GPU and compared different ways to measure and inspect GPU work:

1. **`time.time()`** — misleading for GPU ops; CUDA kernels launch asynchronously, so CPU wall time does not reflect actual GPU execution.
2. **CUDA Events + `torch.cuda.synchronize()`** — records GPU-side elapsed time between events after forcing the CPU to wait for queued work.
3. **PyTorch Autograd Profiler** — tabular breakdown of ops (`aten::square`, `cudaLaunchKernel`, etc.).
4. **Chrome Trace export** — visual timeline of CPU ops, memcpy, and kernels (`trace.json`).
5. **Custom CUDA extension** (`load_inline`) — hand-written square kernel, profiled with Nsight Compute.
6. **Triton kernel** — JIT-compiled square; benchmarked vs native PyTorch (`x * x`).
7. **`torch.compile`** — graph capture + Inductor; first call pays compilation cost.

## Key result (from `trace.json`)

Workload: move a 1024×1024 float tensor to GPU, then square it.

| Phase | Time | Notes |
|-------|------|-------|
| H2D copy (`aten::to` / `Memcpy HtoD`) | ~7 ms CPU-side, ~0.8 ms GPU-side | ~4 MB transferred at ~5 GB/s |
| GPU kernel (`aten::pow` / square) | **~5 µs** | Actual compute is tiny |
| `cudaLaunchKernel` (CPU) | dominates profiler table | Launch + sync overhead, not GPU math |

**Takeaway:** For small/elementwise ops, **data movement and launch overhead dominate** — the GPU kernel itself is negligible. Always profile the full pipeline (copy → kernel → sync), not just kernel time.

## Other benchmarks (notebook)

- **Triton vs PyTorch** on 10M elements: ~0.35 ms each — comparable for this simple op.
- **`torch.compile` first run:** ~55 ms total (mostly Dynamo/Inductor compilation); useful only when the compiled path is reused many times.
