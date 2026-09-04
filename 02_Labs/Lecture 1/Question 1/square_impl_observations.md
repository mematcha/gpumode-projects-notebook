# Square implementations — plot observations

Hardware: Tesla T4. Metric: GPU latency from CUDA events; effective bandwidth `8N / T` (float32 read + write).

What looks right

- `torch.square`, `x*x`, and `torch.pow` overlap. Same work: one elementwise kernel, memory-bound.
- Latency is flat until about `N = 2^18`. Small `N` is launch + timer overhead, not multiply. Bandwidth near 0 at `N = 2^10` is expected (`8N` is tiny, `T` is a fixed ~0.01–0.02 ms).
- At `N = 2^26` the curves meet around 2–3 ms and ~230–250 GB/s. That is ~512 MB moved; at 240 GB/s, `T ≈ 2.1 ms`. T4 peak DRAM is ~320 GB/s, so ~75% of peak is a normal streaming kernel. `8N/T` is in the right units.
- Hand CUDA and Triton lag a bit at medium `N` (Python / `empty_like` / extension launch vs cached ATen), then catch PyTorch once the GPU is busy. Expected.

What feels wrong — and is

- `compile(x*x)` and `compile(triton)` are ~10× slower until huge `N`. `compile(triton)` peaks around 180 GB/s instead of ~240.
- `torch.compile` is a poor fit for a one-line square. Eager `x*x` is already one tiny kernel. Compile adds Dynamo/Inductor (guards, extra allocs, sometimes extra kernels) and a higher GPU-time floor (~0.1–0.2 ms) until `N` hides it.
- `torch.compile(square_triton)` is mismatched. Compile traces PyTorch ops; wrapping a `kernel[grid](...)` launcher often graph-breaks, so you pay inductor overhead plus the Triton launch. The lower peak bandwidth suggests extra bytes moved, not a better square.
- The loop changes `N` every time. Compile specializes on shape, so each size can recompile. Warmup hides CPU compile time from CUDA events, but the compiled path can still be unsettled.

What to trust

| Curve | Trust? |
|---|---|
| `torch.square` / `x*x` / `torch.pow` | Yes — same kernel family, memory-bound |
| `cuda`, `triton` at large `N` | Yes — they hit the same bandwidth wall |
| Compile until `N = 2^22` | Overhead, not kernel quality |
| `compile(triton)` peak ~180 GB/s | Check kernel count with `torch.profiler` |

Takeaway: compare eager PyTorch vs CUDA vs Triton. Use `torch.compile` only on `x*x`, at a fixed `N` after a long warmup. Do not compile a hand-written Triton launcher.
