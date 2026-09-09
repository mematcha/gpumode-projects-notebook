### CUDA Kernel Debugging Notes

CUDA kernel debugging can be approached in layers:

- **Runtime checks:** use `cudaGetLastError()` after launch and `cudaDeviceSynchronize()` during debugging to catch launch and execution errors.
- **Selective `printf`:** print values from only a few threads/blocks to inspect indexing, coordinates, shared-memory values, etc.
- **Compute Sanitizer:**
  - `memcheck` → invalid/out-of-bounds memory access
  - `racecheck` → shared-memory race conditions
  - `synccheck` → synchronization/barrier errors
- **Reference testing:** use very small deterministic inputs and compare CUDA output against a trusted implementation or another kernel.
- **Debug compilation:** use `-G -g -lineinfo` while debugging; switch back to `-O3 -lineinfo` for performance profiling.

Recommended workflow:

`small correctness test → printf if needed → compute-sanitizer → benchmark → NCU profiling`

> `compute-sanitizer` answers **"Is the kernel incorrect/unsafe?"**, while NCU answers **"Why is the correct kernel slow?"**