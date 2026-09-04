Triton is a **blocked SPMD** programming model with a **JIT compiler**. You write Python that looks like NumPy; Triton compiles it into a real GPU kernel. You do not write CUDA C, and you do not pick `threadIdx`.

1. What a “program” is

CUDA’s unit of work is a **thread**: one index, one element.

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx] * input[idx];
```

Triton’s unit of work is a **program** (sometimes called a program instance). One program owns a **tile** of elements and does vector ops on the whole tile.

For your square kernel with `BLOCK_SIZE=1024` and `n=1000`:

| Who | What it owns |
|---|---|
| CUDA thread 0 | element 0 |
| CUDA thread 1 | element 1 |
| … | … |
| **Triton program 0** | elements **0–1023** (mask kills 1000–1023) |

If `n=5000` and `BLOCK_SIZE=1024`, you launch `triton.cdiv(5000, 1024) = 5` programs:

| `pid` | offsets | mask keeps |
|---|---|---|
| 0 | 0–1023 | all |
| 1 | 1024–2047 | all |
| 2 | 2048–3071 | all |
| 3 | 3072–4095 | all |
| 4 | 4096–5119 | 4096–4999 only |

Each program runs the **same kernel body**. The only thing that differs is `pid`. That is SPMD: Single Program, Multiple Data.

2. What `tl.program_id` actually is

```python
pid = tl.program_id(axis=0)
```

This is Triton’s analog of `blockIdx.x`.

When you launch:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
square_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
```

you are saying: “start `ceil(n / 1024)` independent copies of this kernel.” The GPU assigns each copy an id:

- `tl.program_id(0)` → 0, 1, 2, … along the x grid
- `tl.program_id(1)` → y grid (2D kernels: tiles of a matrix)
- `tl.program_id(2)` → z grid

There is **no** `threadIdx` in the source. Inside one program, work is a **vector**, not a loop over threads you wrote:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, BLOCK_SIZE)` is a compile-time vector `[0, 1, …, 1023]`.  
If `pid == 2` and `BLOCK_SIZE == 1024`, then `offsets` is `[2048, 2049, …, 3071]`.

That one line replaces the CUDA index math **and** the implicit parallelism inside the block. Triton’s compiler later maps that vector onto real CUDA threads / warps / tensor cores. You never see those threads in the Python.

CUDA mapping, roughly:

| CUDA | Triton |
|---|---|
| `blockIdx.x` | `tl.program_id(0)` |
| `blockDim.x` | `BLOCK_SIZE` (`tl.constexpr`) |
| `threadIdx.x` | hidden; you use `tl.arange` |
| `if (idx < n)` | `mask = offsets < n_elements` |
| `input[idx]` | `tl.load(x_ptr + offsets, mask=mask)` |

`pid` is **not** a Python integer you can `print()` from inside the kernel. It is a device scalar, different in each program instance.

3. How the kernel is built (compiler pipeline)

`@triton.jit` does **not** run the function as Python on the GPU. It marks: “this is a kernel AST; compile it on first launch.”

Your lecture note already says this: first `square_kernel[grid](...)` traces/compiles; later calls reuse the cache.

Pipeline, simplified:

```
Python source
    │  @triton.jit  (on first launch with concrete args)
    ▼
Triton traces the function  (not CPython bytecode on GPU)
    ▼
TTIR   Triton IR          — loads, stores, arange, masks, constexpr folded
    ▼
TTGIR  Triton GPU IR      — tiles mapped to warps, shared memory, layouts
    ▼
LLVM IR
    ▼
PTX    (NVIDIA assembly)  — this is what you were reading in the profiling notebook
    ▼
CUBIN  (GPU binary)       — actually launched
```

Important details:

**It is not Python on the GPU.** `tl.load`, `tl.arange`, `x * x` are Triton ops. Inside the jit function you cannot use arbitrary Python (`print`, lists, NumPy). Only Triton language, a few Python literals, and other `@triton.jit` functions.

**Specialization.** The first launch with `BLOCK_SIZE=1024`, `float32` pointers, this GPU architecture produces one compiled kernel. A different `BLOCK_SIZE` or dtype compiles **another** binary. That is why `BLOCK_SIZE: tl.constexpr` exists: it is baked into the code like a C++ template parameter, not a runtime argument.

**Pointers.** The jit docstring you already printed: if an argument has `.data_ptr()` and `.dtype` (a `torch.Tensor`), Triton passes the raw device pointer. So `square_kernel[grid](x, y, ...)` turns `x` into `x_ptr`.

**Cache.** After the first compile, later launches with the same specialization just enqueue the CUBIN. First call is slow; later calls are kernel-launch cheap.

**You do not write PTX.** Triton writes it. Your profiling notebook notes that Triton emits IR/PTX, not CUDA C. The Python is a **frontend**; the GPU runs the compiled binary.

4. What happens at launch, step by step

Take `n = 1000`, `BLOCK_SIZE = 1024`.

**Host (Python wrapper):**

1. `y = torch.empty_like(x)` — allocate output on GPU.
2. `grid = (1,)` because `cdiv(1000, 1024) = 1`. One program is enough.
3. `square_kernel[grid](x, y, 1000, BLOCK_SIZE=1024)`
   - First time: compile for this `BLOCK_SIZE` / dtype / GPU.
   - Always: launch 1 program instance on the GPU.

**Device (that one program, `pid = 0`):**

```text
offsets = [0, 1, 2, ..., 1023]
mask    = [True, True, ..., True, False, False, ...]   # False from 1000 onward
x       = load 1000 real values; masked lanes get 0 (or `other`)
y       = x * x                                        # vector multiply
store y to y_ptr + offsets, but only where mask is True
```

If `n = 5000`, the host launches 5 programs. Program 3 does indices 3072–4095; program 4 does 4096–5119 with a partial mask. They do not communicate. No `__syncthreads__` is needed for this kernel because each program writes a disjoint slice of `y`.

That is how Triton “writes” the kernel: **you describe one tile; the grid replicates it.**

5. Load / store are vector memory ops

```python
x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
tl.store(y_ptr + offsets, y, mask=offsets < n_elements)
```

`x_ptr + offsets` is not one address. It is a **block of pointers**, length `BLOCK_SIZE`. `tl.load` issues a vector/coalesced load of that whole tile. The mask is the OOB guard: false lanes do not read/write, so the last program does not walk off the end of the tensor.

This is why Triton kernels look short. The compiler is supposed to turn that vector load into efficient PTX (`ld.global`, vectorized `ld.global.v4.f32`, etc.). You choose tile size (`BLOCK_SIZE`); it chooses threads and instructions.

6. The two functions you need

You already wrote this split in the compare notebook:

| Function | Runs on | Role |
|---|---|---|
| `square_kernel` (`@triton.jit`) | GPU | One program: compute one tile |
| `square_triton` (normal Python) | CPU | Allocate `y`, pick grid, launch, return `y` |

The `[grid]` syntax is the launch, analogous to `<<<blocks, threads>>>` in CUDA. The kernel body never sees the grid size; it only sees its own `pid` and `BLOCK_SIZE`.

**Short version:** `program_id` is “which tile am I?” Triton compiles the Python tile description into PTX/CUBIN on first launch, then runs one compiled copy per grid slot. You write the tile math (`arange`, `load`, `*`, `store`); the compiler writes the actual GPU kernel.