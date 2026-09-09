# CUDA Programming Guide for NCU

- CUDA is a **parallel computing platform and programming model** developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU.
- The CUDA Programming Guide is the **official, comprehensive resource on the CUDA programming model** and how to write code that executes on the GPU using the CUDA platform.

We will cover only select chapters for reference and understanding

## The CUDA Programming Model

The code an application executes on the GPU is referred to as device code, and a function that is invoked for execution on the GPU is, for historical reasons, called a kernel. The act of starting a kernel running is called launching the kernel.

**A kernel launch can be thought of as starting many threads executing the kernel code in parallel on the GPU.**

> A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. 

> A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.



## Thread Blocks and Grids

When an application launches a kernel, it does so with many threads, often millions of threads. These threads are organized into blocks. A block of threads is referred to, perhaps unsurprisingly, as a thread block. Thread blocks are organized into a grid. All the thread blocks in a grid have the same size and dimensions.

**All threads of a thread block are executed in a single SM.**

This allows threads within a thread block to communicate and synchronize with each other efficiently. **Threads within a thread block all have access to the on-chip shared memory**, which can be used for exchanging information between threads of a thread block.

In addition to thread blocks, GPUs with compute capability 9.0 and higher have an optional level of grouping called **clusters**. Clusters are a group of thread blocks which, like thread blocks and grids, can be laid out in 1, 2, or 3 dimensions.

Specifying clusters groups adjacent thread blocks into clusters and provides some additional opportunities for synchronization and communication at the cluster level. **Specifically, all thread blocks in a cluster are executed in a single GPC.**

Because the thread blocks are scheduled simultaneously and within a single GPC, threads in different blocks but within the same cluster can communicate and synchronize with each other using software interfaces provided by **[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-cooperative-groups)**. 

Threads in clusters can access the shared memory of all blocks in the cluster, which is referred to as **distributed shared memory.** The maximum size of a cluster is hardware dependent and varies between devices.

## Warps and SIMT

Within a thread block, threads are organized into groups of 32 threads called **warps**. A warp executes the kernel code in a ***Single-Instruction Multiple-Threads* (SIMT) paradigm.**

**In SIMT, all threads in the warp are executing the same kernel code**, but each thread may follow different branches through the code (divergence). That is, though all threads of the program execute the same code, threads do not need to follow the same execution path.

**SIMT (the warp):** 32 threads share one **instruction stream**. On a given cycle the warp issues **one instruction**; all 32 apply it to **their own data**.

While it is not necessary to consider warps when writing CUDA code, understanding the warp execution model is helpful in understanding concepts such as ++[global memory coalescing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-coalesced-global-memory-access)++ and ++[shared memory bank access patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns)++.

## Performance Guidelines

Four official levers: keep the GPU busy, move data efficiently, issue useful instructions, and avoid extra memory/driver churn.

**1. Maximize utilization** — keep SMs fed with work  

- **App:** overlap CPU, GPU, and copies with streams (do not stall on every `cudaMemcpy` / kernel).  

- **Device:** launch enough threads/blocks that every SM has something to run `N = 2**10` square is too small; `2**26` saturates).  

- **SM:** occupancy = active warps / max warps. Block size a multiple of 32; too many registers or too much shared memory cuts warps per SM.

**2. Maximize memory throughput** — bandwidth is usually the limit (your square kernel)  

- **PCIe:** few, large transfers; keep data on the GPU between kernels; pinned host memory if you must copy.  

- **Global:** warp threads should hit consecutive addresses (coalescing) so one transaction serves 32 lanes.  

- **Shared:** reuse data in on-chip SRAM; avoid bank conflicts that serialize the warp.

**3. Maximize instruction throughput** — do not stall the warp  

- Same path inside a warp `if (idx < n)` on the last partial warp is the usual hit). Divergence serializes branches.  

- Fast math `__sinf`) when precision allows.  

- Smaller types / Tensor Cores (FP16, BF16, TF32) when the algorithm is compute-heavy, not a tiny elementwise square.

**4. Minimize memory thrashing / extra overhead** — do not keep restarting work  

- Launch cost is real (your flat latency at small `N`). CUDA Graphs help tight loops of tiny kernels.  

- Unified Memory: without `cudaMemAdvise` / prefetch, pages migrate on fault and the GPU waits.

For this lab: square is **memory-bound**. Occupancy and coalescing matter; Tensor Cores and fast `sin` do not. Small `N` fails (1) and (4) — not enough blocks, launch overhead dominates `8N/T`.

