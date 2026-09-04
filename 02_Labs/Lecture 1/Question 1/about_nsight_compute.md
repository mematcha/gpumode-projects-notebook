# NVIDIA Nsight Compute Notes

## What is NVIDIA Nsight Compute?

NVIDIA Nsight Compute is an interactive profiler for CUDA that provides detailed performance metrics and API debugging via a user interface and command-line tool. Users can run guided analysis and compare results with a customizable and data-driven user interface, as well as post-process and analyze results in their own workflows.

## What actually happens behind the scenes

During regular execution, a CUDA application process will be launched by the user. It communicates directly with the CUDA user-mode driver, and potentially with the CUDA runtime library.

> The CUDA user-mode driver is the userspace component of the NVIDIA driver stack that implements the [CUDA Driver API](https://modal.com/gpu-glossary/host-software/cuda-driver-api). Typically provided as a shared library (`libcuda.so` on Linux or `nvcuda.dll` on Windows), it acts as the critical bridge between the host application, the CUDA runtime library, and the underlying OS kernel-mode driver. Its core responsibility is providing low-level control over the GPU by handling essential operations like memory allocation, context management, module loading, data transfers, and launching [CUDA kernels](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda).

> When an application is profiled, tools like Nsight Compute rely heavily on intercepting this driver. As detailed in the [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#introduction), the profiler injects its measurement libraries directly into the application process to intercept its communications with the user-mode driver. This interception allows the profiling tool to detect kernel launches and seamlessly collect hardware performance metrics directly from the GPU.

> The [CUDA runtime library](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#introduction) (`libcudart`) is a high-level programming interface built on top of the low-level CUDA Driver API to simplify GPU application development. While an application can communicate directly with the user-mode driver, the runtime library abstracts away verbose device management tasks like explicit context creation, module loading, and device initialization. This allows developers to easily manage GPU memory (using functions like `cudaMalloc`) and launch parallel workloads (using the standard `<<<...>>>` execution syntax) without needing to manually orchestrate the complex underlying hardware interactions.

While host and target are often the same machine, the target can also be a remote system with a potentially different operating system.

When profiling an application with NVIDIA Nsight Compute, the behavior is different. The user launches the NVIDIA Nsight Compute frontend (either the UI or the CLI) on the host system, which in turn starts the actual application as a new process on the target system. While host and target are often the same machine, the target can also be a remote system with a potentially different operating system.

The tool inserts its measurement libraries into the application process, which allow the profiler to intercept communication with the CUDA user-mode driver. In addition, when a kernel launch is detected, the libraries can collect the requested performance metrics from the GPU. The results are then transferred back to the frontend.

## Metric Collection

**Collection of performance metrics is the key feature of NVIDIA Nsight Compute.** Since there is a huge list of metrics available, it is often easier to use some of the tool’s pre-defined [sets or sections](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sets-and-sections) to collect a commonly used subset.-- just refer this link!.

Users are free to adjust which metrics are collected for which kernels as needed, but it is important to keep in mind the [Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead) associated with data collection.

## ncu modes

- `launch-and-attach` *(default):** `ncu` starts your program and profiles it in one command `ncu python square_ncu.py ...`). Use this.
- `launch`*:** `ncu` starts the program and **pauses** it. A GUI or a later `ncu --mode attach` hooks in afterward.
- `attach`*:** the program is **already running**; `ncu` only connects to its PID and profiles from then on.

