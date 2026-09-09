### Multidimensional CUDA Blocks and Grids

CUDA lets you organize **threads inside a block** and **blocks inside a grid** in 1D, 2D, or 3D using `dim3`.

For a 2D image kernel, a common choice is:

```cpp
dim3 block(16, 16);
```

This means:

```text
blockDim.x = 16
blockDim.y = 16
total threads/block = 16 × 16 = 256
```

Then the grid is sized to cover the image:

```cpp
dim3 grid(
    (width  + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
```

and the kernel launch becomes:

```cpp
kernel<<<grid, block>>>(...);
```

Inside the kernel:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

So:

```text
threadIdx.x → position inside block along width
threadIdx.y → position inside block along height

blockIdx.x  → which block along width
blockIdx.y  → which block along height
```

For 3D, you can write:

```cpp
dim3 block(8, 8, 4);
dim3 grid(gx, gy, gz);
```

and use:

```cpp
threadIdx.x
threadIdx.y
threadIdx.z

blockIdx.x
blockIdx.y
blockIdx.z
```

The total number of threads in a block is always:

$$
blockDim.x \times blockDim.y \times blockDim.z
$$

and must stay within the GPU's maximum threads-per-block limit, commonly 1024.

You can also still use a 1D launch for a 2D problem:

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;

int row = tid / width;
int col = tid % width;
```

Both approaches are valid. For image operations such as blur, convolution, and stencils, **2D blocks are usually easier to reason about**.

### Triton equivalent

Triton is a little different. You normally do not explicitly define CUDA-style `dim3` thread blocks. Instead, you define a **program grid**, and each Triton program processes a block/tile of data.

For example:

```python
grid = (triton.cdiv(width, BLOCK_W),
        triton.cdiv(height, BLOCK_H))
```

Inside the Triton kernel:

```python
pid_x = tl.program_id(0)
pid_y = tl.program_id(1)
```

Then a program handles a tile such as:

```text
BLOCK_H × BLOCK_W
```

Conceptually:

```text
CUDA:
grid → blocks → threads

Triton:
grid → program instances → vectorized/tiled work
```

So `tl.program_id()` is roughly analogous to selecting a CUDA block, while Triton itself handles the lower-level mapping of work onto GPU threads and warps.