## Shared Memory Mean Filter

### Goal

Implement a 2D mean filter using CUDA shared memory so that neighboring output threads can reuse overlapping input pixels instead of repeatedly reading the same values from global memory.

---

## 1. Kernel Inputs

The kernel receives:

- `image_data_ptr`: pointer to the input image stored in contiguous GPU memory.
- `output_data_ptr`: pointer to the output image stored in contiguous GPU memory.
- `width`: width of the input image.
- `height`: height of the input image.
- `radius`: radius of the mean filter.

For a radius `r`, each output pixel uses a neighborhood of:


(2r+1) \times (2r+1)


For example:

- `r = 1` → `3×3`
- `r = 2` → `5×5`

---



## 2. Main Policies



### One thread = one output pixel

Each CUDA thread is responsible for computing exactly one output pixel.

For a 2D CUDA launch:

```cpp
int out_row = blockIdx.y * blockDim.y + threadIdx.y;
int out_col = blockIdx.x * blockDim.x + threadIdx.x;
```

Remember:

```text
y → row
x → column
```

The reason is:

- moving along `y` changes which row we are in,
- moving along `x` changes which column we are in.

Therefore:

```text
array[row][col]
array[y][x]
```

---



## 3. Why Shared Memory Needs a Halo

A `16×16` block computes `16×16 = 256` output pixels.

However, each output pixel requires neighboring input pixels.

For radius `r`, the block therefore needs an input tile of size:

$$
(blockDim.x + 2r)
\times
(blockDim.y + 2r)
$$

For example:

```text
block = 16×16
radius = 1

shared tile = 18×18
```

The extra pixels around the block are called the **halo**.

Conceptually:

```text
+--------------------+
|       halo         |
|   +------------+   |
|   |            |   |
|   | 16×16      |   |
|   | outputs    |   |
|   |            |   |
|   +------------+   |
|       halo         |
+--------------------+
```

---



## 4. Why Dynamic Shared Memory Is Used

The shared-memory size depends on `radius`, which is a runtime kernel parameter.

Therefore, we cannot write:

```cpp
__shared__ uint8_t shared_mem[18 * 18];
```

unless the radius is permanently fixed.

Instead we use:

```cpp
extern __shared__ uint8_t shared_mem[];
```

This means:

> the shared-memory array exists, but its size will be provided when the kernel is launched.

The launcher supplies the size using:

```cpp
kernel<<<grid, block, shared_mem_bytes>>>(...);
```

where:

```cpp
shared_mem_bytes =
    (blockDim.x + 2 * radius) *
    (blockDim.y + 2 * radius) *
    sizeof(uint8_t);
```

Each block receives its own independent shared-memory tile.

---



## 5. Identify the Block and Thread Coordinates

The top-left output coordinate handled by a block is:

```cpp
int block_start_row = blockIdx.y * blockDim.y;
int block_start_col = blockIdx.x * blockDim.x;
```

The current thread's output coordinate is:

```cpp
int out_row = block_start_row + threadIdx.y;
int out_col = block_start_col + threadIdx.x;
```

The thread's coordinates **inside the block** are simply:

```cpp
int local_row = threadIdx.y;
int local_col = threadIdx.x;
```

To flatten the thread coordinates into one local thread ID:

```cpp
int local_tid =
    threadIdx.y * blockDim.x + threadIdx.x;
```

Important:

```text
local_tid must contain ONLY threadIdx information.

It must NOT contain blockIdx.
```

For a `16×16` block:

```text
local_tid ranges from 0 to 255
```

inside every block.

---



## 6. Determine Shared-Memory Size

The number of elements in the shared-memory tile is:

```cpp
int shared_width  = blockDim.x + 2 * radius;
int shared_height = blockDim.y + 2 * radius;

int shared_mem_size =
    shared_width * shared_height;
```

For:

```text
block = 16×16
radius = 1
```

we get:

```text
shared_width  = 18
shared_height = 18

shared_mem_size = 324
```

---



## 7. Cooperatively Load Shared Memory

There are only:

```text
256 threads
```

but the shared tile contains:

```text
324 values
```

Therefore, some threads must load more than one value.

We use:

```cpp
for (
    int idx = local_tid;
    idx < shared_mem_size;
    idx += blockDim.x * blockDim.y
)
```

This distributes the work across all threads.

For example:

```text
threads 0..255
→ load shared elements 0..255

threads 0..67
→ loop again
→ load shared elements 256..323
```

This allows 256 threads to cooperatively load all 324 shared-memory elements.

---



## 8. Convert Flattened Shared Index to Row and Column

For each shared-memory index:

```cpp
int shared_row = idx / shared_width;
int shared_col = idx % shared_width;
```

Then:

```cpp
int shared_index =
    shared_row * shared_width + shared_col;
```

Since `idx` is already the flattened shared index, technically:

```cpp
shared_index == idx
```

but writing the explicit formula makes the mapping easier to understand.

---



## 9. Map Shared-Memory Coordinates to Global Image Coordinates

The shared-memory tile includes a halo.

Therefore:

```text
shared coordinate (radius, radius)
```

corresponds to:

```text
the top-left output pixel of the block
```

So the global image coordinate represented by a shared-memory cell is:

```cpp
int global_row =
    block_start_row + shared_row - radius;

int global_col =
    block_start_col + shared_col - radius;
```

The `- radius` shifts the shared-memory halo back into global-image coordinates.

Example for radius `1`:

```text
shared_row = 0
→ global row = block_start_row - 1

shared_row = 1
→ global row = block_start_row

shared_row = 2
→ global row = block_start_row + 1
```

---



## 10. Bounds Check During Shared-Memory Loading

At image boundaries, some halo locations correspond to invalid global coordinates.

For example, for the top-left block:

```text
shared (0,0)
→ global (-1,-1)
```

The shared-memory location itself is valid, but the corresponding global image location is not.

Therefore:

```cpp
if (
    global_row >= 0 &&
    global_row < height &&
    global_col >= 0 &&
    global_col < width
) {

    int global_index =
        global_row * width + global_col;

    shared_mem[shared_index] =
        image_data_ptr[global_index];

} else {

    shared_mem[shared_index] = 0;
}
```

Important:

> The shared-memory location is not out of bounds.
> The global image coordinate it represents is out of bounds.

So we still initialize the shared-memory cell, typically to `0`.

---



## 11. Synchronize the Block

After loading the shared tile:

```cpp
__syncthreads();
```

This ensures that every thread in the block has finished writing its assigned shared-memory values before any thread begins reading from shared memory.

Important rule:

> Every thread in the block must reach this `__syncthreads()`.

Therefore, do NOT place an output bounds check such as:

```cpp
if (out_row < height && out_col < width)
```

around the shared-memory loading phase or around `__syncthreads()`.

Even threads that correspond to invalid output pixels in the final partial block must still participate in loading shared memory and synchronization.

---



## 12. Compute the Mean From Shared Memory

Only after synchronization do we restrict computation to valid output threads:

```cpp
if (out_row < height && out_col < width) {
```

Each thread computes its neighborhood:

```cpp
for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
```

The global coordinates of the neighbor are:

```cpp
int global_neighbor_row = out_row + i;
int global_neighbor_col = out_col + j;
```

We check these global coordinates because our policy is:

> average only valid image neighbors.

So:

```cpp
if (
    global_neighbor_row >= 0 &&
    global_neighbor_row < height &&
    global_neighbor_col >= 0 &&
    global_neighbor_col < width
)
```

Then we determine where that same neighbor exists inside shared memory.

The current thread's center pixel is located at:

```cpp
shared center row = threadIdx.y + radius
shared center col = threadIdx.x + radius
```

Therefore a neighbor at offset `(i,j)` is located at:

```cpp
int shared_row =
    threadIdx.y + radius + i;

int shared_col =
    threadIdx.x + radius + j;
```

Flatten it:

```cpp
int shared_index =
    shared_row * shared_width + shared_col;
```

Then:

```cpp
sum += shared_mem[shared_index];
count++;
```

---



## 13. Why `+ radius` Is Needed During Computation

The shared-memory tile contains a halo before the actual block data.

For radius `1`:

```text
shared row 0 → upper halo
shared row 1 → first real block row
shared row 2 → second real block row
...
```

Therefore:

```text
threadIdx.y = 0
```

does not correspond to:

```text
shared_row = 0
```

Its center is:

```text
shared_row = radius
```

Hence:

```cpp
shared_row = threadIdx.y + radius + i;
```

---



## 14. Write the Output

The flattened output index is:

```cpp
int out_tid =
    out_row * width + out_col;
```

Then:

```cpp
if (count > 0) {
    output_data_ptr[out_tid] = sum / count;
} else {
    output_data_ptr[out_tid] = 0;
}
```

For a valid output thread, `count` should normally be greater than zero because the center pixel itself is valid.

---



## Final Mental Model

The shared-memory mean-filter kernel has two major phases:

### Phase 1: Global Memory → Shared Memory

```text
Block + halo
      ↓
cooperative load
      ↓
global coordinates checked
      ↓
shared-memory tile filled
      ↓
__syncthreads()
```

Mapping used:

```text
global =
block_start
+ shared_coordinate
- radius
```



### Phase 2: Shared Memory → Output

```text
one thread
      ↓
one output pixel
      ↓
iterate neighborhood
      ↓
read reused values from shared memory
      ↓
compute mean
      ↓
write output
```

Mapping used:

```text
shared neighbor =
threadIdx
+ radius
+ neighborhood offset
```

The core idea is:

> Global memory is accessed once cooperatively to fill the block's tile and halo.
> Neighboring threads then reuse those values from much faster shared memory instead of repeatedly fetching overlapping neighborhoods from global memory.

```

One especially important correction from your original notes: this formula was **not** a local thread ID:

```cpp
(blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x
+ (blockIdx.x * blockDim.x + threadIdx.x)
```

Your real local thread ID is only:

```cpp
threadIdx.y * blockDim.x + threadIdx.x;
```

That distinction matters a lot for the cooperative shared-memory loading loop.