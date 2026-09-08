#include <torch/extension.h>
#include <cstdint>


// Building kernels for naive and shared memory mean filter
// naive mean filter kernel

/*

In the naive mean filter kernel, we will iterate over the image and compute the mean of the surrounding pixels.

We will use the following parameters:
- @param image_ptr: pointer to the image
- @param output_ptr: pointer to the output
- @param width: width of the image
- @param height: height of the image
- @param radius: radius of the mean filter

An important point to note that even though we are using a 2D image, 
the image is stored at memory as a 1D array, and CUDA kernels operate on 1D arrays instead of tensors.

Now to solve the problem: here are the steps:

1. A single thread corresponds to a single output pixel.
- We need to pick the exact row and col from the block dim2 values to compute the output pixel global index.
2. We need to use the radius around the input index at tid to compute the mean of the surrounding pixels, ensure the mean is within bounds only.
3. In order to do this effectively, we need to get the row and column of the output pixel, 
   and then use the radius to compute the mean of the surrounding input pixels in a loop.
*/

__global__ void naive_mean_filter_kernel(
    const uint8_t* image_ptr, 
    uint8_t* output_ptr, 
    int width, int height, 
    int radius
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = row * width + col;

    int sum = 0;
    int count = 0;

    // ensure the output is within bounds
    if (row <height && col < width){
        // 2D kernel loop to compute the mean of the surrounding pixels
        for (int i = -radius; i <= radius; i++){
            for (int j = -radius; j <= radius; j++){
                int input_row = row + i;
                int input_col = col + j;
                // ensure the input is within bounds only
                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width){
                    int input_index = input_row * width + input_col;
                    sum += image_ptr[input_index];
                    count++;
                }
            }
        }
        // compute the mean of the surrounding pixels
        if (count > 0){
            output_ptr[tid] = sum / count;
        }
        else{
            output_ptr[tid] = 0;
        }
    }
    
}

// shared memory mean filter kernel

/*




*/

__global__ void shared_mean_filter_kernel(
    const uint8_t* image_ptr, 
    uint8_t* output_ptr, 
    int width, int height, 
    int radius
){

    //Step 1: get local thread index
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int block_start_row = blockIdx.y * blockDim.y;
    int block_start_col = blockIdx.x * blockDim.x;
    int local_tid = local_row * blockDim.x + local_col;
    //Step 2: plan for the shared memory
    // we will use the shared memory to store the surrounding pixels of the output pixel
    extern __shared__ uint8_t shared_mem[];
    int shared_mem_size = (blockDim.x+2*radius) * (blockDim.y+2*radius); // e.g. 18*18 for a 16*16 block and radius 1. 
    // loop over to fill the shared memory from the input image
    for (int idx = local_tid; idx<shared_mem_size; idx += blockDim.x * blockDim.y){
        // get the shared row and col from the local thread index
        int shared_row = idx / (blockDim.x+2*radius);
        int shared_col = idx % (blockDim.x+2*radius);
        int shared_index = shared_row * (blockDim.x+2*radius) + shared_col;

        // get global row and col from the shared row and col
        int global_row = block_start_row + shared_row - radius;
        int global_col = block_start_col + shared_col - radius;

        if (global_row >= 0 && global_row < height && global_col >= 0 && global_col < width){
            int image_index = global_row * width + global_col;
            shared_mem[shared_index] = image_ptr[image_index];
        }
        else{
            shared_mem[shared_index] = 0;
        }
        
    }

    __syncthreads();//ensure that all the threads have finished writing to the shared memory before continuing
    
    // Step 3: compute the mean of the surrounding pixels

    int sum = 0;
    int count = 0;

    // Global coordinates of this thread's output pixel
    int out_row = block_start_row + local_row;
    int out_col = block_start_col + local_col;

    // Only valid output threads should compute/write
    if (out_row < height && out_col < width) {

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                // Corresponding global neighbor coordinates
                int global_neighbor_row = out_row + i;
                int global_neighbor_col = out_col + j;

                // Match naive-kernel behavior:
                // only include valid neighbors
                if (global_neighbor_row >= 0 &&
                    global_neighbor_row < height &&
                    global_neighbor_col >= 0 &&
                    global_neighbor_col < width) {

                    // Location of that neighbor inside shared memory
                    int shared_row = local_row + radius + i;
                    int shared_col = local_col + radius + j;

                    int shared_index =
                        shared_row * (blockDim.x + 2 * radius)
                        + shared_col;

                    sum += shared_mem[shared_index];
                    count++;
                }
            }
        }

        int out_tid = out_row * width + out_col;

        if (count > 0) {
            output_ptr[out_tid] = static_cast<uint8_t>(sum / count);
        } else {
            output_ptr[out_tid] = 0;
        }
    }
    
    
    
}    

// Building launchers for naive and shared memory mean filter

/*

Every launcher of a kernel should follow the standard procedure:

- Implement Checks before launching the respective kernels
- Match and also ensure proper parameters are passed to the kernel
- Build the right grid/block configuration
- Launch the kernels
- Synchronize the device
- Copy the result back to the host
- Return the result

In this circumstance, we have 2 kernels to launch: naive_mean_filter_kernel and shared_mean_filter_kernel. Here's our procedure we will strive to follow:
1. Checks to be Implemented
- Check if the image is a 2D tensor
- Check if the radius is a positive integer
- Check if the image is on the GPU
- Check if the dtype of the image is FP32
2. Parameters to be Passed to the Kernel
- Image pointer
- Output pointer
- Width of the image
- Height of the image
- Radius int value
3. Grid/Block Configuration to be Built
- Now we will try to use a 2d block configuration to launch the kernels.
- We will use the width and height of the image to build the grid configuration.
4. Launch the Kernels
- Launch the naive_mean_filter_kernel and shared_mean_filter_kernel with the above parameters.alignas
5. Synchronize the Device
- Synchronize the device after launching the kernels.
6. Return the Result
- Return the result.
*/

// naive mean filter launcher
torch::Tensor naive_mean_filter(torch::Tensor image, int radius){
    // 1. Checks to be Implemented
    TORCH_CHECK(image.dim() == 2, "Image must be a 2D tensor");
    TORCH_CHECK(radius > 0, "Radius must be a positive integer");
    TORCH_CHECK(image.device().is_cuda(), "Image must be on the GPU");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Image must be a FP32 tensor");
    // 2. Parameters to be Passed to the Kernel
    torch::Tensor output = torch::empty_like(image, image.options()); //ensure that the output tensor is on the same device as the image
    const uint8_t* image_ptr = image.data_ptr<uint8_t>();
    uint8_t* output_ptr = output.data_ptr<uint8_t>(); //dataptr ensures that 
    int width = image.size(1);
    int height = image.size(0);
    // 3. Grid/Block Configuration to be Built
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // 4. Launch the Kernels
    naive_mean_filter_kernel<<<grid, block>>>(image_ptr, output_ptr, width, height, radius);
    // 5. Synchronize the Device
    cudaDeviceSynchronize();
    // 6. Return the Result
    return output;
}

// shared memory mean filter launcher
torch::Tensor shared_mean_filter(torch::Tensor image, int radius){
    // 1. Checks to be Implemented
    TORCH_CHECK(image.dim() == 2, "Image must be a 2D tensor");
    TORCH_CHECK(radius > 0, "Radius must be a positive integer");
    TORCH_CHECK(image.device().is_cuda(), "Image must be on the GPU");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "Image must be a FP32 tensor");
    // 2. Parameters to be Passed to the Kernel
    torch::Tensor output = torch::empty_like(image, image.options()); //ensure that the output tensor is on the same device as the image
    const uint8_t* image_ptr = image.data_ptr<uint8_t>();
    uint8_t* output_ptr = output.data_ptr<uint8_t>(); //dataptr ensures that 
    int width = image.size(1);
    int height = image.size(0);
    // 3. Grid/Block Configuration to be Built
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t shared_mem_size = (block.x+2*radius) * (block.y+2*radius) * sizeof(uint8_t);
    // 4. Launch the Kernels
    shared_mean_filter_kernel<<<grid, block, shared_mem_size>>>(image_ptr, output_ptr, width, height, radius);
    // 5. Synchronize the Device
    cudaDeviceSynchronize();
    // 6. Return the Result
    return output;
}