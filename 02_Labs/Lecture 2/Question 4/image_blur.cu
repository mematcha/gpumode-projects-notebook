#include <cuda_runtime.h>

__global__ void image_blur_naive_kernel(float* input, float* output, int height, int width, int radius, int threads_per_block) {
    // TODO: Implement the naive kernel
    // Step 1: Load the input tile
    // Step 2: Blur the tile
    // Step 3: Write the tile back to output
       
}

__global__ void image_blur_tiled_kernel(float* input, float* output, int height, int width, int radius, int threads_per_block, int tile_size) {
    // TODO: Implement the tiled kernel
    // Step 1: Load the input tile and halo region into shared memory
    // Step 2: Blur the tile from shared memory
    // Step 3: Update the output in loop over output tile
    // Step 4: Write the output tile back to global memory
}

// launchers
void launch_image_blur_naive(const float* input, float* output, int height, int width, int radius, int threads_per_block) {
    int num_pixels = height * width;
    image_blur_naive_kernel<<<blocks, threads_per_block>>>(input, output, height, width, radius, threads_per_block);
}

void launch_image_blur_tiled(const float* input, float* output, int height, int width, int radius, int threads_per_block, int tile_size) {
    // compute the number of blocks and threads per block
    int num_pixels = height * width;
    image_blur_tiled_kernel<<<blocks, threads_per_block>>>(input, output, height, width, radius, threads_per_block, tile_size);
}