#include <cuda_runtime.h>

// ITU-R BT.601 luma weights (matches PyTorch rgb_to_grayscale)
__global__ void rgbGrayscaleKernelHWC(const float* input, float* output, int num_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pixels) {
        int base = tid * 3;
        float r = input[base + 0];
        float g = input[base + 1];
        float b = input[base + 2];
        output[tid] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

__global__ void rgbGrayscaleKernelCHW(const float* input, float* output, int num_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pixels) {
        float r = input[tid];
        float g = input[num_pixels + tid];
        float b = input[2 * num_pixels + tid];
        output[tid] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void launch_rgb_grayscale_hwc(
    const float* input,
    float* output,
    int height,
    int width,
    int threads_per_block) {
    int num_pixels = height * width;
    int blocks = (num_pixels + threads_per_block - 1) / threads_per_block;
    rgbGrayscaleKernelHWC<<<blocks, threads_per_block>>>(input, output, num_pixels);
}

void launch_rgb_grayscale_chw(
    const float* input,
    float* output,
    int height,
    int width,
    int threads_per_block) {
    int num_pixels = height * width;
    int blocks = (num_pixels + threads_per_block - 1) / threads_per_block;
    rgbGrayscaleKernelCHW<<<blocks, threads_per_block>>>(input, output, num_pixels);
}
