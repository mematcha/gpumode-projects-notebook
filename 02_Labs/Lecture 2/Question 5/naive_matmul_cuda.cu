#include <torch/extension.h>
#include <cstdint>

// naive matrix multiplication kernel
__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }

}

// kernel launcher for naive matrix multiplication
torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B, int block_x, int block_y) {
    // Step 1: Implement Checks
    // 1.1: Check if the matrices are on the GPU
    TORCH_CHECK(A.device().is_cuda(), "A must be on the GPU");
    TORCH_CHECK(B.device().is_cuda(), "B must be on the GPU");
    // 1.2: Check if the matrices are float tensors
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be a float tensor");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be a float tensor");
    // 1.3 Make sure the matrices are two dimensional only
    TORCH_CHECK(A.dim() == 2, "A must be a two dimensional tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a two dimensional tensor");
    // 1.4 Check if the matrices are of the appropriate dimensions to allow matrix multiplication
    TORCH_CHECK(A.size(1) == B.size(0), "The number of columns in A must be equal to the number of rows in B");
    // 1.5 Make sure the matrices are contiguous
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    // 1.6 Make sure the matrices are not empty
    TORCH_CHECK(A.size(0) > 0, "A must not be empty");
    TORCH_CHECK(B.size(0) > 0, "B must not be empty");

    // Step 2: Parameters to be Passed to the Kernel
    // 2.1: Get the dimensions of the matrices
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);
    // 2.2: Allocate memory for the result matrix
    torch::Tensor C = torch::zeros({M, K}, A.options());
    // 2.3: Get the device data pointers for the input and result matrices
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    // 2.4: Define the grid and block dimensions
    // Note: K is the number of columns in C , so we divide by the block size to get the number of blocks in the x direction
    // Note: M is the number of rows in C , so we divide by the block size to get the number of blocks in the y direction
    dim3 grid(
        (K+block_x-1)/block_x,
        (M+block_y-1)/block_y
    );
    dim3 block(block_x, block_y);
    // 2.5: Launch the kernel with the grid/block configuration (no shared memory so not third parameter)
    naive_matmul_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    // 2.6: Return the result matrix
    return C;
}
