#include <iostream> // for input and output operations
#include <vector> // for vector operations
#include <numeric> // for numeric operations
#include <cstdlib> // for standard library operations
#include <cuda_runtime.h> // for CUDA runtime API

// macro to check for CUDA errors
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                      \
            std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__  \
                      << ", code: " << error                            \
                      << ", reason: " << cudaGetErrorString(error)      \
                      << std::endl;                                     \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// kernel function to add two vectors
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // get the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if the thread index is less than the number of elements, add the corresponding elements

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void run_experiment(int n, int threadsPerBlock) {
    
    // calculate the number of bytes to allocate
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // allocate vectors on the host
    std::vector<float> h_a(n), h_b(n), h_c(n);

    // fill the vectors with the values 0.0, 1.0, 2.0, ..., n-1.0
    //std::iota is a function that fills a container with a sequence of values
    std::iota(h_a.begin(), h_a.end(), 0.0f);
    std::iota(h_b.begin(), h_b.end(), 0.0f);

    // allocate vectors on the device
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    // allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    // define CUDA events
    cudaEvent_t h2d_start, h2d_stop; // host to device start and stop events
    cudaEvent_t kernel_start, kernel_stop; // kernel execution
    cudaEvent_t d2h_start, d2h_stop; // device to host
    // create CUDA events
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    // define variables to store the time taken for each operation
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;

    // record the time for the host to device copy
    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

    // define the number of threads per block and the number of blocks per grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "n: " << n << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
    std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
    // record the time for the kernel execution
    CUDA_CHECK(cudaEventRecord(kernel_start));
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError()); // check for errors
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

    // record the time for the device to host copy
    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

    // calculate the total time taken for the operation
    float total_ms = h2d_ms + kernel_ms + d2h_ms;

    // print the results
    std::cout << "CUDA Vector Addition (Size: " << n << ")" << std::endl;
    std::cout << "Host-to-Device Copy Time: " << h2d_ms << " ms" << std::endl;
    std::cout << "Kernel Execution Time:    " << kernel_ms << " ms" << std::endl;
    std::cout << "Device-to-Host Copy Time: " << d2h_ms << " ms" << std::endl;
    std::cout << "Total End-to-End Time:    " << total_ms << " ms" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << std::endl;
    // free the memory on the device
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // destroy the CUDA events
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));
    
}

// main function; takes in integer input from command line (argc), and gets a character array also as input (argv)
int main(int argc, char** argv) {
    // if the number of arguments is not 2, print an error message and return 1
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <vector_size> <threads_per_block>" << std::endl;
        return 1;
    }

    // convert the first argument to an integer (argv[0] is actually the name of the program)
    int n = std::stoi(argv[1]);
    int threadsPerBlock = std::stoi(argv[2]);
    // if the integer is less than or equal to 0, print an error message and return 1
    if (n <= 0) {
        std::cerr << "Vector size must be positive." << std::endl;
        return 1;
    }
    if (threadsPerBlock <= 0) {
        std::cerr << "Threads per block must be positive." << std::endl;
        return 1;
    }


    run_experiment(n, threadsPerBlock);

    return 0;
}