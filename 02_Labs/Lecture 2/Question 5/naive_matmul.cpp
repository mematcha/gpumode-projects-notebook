#include <torch/extension.h>

//naive matrix multiplication prototype with signature
torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B, int block_x, int block_y);
