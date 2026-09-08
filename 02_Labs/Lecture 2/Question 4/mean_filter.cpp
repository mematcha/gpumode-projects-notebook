#include <torch/extension.h>

// naive mean filter prototype with signature
torch::Tensor naive_mean_filter(torch::Tensor image, int radius);
// shared memory mean filter prototype with signature
torch::Tensor shared_mean_filter(torch::Tensor image, int radius);