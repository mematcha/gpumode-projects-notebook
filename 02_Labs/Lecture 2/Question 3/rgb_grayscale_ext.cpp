#include <torch/extension.h>

void launch_rgb_grayscale_hwc(
    const float* input,
    float* output,
    int height,
    int width,
    int threads_per_block);

void launch_rgb_grayscale_chw(
    const float* input,
    float* output,
    int height,
    int width,
    int threads_per_block);

namespace {

void check_float32_cuda_contiguous(const torch::Tensor& input, const char* expected_shape_msg) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "expected float32 input");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, expected_shape_msg);
}

torch::Tensor rgb_to_grayscale_cuda_hwc(torch::Tensor input, int64_t threads_per_block) {
    check_float32_cuda_contiguous(input, "expected shape (H, W, 3)");
    TORCH_CHECK(input.size(2) == 3, "expected shape (H, W, 3)");

    const int64_t height = input.size(0);
    const int64_t width = input.size(1);
    TORCH_CHECK(height > 0 && width > 0, "H and W must be positive");
    TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024, "threads_per_block must be in (0, 1024]");

    auto output = torch::empty({height, width}, input.options());
    launch_rgb_grayscale_hwc(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(threads_per_block));
    return output;
}

torch::Tensor rgb_to_grayscale_cuda_chw(torch::Tensor input, int64_t threads_per_block) {
    check_float32_cuda_contiguous(input, "expected shape (3, H, W)");
    TORCH_CHECK(input.size(0) == 3, "expected shape (3, H, W)");

    const int64_t height = input.size(1);
    const int64_t width = input.size(2);
    TORCH_CHECK(height > 0 && width > 0, "H and W must be positive");
    TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024, "threads_per_block must be in (0, 1024]");

    auto output = torch::empty({height, width}, input.options());
    launch_rgb_grayscale_chw(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(threads_per_block));
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rgb_to_grayscale_hwc",
        &rgb_to_grayscale_cuda_hwc,
        "Convert RGB image (H, W, 3) to grayscale on CUDA",
        pybind11::arg("input"),
        pybind11::arg("threads_per_block") = 256);
    m.def(
        "rgb_to_grayscale_chw",
        &rgb_to_grayscale_cuda_chw,
        "Convert RGB image (3, H, W) to grayscale on CUDA",
        pybind11::arg("input"),
        pybind11::arg("threads_per_block") = 256);
}
