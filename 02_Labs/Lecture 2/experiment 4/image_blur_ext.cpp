#include <torch/extension.h>

void launch_image_blur_naive(
    const float* input,
    float* output,
    int height,
    int width,
    int radius,
    int threads_per_block);

void launch_image_blur_tiled(
    const float* input,
    float* output,
    int height,
    int width,
    int radius,
    int threads_per_block,
    int tile_size);

namespace {
    void check_blur_input(const torch::Tensor& input, const char* expected_shape_msg) {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "expected float32 input");
        TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
        TORCH_CHECK(input.dim() == 2, expected_shape_msg);
    }

    torch::Tensor image_blur_cuda_naive(torch::Tensor input, int radius, int threads_per_block) {
        check_blur_input(input, "expected shape (H, W)");
        TORCH_CHECK(radius > 0, "radius must be positive");
        TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024, "threads_per_block must be in (0, 1024]");

        // get the height and width
        const int height = input.size(0);
        const int width = input.size(1);
        TORCH_CHECK(height > 0 && width > 0, "H and W must be positive");
        

        // create the output tensor
        auto output = torch::empty({height, width}, input.options());

        // call the launcher function
        launch_image_blur_naive(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            height,
            width,
            radius,
            threads_per_block);
        return output; // return the output tensor
    }
    
    torch::Tensor image_blur_cuda_tiled(torch::Tensor input, int radius, int threads_per_block, int tile_size) {
        check_blur_input(input, "expected shape (H, W)");
        TORCH_CHECK(radius > 0, "radius must be positive");
        TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024, "threads_per_block must be in (0, 1024]");
        TORCH_CHECK(tile_size > 0 && tile_size <= 1024, "tile_size must be in (0, 1024]");

        // get the height and width
        const int height = input.size(0);
        const int width = input.size(1);
        TORCH_CHECK(height > 0 && width > 0, "H and W must be positive");


        // create the output tensor
        auto output = torch::empty({height, width}, input.options());

        // call the launcher function
        launch_image_blur_tiled(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            height,
            width,
            radius,
            threads_per_block,
            tile_size);
        
        return output; // return the output tensor
    }   
} // namespace

PYBIND11_MODULE(){
    m.def(),
    m.def()
}