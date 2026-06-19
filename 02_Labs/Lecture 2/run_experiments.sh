#!/bin/bash
set -e

echo "Conducting Experiment 1: To compare different Execution methods for vector addition"
echo "----------------------------------------"

# Define vector sizes
VECTOR_SIZES=(1000 10000 100000 1000000 10000000 100000000) # 10^3 to 10^8

# Output file
OUTPUT_FILE="experiment_results.csv"

# Compile the C++ and CUDA code (if not already compiled)
echo "Compiling C++ and CUDA code..."
g++ vector_add_cpu.cpp -o vector_add_cpu -std=c++11 -O3 -Wall
nvcc vector_add_cuda.cu -o vector_add_cuda -std=c++11 -O3 -Xcompiler -Wall

echo "Running experiments..."
echo "Vector_Size,CPU_Total_ms,CUDA_Copy_H2D_ms,CUDA_Kernel_ms,CUDA_Copy_D2H_ms,CUDA_Total_ms,PyTorch_Kernel_ms" > "$OUTPUT_FILE"

for n in "${VECTOR_SIZES[@]}"; do
    echo "--- Running for Vector Size: $n ---"

    # Run CPU C++
    # Note: C++ times are in seconds, convert to ms
    cpu_time_sec=$(./vector_add_cpu $n | grep "Total End-to-End Time:" | awk '{print $(NF-1)}')
    cpu_time_ms=$(awk "BEGIN {printf \"%.3f\", $cpu_time_sec * 1000}")
    echo "  CPU: ${cpu_time_ms} ms"

    # Run CUDA
    # CUDA times are already in ms
    cuda_times=$(./vector_add_cuda $n | grep -E "Copy Time:|Kernel Execution Time:|Device-to-Host Copy Time:|Total End-to-End Time:" | awk '{print $(NF-1)}')
    cuda_copy_h2d_ms=$(echo "$cuda_times" | sed -n 1p)
    cuda_kernel_ms=$(echo "$cuda_times" | sed -n 2p)
    cuda_copy_d2h_ms=$(echo "$cuda_times" | sed -n 3p)
    cuda_total_ms=$(echo "$cuda_times" | sed -n 4p)
    echo "  CUDA:"
    echo "    H2D Copy: ${cuda_copy_h2d_ms} ms"
    echo "    Kernel:   ${cuda_kernel_ms} ms"
    echo "    D2H Copy: ${cuda_copy_d2h_ms} ms"
    echo "    Total:    ${cuda_total_ms} ms"


    # Run PyTorch
    # PyTorch times are in ms
    pytorch_times=$(python vector_add_pytorch.py $n | grep -E "Kernel Execution Time:|Total End-to-End Time" | awk '{print $(NF-1)}')
    pytorch_kernel_ms=$(echo "$pytorch_times" | sed -n 1p)
    pytorch_total_ms=$(echo "$pytorch_times" | sed -n 2p)
    echo "  PyTorch (GPU Kernel): ${pytorch_kernel_ms} ms"

    # Append results to CSV
    echo "$n,${cpu_time_ms},${cuda_copy_h2d_ms},${cuda_kernel_ms},${cuda_copy_d2h_ms},${cuda_total_ms},${pytorch_kernel_ms}" >> "$OUTPUT_FILE"
done

echo "Experiments finished. Results saved to $OUTPUT_FILE"
echo "You can view the results by running: cat $OUTPUT_FILE"

echo "----------------------------------------"

echo "Conducting Experiment 2: To compare CUDA Kernel over different block sizes"
echo "----------------------------------------"

BLOCK_SIZES=(32 64 128 256 512 1024)

BLOCK_OBS_FILE="block_observations.csv"

nvcc vector_add_var_block_cuda.cu -o vector_add_var_block_cuda -std=c++11 -O3 -Xcompiler -Wall

echo "Running experiments..."

echo "Vector_Size,Block_Size,Blocks_Per_Grid,CUDA_Copy_H2D_ms,CUDA_Kernel_ms,CUDA_Copy_D2H_ms,CUDA_Total_ms" > "$BLOCK_OBS_FILE"

for n in "${VECTOR_SIZES[@]}"; do
    echo "--- Vector Size: $n ---"

    for block_size in "${BLOCK_SIZES[@]}"; do
        echo "  Running block size: $block_size"

        output=$(./vector_add_var_block_cuda "$n" "$block_size")

        blocks_per_grid=$(echo "$output" | grep "blocksPerGrid:" | awk '{print $NF}')

        h2d_ms=$(echo "$output" | grep "Host-to-Device Copy Time:" | awk '{print $(NF-1)}')
        kernel_ms=$(echo "$output" | grep "Kernel Execution Time:" | awk '{print $(NF-1)}')
        d2h_ms=$(echo "$output" | grep "Device-to-Host Copy Time:" | awk '{print $(NF-1)}')
        total_ms=$(echo "$output" | grep "Total End-to-End Time:" | awk '{print $(NF-1)}')

        echo "    Blocks/Grid: $blocks_per_grid"
        echo "    Kernel:      ${kernel_ms} ms"
        echo "    Total:       ${total_ms} ms"

        echo "$n,$block_size,$blocks_per_grid,$h2d_ms,$kernel_ms,$d2h_ms,$total_ms" >> "$BLOCK_OBS_FILE"
    done
done

echo "Experiment 2 finished. Results saved to $BLOCK_OBS_FILE"
echo "You can view the results by running: cat $BLOCK_OBS_FILE"