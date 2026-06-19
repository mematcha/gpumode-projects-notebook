import torch # for tensor operations
import time # for timing operations
import sys # for command line arguments

def main():
    # if the number of arguments is not 2, print an error message and exit the program
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <vector_size>")
        sys.exit(1)

    # convert the first argument to an integer (argv[0] is actually the name of the program)
    n = int(sys.argv[1])
    # if the integer is less than or equal to 0, print an error message and exit the program
    if n <= 0:
        print("Vector size must be positive.")
        sys.exit(1)

    # Check if CUDA is available and set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU device.")

    # Use CUDA events for accurate timing if on GPU
    if device.type == 'cuda':
        # Synchronize before starting to ensure a clean slate
        torch.cuda.synchronize()
        # Allocate tensors directly on the device
        a = torch.arange(n, dtype=torch.float64, device=device)
        b = torch.arange(n, dtype=torch.float64, device=device)
        c = torch.empty(n, dtype=torch.float64, device=device)

        # --- Timing ---
        # Record start time for host-side measurement of the overall process
        # but timing GPU ops accurately needs cuda events or similar
        
        # Warm-up run to initialize CUDA context and caches
        c.copy_(a + b)
        torch.cuda.synchronize()

        # Start timing the computation block
        torch.cuda.synchronize() # Ensure previous ops are done
        start_op = torch.cuda.Event(enable_timing=True)
        end_op = torch.cuda.Event(enable_timing=True)
        
        start_op.record()
        
        # Perform the addition
        c.copy_(a + b) # using copy_ for assignment to measure this specific op timing

        end_op.record()
        torch.cuda.synchronize() # wait for the operations to complete

        # Calculate times
        host_elapsed_time_sec = (time.time() - start_op.elapsed_time(end_op) / 1000.0) # Crude approximation for host time portion
        gpu_elapsed_time_ms = start_op.elapsed_time(end_op)

        # PyTorch tensor operations on GPU often include data transfer implicitly
        # The 'a + b' operation on GPU tensors is the kernel execution.
        # Copying to/from GPU is handled by moving tensors: tensor.to(device)
        # Here, a, b, c are already on device, so we are timing the computation.
        # It's harder to precisely isolate host<->device copy in a simple PyTorch script
        # like this if tensors are created directly on the device.
        # If we were doing:
        # a_cpu = torch.arange(n, dtype=torch.float64)
        # b_cpu = torch.arange(n, dtype=torch.float64)
        # start_copy = time.time()
        # a_gpu = a_cpu.to(device)
        # b_gpu = b_cpu.to(device)
        # end_copy = time.time()
        # ... then time a_gpu + b_gpu ...
        # This setup aims to time the GPU computation phase primarily.
        
        # print the results
        print(f"PyTorch Vector Addition (Size: {n})")
        print(f"Kernel Execution Time: {gpu_elapsed_time_ms:.3f} ms")
        print(f"(Note: Copy times are implicitly handled by PyTorch and not explicitly measured here for simplicity.)")
        print(f"Total End-to-End Time (approx, includes host overhead): {gpu_elapsed_time_ms:.3f} ms")

    else: # CPU execution
        a = torch.arange(n, dtype=torch.float64, device=device)
        b = torch.arange(n, dtype=torch.float64, device=device)
        
        start_time = time.time()
        c = a + b
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"PyTorch Vector Addition (Size: {n})")
        print(f"Total End-to-End Time: {elapsed * 1000:.3f} ms")

if __name__ == "__main__":
    main()