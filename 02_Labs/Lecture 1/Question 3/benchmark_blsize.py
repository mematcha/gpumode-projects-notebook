# import libraries
import sys
import torch # for tensor operations
import triton # for triton operations
import triton.language as tl # for triton language
import matplotlib.pyplot as plt # for plotting

BLOCK_SIZE= int(sys.argv[1])
N = 2**30
x = torch.randn(N, device='cuda')
y = torch.empty_like(x)
# define the kernel
@triton.jit
def triton_square(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # get the block index
    program_id = tl.program_id(0)
    # offset
    offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask
    mask = offset < n
    # load the input
    x = tl.load(x_ptr + offset, mask=mask)
    # square the input
    y = x * x
    # store the output
    tl.store(y_ptr + offset, y, mask=mask)

grid= (triton.cdiv(N, BLOCK_SIZE),)

for _ in range(10):
    triton_square[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
triton_square[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
end.record()
torch.cuda.synchronize()