import torch
import triton.language as tl

from mltools.numerical.format import Format

default_format = Format.from_shorthand("BFP[8|8]{64,-1}(SN)")

import triton
from common import get_grid, get_biased_exponent, get_shared_scale
from quantize import quantize_elemwise

from common import ROUNDING_MODE_INT


@triton.jit
def quantize_mx_kernel  (input_ptr, max_ptr, n,
                        output_ptr,
                        scale_bits, elem_ebits, elem_mbits, elem_max_norm,
                        axis_size, post_axis_size, rounding_mode, flush_fp32_subnorms,
                        BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Computing the index of the max element of current element
    post_axis_i = offsets % post_axis_size
    pre_axis_i = offsets / (post_axis_size * axis_size)
    
    # Get the shared exponent
    m_i = pre_axis_i * post_axis_size + post_axis_i
    shared_exp = get_biased_exponent(tl.load(max_ptr + m_i, mask))  ## triton kernel
    flush_tile = (shared_exp == 0) and flush_fp32_subnorms
    
    # Compute the shared scale
    scale = get_shared_scale(shared_exp, scale_bits, elem_max_norm)

    if flush_tile:
        scaled_in = 0
    else:
        scaled_in = tl.load(input_ptr + offsets, mask) / scale

    scaled_out = quantize_elemwise(scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                                   rounding_mode, True, True)
    scaled_out = scaled_out * scale

    tl.store(output_ptr, scaled_out, mask)
    
def quantize_mx(in_tensor, scale_bits, ebits, mbits, max_norm, max_values, 
                axis, block_size, flush_fp32_subnorms = False, rounding_mode="N"):
    # in_tensor = torch.tensor([[1,2,3,4],[5,6,7,8]], device = "cuda")
    out_tensor = torch.zeros_like(in_tensor, device = in_tensor.device)
    ndim = in_tensor.dim()
    input_size = in_tensor.size()
    axis_size = input_size[axis]
    pre_axis_size=1
    for i in range(axis):
        pre_axis_size *= input_size[i]

    post_axis_size = 1
    for i in range(axis+1, ndim):
        post_axis_size *= input_size[i]

    total_size = pre_axis_size * axis_size * post_axis_size

    grid = get_grid(total_size)

    BLOCK_SIZE = block_size

    rounding_mode = ROUNDING_MODE_INT[rounding_mode]

    # call the triton kernel
    quantize_mx_kernel[grid](in_tensor, max_values, total_size,
                             out_tensor,
                             scale_bits, ebits, mbits, max_norm,
                             axis_size, post_axis_size, rounding_mode, flush_fp32_subnorms,
                             BLOCK_SIZE=BLOCK_SIZE)


