import torch
import triton.language as tl

from mltools.numerical.format import Format

default_format = Format.from_shorthand("BFP[8|8]{64,-1}(SN)")

import triton
from common import get_grid, get_biased_exponent, get_shared_scale
from quantize import quantize_elemwise

from common import ROUNDING_MODE_INT


@triton.jit
def quantize_mx_kernel  (input_ptr, zero_ptr, max_ptr, n, W, H,
                        output_ptr,
                        scale_bits, elem_ebits, elem_mbits, elem_max_norm,
                        axis_size, post_axis_size, rounding_mode, flush_fp32_subnorms,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    shared_exp = get_biased_exponent(tl.load(max_ptr + offsets, mask))
    # tl.device_print("shared_exp", shared_exp)
    # ones_tensor = tl.zeros_like(shared_exp) + 1
    # zeros_tensor = tl.zeros_like(shared_exp)    
    # check_zeros = ((shared_exp == tl.zeros_like(shared_exp)) & flush_fp32_subnorms)
    # flush_tile = tl.where(check_zeros, ones_tensor, zeros_tensor)
    
    # # Compute the shared scale
    # scale_bits = scale_bits.to(tl.int32)
    # scale = get_shared_scale(shared_exp, scale_bits, elem_max_norm)
    # tl.static_print(shared_exp)

    # if flush_fp32_subnorms:
    #     scaled_block_input = tl.div_rn(tl.load(input_ptr + offsets, mask), scale)
    #     scaled_in = tl.where(flush_tile, scaled_block_input, tl.load(zero_ptr + offsets, mask))
    # else:
    #     scaled_in = tl.load(zero_ptr + offsets, mask)

    # scaled_out = quantize_elemwise(scaled_in, elem_mbits, elem_ebits, elem_max_norm,
    #                                rounding_mode, True, True)
    # scaled_out = scaled_out * scale
    
    # tl.store(output_ptr+offsets, scaled_out, mask) # .to(output_ptr.type.element_ty))
    # tl.store(output_ptr, scaled_out, mask)
    tl.device_print("offsets", offsets)
    tl.store(output_ptr + offsets, shared_exp, mask)
    
def quantize_mx(in_tensor, scale_bits, ebits, mbits, max_norm, max_values, 
                axis, block_size, flush_fp32_subnorms = False, rounding_mode="N"):
    # in_tensor = torch.tensor([[1,2,3,4],[5,6,7,8]], device = "cuda")
    out_tensor = torch.empty_like(in_tensor, device = in_tensor.device)
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

    zero_tensor = torch.zeros_like(in_tensor)

    tensor_shape = list(in_tensor.shape)
    W = tensor_shape[0]
    H = tensor_shape[1]

    in_tensor = in_tensor.contiguous()
    zero_tensor = zero_tensor.contiguous()
    max_values = max_values.contiguous()
    out_tensor = out_tensor.contiguous()

    # call the triton kernel
    quantize_mx_kernel[grid](in_tensor, zero_tensor, max_values, total_size, W, H,
                             out_tensor,
                             scale_bits, ebits, mbits, max_norm,
                             axis_size, post_axis_size, rounding_mode, flush_fp32_subnorms,
                             BLOCK_SIZE=BLOCK_SIZE)


