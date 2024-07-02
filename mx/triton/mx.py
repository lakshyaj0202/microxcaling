import torch
import triton.language as tl

from mltools.numerical.format import Format

default_format = Format.from_shorthand("BFP[8|8]{64,-1}(SN)")

import triton
from common import get_grid, get_biased_exponent, get_shared_scale, get_trailing_mantissa, construct_float, get_sign
from quantize import shift_right_round_mantissa, shift_left_mantissa
from quantize import quantize_elemwise

from common import ROUNDING_MODE_INT

import pdb


@triton.jit
def quantize_mx_kernel  (input_ptr, zero_ptr, max_ptr, n,
                        output_ptr,
                        scale_bits, elem_ebits, elem_mbits, elem_max_norm,
                        rounding_mode, flush_fp32_subnorms,
                        BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    shared_exp = get_biased_exponent(tl.load(max_ptr + offsets, mask))
    
    ones_tensor = tl.zeros_like(shared_exp) + 1
    zeros_tensor = tl.zeros_like(shared_exp)    
    check_zeros = ((shared_exp == tl.zeros_like(shared_exp)) & flush_fp32_subnorms)
    flush_tile = tl.where(check_zeros, ones_tensor, zeros_tensor)
    
    # Compute the shared scale
    scale_bits = scale_bits.to(tl.int32)
    max_norm_tensor = tl.zeros_like(shared_exp) + elem_max_norm
    scale = get_shared_scale(shared_exp, scale_bits, max_norm_tensor)

    if flush_fp32_subnorms:
        scaled_input = tl.div_rn(tl.load(input_ptr + offsets, mask), scale)
        scaled_in = tl.where(flush_tile, scaled_input, tl.load(zero_ptr + offsets, mask))
    else:
        scaled_in = tl.load(zero_ptr + offsets, mask)



    # biased_exp = get_biased_exponent(scaled_in)
    # sign = get_sign(scaled_in)
    # tmant = get_trailing_mantissa(scaled_in)
    # is_int = True
    # if is_int:
    #     new_bias = tl.zeros_like(biased_exp) + 1
    # else:
    #     new_bias = tl.zeros_like(biased_exp) + ((1 << (elem_ebits)) - 1)
    # new_biased_exp = biased_exp - 127 + new_bias # biased_exp - FLOAT32_EXP_BIAS + new_bias
    
    # exp_diff = tl.where(new_biased_exp <= 0, 1-new_biased_exp, tl.zeros_like(new_biased_exp))
    # exp_diff = tl.where(exp_diff > 24, tl.zeros_like(exp_diff) + 24, exp_diff)
    # biased_exp_check = biased_exp == 0
    
    # tmant = shift_right_round_mantissa(tmant, biased_exp_check,
    #                                    exp_diff, elem_mbits, rounding_mode,
    #                                    not is_int)
    # overflow = shift_left_mantissa(tmant, exp_diff, biased_exp_check, elem_mbits)
    # biased_exp = tl.where(overflow, biased_exp+1, biased_exp)
    # output = construct_float(sign, biased_exp, tmant)
    # neg_max_norm_tensor = tl.zeros_like(output) - elem_max_norm
    # max_norm_tensor = tl.zeros_like(output) + elem_max_norm
    # biased_exp_tensor = tl.zeros_like(biased_exp) + 0xFF
    # tmant_tensor = tl.zeros_like(tmant)
    # output = tl.where((tl.abs(output) >  max_norm_tensor) & (is_int), 
    #                   tl.where(sign, neg_max_norm_tensor, max_norm_tensor),
    #                   construct_float(sign, biased_exp_tensor, tmant_tensor))
    # scaled_out = tl.where(tmant == 0, tl.zeros_like(output), output)

    scaled_out = quantize_elemwise(scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                                   rounding_mode, True, True)
    scaled_out = scaled_out * scale
    
    # pdb.set_trace()
    tl.store(output_ptr + offsets, scaled_out, mask)
    
def quantize_mx(in_tensor, scale_bits, ebits, mbits, max_norm, max_values, 
                axis, block_size, flush_fp32_subnorms = False, rounding_mode="N"):
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
    quantize_mx_kernel[grid](in_tensor, zero_tensor, max_values, total_size,
                             out_tensor,
                             scale_bits, ebits, mbits, max_norm,
                             rounding_mode, flush_fp32_subnorms,
                             BLOCK_SIZE=BLOCK_SIZE)

    return out_tensor
