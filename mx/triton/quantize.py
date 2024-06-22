import torch
import triton

import triton.language as tl
from common import get_biased_exponent, get_sign, get_trailing_mantissa, construct_float

FLOAT32_EXP_BIAS = 127
FLOAT32_TRAILING_MBITS = 23
FLOAT32_FULL_MBITS = (FLOAT32_TRAILING_MBITS + 1)
FLOAT32_IMPLIED1 = (1 << FLOAT32_TRAILING_MBITS)

@triton.jit
def shift_right_round_mantissa(mantissa: tl.tensor, is_subnorm,
                               mbits, exp_diff,
                               rounding_mode, allow_overflow):
    if not is_subnorm:
        mantissa = mantissa + FLOAT32_IMPLIED1
        fp32_sig_bits = 24
    else:
        fp32_sig_bits = 23

    if rounding_mode == "nearest":
        tbits = exp_diff + (fp32_sig_bits - mbits) # number of bits that will be removed
        mask = (1 << (tbits - 1)) - 1
        tie = ~(mantissa & mask)
        mask = 1 << tbits
        even = ~(mantissa & mask)

    mantissa = mantissa >> exp_diff
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1)
    if rounding_mode == "nearest" and (allow_overflow or mantissa != ((1 << (mbits+1)) - 1)):
        if not (tie and even):
            mantissa = mantissa + 1
    
    mantissa = mantissa >> 1

    return mantissa.to(tl.int32)
    

@triton.jit
def shift_left_mantissa(mantissa: tl.tensor, is_subnorm, mbits, exp_diff):
    if is_subnorm:
        fp32_sig_bits = 23
    else:
        fp32_sig_bits = 24

    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff)
    # Handle overflow - don't shift when subnorm overflows into a normal
    overflow = mantissa >= (1 << fp32_sig_bits)
    if overflow and not is_subnorm:
        mantissa =  mantissa >> 1
    mantissa = mantissa & (FLOAT32_IMPLIED1 - 1)
    return overflow


@triton.jit
def quantize_elemwise(input:tl.tensor, bits, exp_bits, max_norm, 
                      rounding_mode, saturate_normals, allow_denorm):
    biased_exp = get_biased_exponent(input)
    sign = get_sign(input)
    tmant = get_trailing_mantissa(input)

    # Mantissa bits to quantize to
    mbits = bits - 1
    is_int = exp_bits == 0

    # Integers can be treated as having exp bias of 1 ??
    if is_int:
        new_bias =1
    else:
        new_bias = (1 << (exp_bits)) - 1
    new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias

    # Skip denorms
    if ((not is_int) and (not allow_denorm) and (new_biased_exp < 1)):
        return 0.0
    
    # Use exp_diff to truncate additional bits for subnorms
    # mbits includes implicit 1, so when new_biased_exp==0
    # we want exp_diff = 1 to truncate away 1 bit

    if new_biased_exp <=0:
        exp_diff = 1-new_biased_exp
    else:
        exp_diff = 0
    
    if exp_diff > FLOAT32_FULL_MBITS:
        exp_diff = FLOAT32_FULL_MBITS

    tmant = shift_right_round_mantissa(tmant, biased_exp==0,
                                       mbits, exp_diff, rounding_mode,
                                       not is_int)
    
    if tmant == 0:
        return 0.0
    
    overflow = shift_left_mantissa(tmant, biased_exp==0, mbits, exp_diff)
    if overflow:
        biased_exp = biased_exp + 1
    
    # reconstruct the float number
    output = construct_float(sign, biased_exp, tmant)

    if (tl.abs(output) >  max_norm):
        if is_int or saturate_normals:
            if sign:
                output = -max_norm
            else:
                output = max_norm
        else:
            output = construct_float(sign, 0xFF, 0)

    return output
    