import torch
import triton.language as tl
import triton

FLOAT32_EXP_BIAS = 127
FLOAT32_EXP_MAX = 255
FLOAT32_EXP_OFFSET = 23
FLOAT32_SIGN_OFFSET = 31
FLOAT32_EXP_MASK = 0x7f800000
FLOAT32_MANTISSA_MASK = 0x007fffff
FLOAT32_TRAILING_MBITS = 23
FLOAT32_IMPLIED1 = (1 << FLOAT32_TRAILING_MBITS)
FLOAT32_MIN_NORMAL = 2 ** (-FLOAT32_EXP_BIAS + 1)

ROUNDING_MODE_INT = dict(
    {
        "U":0, "up":0,
        "D":1, "down":1,
        "N":2, "nearest":2,
        "S":3, "stochastic":3,
    }
)

import struct

@triton.jit
def float_to_int_preserve_bits(float_number):
    # Pack the float as a 64-bit double (assuming double precision float)
    packed = struct.pack('f', float_number)
    # Unpack the 64-bit double as a signed integer
    int_number = struct.unpack('I', packed)[0]
    return int_number

@triton.jit
def construct_float(sign, biased_exp, trailing_mantissa):
    bits = (sign << 31) | (biased_exp << 23) | trailing_mantissa
    return bits.to(tl.float32, bitcast=True)
    
def get_total_size(A: torch.tensor):
    return torch.numel(A)

def get_grid(numel):
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']), )
    return grid

@triton.jit
def get_sign(float_input: tl.tensor):
    sign = float_input.to(tl.uint32, bitcast=True) >> 31 #FLOAT32_SIGN_OFFSET
    return sign

@triton.jit
def get_trailing_mantissa(float_input: tl.tensor):
    int_input = float_input.to(tl.uint32, bitcast=True)
    trail_mant = int_input & 0x007fffff
    return  trail_mant #FLOAT32_MANTISSA_MASK

@triton.jit
def get_biased_exponent(float_input: tl.tensor):
    # float_input = float_input.to(tl.float32)
    int_input = float_input.to(tl.uint32, bitcast=True)
    exp = int_input & 0x7F800000 #FLOAT32_EXP_MASK
    exp = exp >> 23 # FLOAT32_EXP_OFFSET
    return int_input

@triton.jit
def get_unbiased_exponent(input: tl.tensor):
    float_input = input.to(tl.float32)
    exp = get_biased_exponent(float_input)
    # if ((exp | (~exp + 1)) >> 31) & 1:    # bit manip to check if exponent is 0
    new_exp_tensor_true = tl.zeros_like(exp) - 126
    exp_new = tl.where(exp==0, new_exp_tensor_true, exp - 127)
    return exp_new.to(tl.int32)

@triton.jit
def clamp_shared_exp(shared_exp: tl.tensor, scale_bits):    
    if scale_bits != 0:
        emax = 1 << (scale_bits-1) -1
    else:
        emax = 255 # FLOAT32_EXP_MAX
    shared_ub = shared_exp - 127 # shared_exp - FLOAT32_EXP_BIAS
    exp_max_tensor = tl.zeros_like(shared_exp) + 255
    exp_bias_tensor = tl.zeros_like(shared_exp) + 127 - emax
    shared_exp = tl.where(shared_ub > emax, exp_max_tensor, shared_exp)
    shared_exp = tl.where(shared_exp < -emax, exp_bias_tensor, shared_exp)
    return shared_exp

@triton.jit
def get_shared_scale(shared_exp: tl.tensor, scale_bits, elem_max_norm):
    elem_emax = get_unbiased_exponent(elem_max_norm)
    shared_exp = tl.where(shared_exp != 255, shared_exp - elem_emax, shared_exp)
    shared_exp = clamp_shared_exp(shared_exp, scale_bits)

    scale_mant_up = tl.zeros_like(shared_exp).to(tl.int32) + ((1 << 23) >> 1)
    scale_mant_zeros = tl.zeros_like(shared_exp).to(tl.int32)

    scale_mant = tl.where(shared_exp == 0 or shared_exp == 0x7f800000, scale_mant_up, scale_mant_zeros)
    
    return construct_float(scale_mant_zeros, shared_exp, scale_mant).to(tl.float32)
    

    