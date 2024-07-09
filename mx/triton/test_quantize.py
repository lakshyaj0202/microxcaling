"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import torch
import numpy as np
import sys

from common_lib import (
        check_diff,
        check_diff_quantize,
        all_encodings
)

from mx_ops import _quantize_mx

from mltools.numerical.format import Format

np.random.seed(0xd10)

MXFP8_E4M3K128_LD=Format.from_shorthand("MXFP8[E4M3]{128,-1}")
MXFP8_E4M3K128_FD=Format.from_shorthand("MXFP8[E4M3]{128,1}")
MXFP8_E4M3K64_LD=Format.from_shorthand("MXFP8[E4M3]{64,-1}")
MXFP8_E4M3K64_FD=Format.from_shorthand("MXFP8[E4M3]{64,1}")
MXFP8_E5M2K128_LD=Format.from_shorthand("MXFP8[E5M2]{128,-1}")
MXFP8_E5M2K128_FD=Format.from_shorthand("MXFP8[E5M2]{128,1}")
MXFP8_E5M2K64_LD=Format.from_shorthand("MXFP8[E5M2]{64,-1}")
MXFP8_E5M2K64_FD=Format.from_shorthand("MXFP8[E5M2]{64,1}")

DEVICE__CUSTOM_CUDA = [
    ("cuda", True),
]

ELEM_FMTS = [
    (MXFP8_E4M3K128_LD),
    (MXFP8_E5M2K128_LD),
]

RANDOM_SEED = 4
@pytest.mark.parametrize("elem_format", ELEM_FMTS)
@pytest.mark.parametrize("round", ("N"))
@pytest.mark.parametrize("flush_fp32_subnorms", (False,True))
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_encoding(elem_format, round,
                     flush_fp32_subnorms, device, custom_cuda):
    # print("elem_format", elem_format)
    # print("flush_fp32_subnorms", flush_fp32_subnorms)
    # print("device", device)
    torch.manual_seed(RANDOM_SEED)
    scale_bits = elem_format.scaler_format.exponent
    block_size = elem_format.block_size
    x1 = torch.rand((128, 128), device="cuda")
    x2 = x1.clone().detach().to(device)
    # x1 = all_encodings(8, 9, device="cuda")
    # x2 = x1.clone().detach().to(device)

    y1 = _quantize_mx(x1, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=False)


    y2 = _quantize_mx(x2, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=custom_cuda)

    # print()
    # print() 
    # print("y1 - cpu")
    # print(y1)
    # print("y2 - triton")
    # print(y2)

    check_diff_quantize(x1, y1, y2)
