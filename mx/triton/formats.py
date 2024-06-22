"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

from enum import Enum, IntEnum
from bidict import bidict

from mltools.numerical.format import Format, FloatingPoint

MXFP8_E4M3K128_LD=Format.from_shorthand("MXFP8[E4M3]{128,-1}")
MXFP8_E4M3K128_FD=Format.from_shorthand("MXFP8[E4M3]{128,1}")
MXFP8_E4M3K64_LD=Format.from_shorthand("MXFP8[E4M3]{64,-1}")
MXFP8_E4M3K64_FD=Format.from_shorthand("MXFP8[E4M3]{64,1}")
MXFP8_E4M3K32_LD=Format.from_shorthand("MXFP8[E4M3]{32,-1}")
MXFP8_E4M3K32_FD=Format.from_shorthand("MXFP8[E4M3]{32,1}")


def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


def _get_max_norm(ebits, mbits):
    """ Valid only for floats that define NaN """
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    emax = 0 if ebits==0 else 2**(ebits - 1) - 1
    return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)


_FORMAT_CACHE = {}
def _get_format_params(fmt):
    """ Allowed formats:
        - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior

        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """
    if type(fmt) is str:
        fmt = Format.from_shorthand(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]
    
    assert isinstance(
            fmt.element_format, FloatingPoint
        )
    
    if fmt.element_format.mantissa == 2 and fmt.element_format.exponent == 5:
        ebits, mbits = 5, 2
        emax = 2**(ebits - 1) - 1
    elif fmt.element_format.mantissa == 3 and fmt.element_format.exponent == 4:
        ebits, mbits = 4, 3
        emax = 2**(ebits - 1)
    else:
        raise Exception("Unknown element format %s" % fmt)
    
    if fmt != MXFP8_E4M3K128_LD and fmt !=  MXFP8_E4M3K128_FD and fmt != MXFP8_E4M3K64_LD and fmt != MXFP8_E4M3K64_FD and fmt != MXFP8_E4M3K32_LD and fmt != MXFP8_E4M3K32_FD:
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)
    
    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm
