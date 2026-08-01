"""CNF encodings for finite polynomial-distance graph coloring."""

from pvdw.encoding.binary import BinaryEncodingResult, decode_binary_model, encode_binary
from pvdw.encoding.onehot import (
    AtMostOneEncoding,
    OneHotEncodingResult,
    encode_onehot,
)

__all__ = [
    "AtMostOneEncoding",
    "BinaryEncodingResult",
    "OneHotEncodingResult",
    "decode_binary_model",
    "encode_binary",
    "encode_onehot",
]
