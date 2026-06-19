from __future__ import annotations

from dataclasses import dataclass
from typing import Any


float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"
float64 = "float64"
int64 = "int64"

_default_dtype = float32
_default_device = "cpu"


def set_default_dtype(dtype: str) -> None:
    global _default_dtype
    _default_dtype = dtype


def set_default_device(device: str) -> None:
    global _default_device
    _default_device = device


@dataclass
class Tensor:
    shape: tuple[int, ...]
    dtype: str
    device: str
    value: Any = 0

    def __repr__(self) -> str:
        return (
            f"tensor(value={self.value}, shape={list(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )


def _tensor(shape: list[int], value: Any, dtype=None, device=None) -> Tensor:
    return Tensor(
        tuple(shape),
        dtype or _default_dtype,
        device or _default_device,
        value,
    )


def zeros(shape, dtype=None, device=None):
    return _tensor(shape, 0, dtype, device)


def ones(shape, dtype=None, device=None):
    return _tensor(shape, 1, dtype, device)


def full(shape, fill_value, dtype=None, device=None):
    return _tensor(shape, fill_value, dtype, device)


def add(input, other, alpha=1):
    return Tensor(input.shape, input.dtype, input.device, input.value + alpha * other.value)


def sum(input, dim=None, keepdim=False, dtype=None):
    shape = list(input.shape)
    if dim is None:
        shape = []
    else:
        dims = [dim] if isinstance(dim, int) else dim
        for index in sorted((item % len(shape) for item in dims), reverse=True):
            if keepdim:
                shape[index] = 1
            else:
                shape.pop(index)
    return Tensor(tuple(shape), dtype or input.dtype, input.device, input.value)
