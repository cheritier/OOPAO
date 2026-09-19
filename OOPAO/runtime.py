"""Process-local backend and precision settings for OOPAO simulations."""

import os
from pathlib import Path

import numpy as np


def precision_bits():
    """Return 32 or 64; preserve the legacy precision file when unset."""
    configured = os.environ.get("OOPAO_PRECISION")
    if configured is None:
        configured = np.load(Path(__file__).resolve().parent.parent /
                             "precision_oopao.npy").item()
    try:
        bits = int(configured)
    except (TypeError, ValueError) as error:
        raise ValueError("OOPAO_PRECISION must be 32 or 64") from error
    if bits not in (32, 64) or str(configured) not in (str(bits),):
        raise ValueError("OOPAO_PRECISION must be 32 or 64")
    return bits


def array_backend():
    """Select NumPy or CuPy before constructing an optical model."""
    mode = os.environ.get("OOPAO_BACKEND", "auto").lower()
    if mode == "cpu":
        return np, False
    if mode not in ("auto", "cuda"):
        raise ValueError("OOPAO_BACKEND must be auto, cpu, or cuda")
    try:
        import cupy as cp
    except ImportError as error:
        if mode == "cuda":
            raise RuntimeError("CUDA backend requested but CuPy is unavailable") from error
        return np, False
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        if mode == "cuda":
            raise RuntimeError("CUDA backend requested but no device is usable") from error
        return np, False
    if count < 1:
        if mode == "cuda":
            raise RuntimeError("CUDA backend requested but no device is visible")
        return np, False
    return cp, True


def gpu_resident():
    """Enable device-resident optical arrays only when explicitly requested."""
    value = os.environ.get("OOPAO_GPU_RESIDENT", "0")
    if value not in ("0", "1"):
        raise ValueError("OOPAO_GPU_RESIDENT must be 0 or 1")
    return value == "1" and array_backend()[1]
