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
    if mode == "cuda":
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("CUDA backend requested but no device is visible")
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError("CUDA backend requested but no device is usable") from error
    return cp, True
