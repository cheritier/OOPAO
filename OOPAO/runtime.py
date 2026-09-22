"""Process-local backend and precision settings for OOPAO simulations."""

import os
import warnings
from pathlib import Path

import numpy as np

# Default precision when neither OOPAO_PRECISION nor the legacy precision file is available
DEFAULT_PRECISION = 64


def precision_bits():
    """Return 32 or 64; preserve the legacy precision file when unset."""
    configured = os.environ.get("OOPAO_PRECISION")
    if configured is None:
        legacy_file = Path(__file__).resolve().parent.parent / "precision_oopao.npy"
        if not legacy_file.exists():
            # e.g. a regular pip install, where the legacy file is not shipped
            return DEFAULT_PRECISION
        configured = np.load(legacy_file).item()
    try:
        bits = int(configured)
    except (TypeError, ValueError) as error:
        raise ValueError("OOPAO_PRECISION must be 32 or 64") from error
    if bits not in (32, 64) or str(configured) not in (str(bits),):
        raise ValueError("OOPAO_PRECISION must be 32 or 64")
    return bits


# The backend is chosen once per process, at its first use (in practice when OOPAO is imported):
# every module binds `xp` at import time, so a later change of OOPAO_BACKEND could otherwise leave
# modules disagreeing about the backend.
_backend = None
_backend_mode = None


def _select_backend(mode):
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


def array_backend():
    """Select NumPy or CuPy before constructing an optical model.

    The choice is made at the first call and kept for the whole process (set OOPAO_BACKEND
    before importing OOPAO). Returns (array module, gpu_flag).
    """
    global _backend, _backend_mode
    mode = os.environ.get("OOPAO_BACKEND", "auto").lower()
    if _backend is None:
        _backend = _select_backend(mode)
        _backend_mode = mode
    elif mode != _backend_mode:
        warnings.warn(f"OOPAO_BACKEND changed from '{_backend_mode}' to '{mode}' after the backend was selected; "
                      "the backend is fixed for the process at its first use (set OOPAO_BACKEND before importing OOPAO).",
                      RuntimeWarning, stacklevel=2)
    return _backend


_resident = None
_resident_value = None


def gpu_resident():
    """Enable device-resident optical arrays only when explicitly requested.

    Like the backend, the choice is made at the first call and kept for the whole process: objects created
    with different settings would hold NumPy and CuPy arrays that cannot be mixed.
    """
    global _resident, _resident_value
    value = os.environ.get("OOPAO_GPU_RESIDENT", "0")
    if value not in ("0", "1"):
        raise ValueError("OOPAO_GPU_RESIDENT must be 0 or 1")
    if _resident is None:
        _resident = value == "1" and array_backend()[1]
        _resident_value = value
    elif value != _resident_value:
        warnings.warn(f"OOPAO_GPU_RESIDENT changed from '{_resident_value}' to '{value}' after it was first used; "
                      "GPU residency is fixed for the process (set OOPAO_GPU_RESIDENT before creating OOPAO objects).",
                      RuntimeWarning, stacklevel=2)
    return _resident


# ---------------------------------------------------------------------------------------------
# Array helpers shared by the optical classes (NumPy and CuPy arrays cannot be mixed)

def backend_of(array):
    """NumPy or CuPy, whichever holds `array`."""
    xp, gpu = array_backend()
    return xp if gpu and isinstance(array, xp.ndarray) else np


def to_backend(array, backend):
    """Move an array to `backend` (NumPy or CuPy). Python and NumPy scalars are returned unchanged."""
    if np.isscalar(array):
        return array
    if backend is np:
        return to_numpy(array)
    return backend.asarray(array)


def to_numpy(array):
    """NumPy version of a NumPy or CuPy array (a copy only when the array is on the GPU)."""
    xp, gpu = array_backend()
    if gpu and isinstance(array, xp.ndarray):
        return xp.asnumpy(array)
    return np.asarray(array)


def stack_squeeze(items):
    """np.squeeze(np.array(items)), keeping GPU arrays on the GPU."""
    xp, gpu = array_backend()
    if gpu and len(items) > 0 and all(isinstance(item, xp.ndarray) for item in items):
        return xp.squeeze(xp.stack(items))
    return np.squeeze(np.array(items))


def fft_kwargs(workers):
    """Extra FFT arguments: CPU thread count for scipy.fft, nothing for CuPy."""
    return {} if array_backend()[1] else {'workers': workers}