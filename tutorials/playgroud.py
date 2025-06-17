import numpy as np
from numba import njit, prange
from math import gamma, pow, sin, pi
from scipy.special import kv

# %%

import numpy as np
import math
import numba as nb
# from scipy.special import gamma
from scipy.sparse import block_diag

@nb.njit(nb.complex128(nb.complex128), cache=False)
def _kv56_scalar(z):
    """Scalar implementation used as kernel for array version"""
    # Precomputed Gamma function values for v=5/6
    gamma_1_6 = 5.56631600178  # Gamma(1/6)
    gamma_11_6 = 0.94065585824  # Gamma(11/6)
    # Precompute constants for numerical stability
    # Constants for the series expansion and asymptotic approximation
    v = 5.0 / 6.0
    z_abs = np.abs(z)
    if z_abs < 2.0:
        # Series expansion for small |z|
        sum_a = 0.0j
        sum_b = 0.0j
        term_a = (0.5 * z)**v / gamma_11_6
        term_b = (0.5 * z)**-v / gamma_1_6
        sum_a += term_a
        sum_b += term_b
        z_sq_over_4 = (0.5 * z)**2
        k = 1
        tol = 1e-15
        max_iter = 1000
        for _ in range(max_iter):
            factor_a = z_sq_over_4 / (k * (k + v))
            term_a *= factor_a
            sum_a += term_a
            factor_b = z_sq_over_4 / (k * (k - v))
            term_b *= factor_b
            sum_b += term_b
            if abs(term_a) < tol * abs(sum_a) and abs(term_b) < tol * abs(sum_b):
                break
            k += 1
        K = np.pi * (sum_b - sum_a)
    else:
        # Asymptotic expansion for large |z|
        z_inv = 1.0 / z
        sum_terms = 1.0 + (2.0/9.0)*z_inv + (-7.0/81.0)*z_inv**2 + \
                    (175.0/2187.0)*z_inv**3 + (-2275.0/19683.0)*z_inv**4 + \
                    (5005.0/177147.0)*z_inv**5  #+ (-2662660.0/4782969.0)*z_inv**6
        prefactor = np.sqrt(np.pi/(2.0*z)) * np.exp(-z)
        K = prefactor * sum_terms
    return K

# Vectorized version outside the class
@nb.vectorize([nb.complex128(nb.complex128),  # Complex input
            nb.complex128(nb.float64)],    # Real input
            nopython=True, target='parallel')
def _kv56(z):
    """
    Modified Bessel function K_{5/6}(z) for numpy arrays
    Handles both real and complex inputs efficiently
    """
    return _kv56_scalar(z)




# %%

@njit(parallel=True)
def bessel_i_cpu(x, n, terms):
    size = x.shape[0]
    result = np.zeros(size, dtype=np.float64)

    for idx in prange(size):
        xi = x[idx]
        coeff = pow(xi / 2.0, n) / gamma(n + 1.0)
        term = coeff
        sum_result = term
        x_sq_half = pow(xi / 2.0, 2)

        for m in range(1, terms):
            term *= x_sq_half / (m * (m + n))
            sum_result += term

        result[idx] = sum_result

    return result


def bessel_k_cpu(n, x, terms=20):
    x = np.asarray(x, dtype=np.float64)
    i_n = bessel_i_cpu(x, n, terms)
    i_neg_n = bessel_i_cpu(x, -n, terms)

    with np.errstate(divide='ignore', invalid='ignore'):
        k = (pi / 2.0) * (i_neg_n - i_n) / np.sin(n * pi)
        # Avoid division by zero for integer n
        k[np.isclose(np.sin(n * pi), 0.0)] = np.nan
    return k


# %%
from time import time


# %%
u = np.random.rand(5000000)

start = time()
sp = kv(5/6, u)
print(time() - start)
print(sp)

print()
start = time()
cpu = bessel_k_cpu(5/6, u, 10)
print(time() - start)
print(np.mean(sp-cpu))


print()
start = time()
usa = _kv56(u)
print(time() - start)
print(np.mean(sp-np.real(usa)))




