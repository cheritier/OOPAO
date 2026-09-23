# -*- coding: utf-8 -*-
"""
Batched bilinear interpolation of an image for several sources at once.

A bilinear interpolation through an axis-aligned transformation (shift, scaling, crop) is separable: each output
row depends on 2 input rows and each output column on 2 input columns. A chain of such transformations is then
described per axis by "taps" (input indices and weights for each output index), computed once on the CPU.
Applying the taps is a gather and a weighted sum: the same code runs on NumPy and CuPy.
The convention is the one of skimage.transform.warp (order=1, mode='constant', cval=0).
"""
import numpy as np

from .interpolateGeometricalTransformation import geometric_transformation


def axis_aligned_params(transformation, tol=1e-12):
    # scale and offset of the inverse map along each axis, None if the transformation mixes the axes
    P = transformation.inverse.params
    if abs(P[0, 1]) > tol or abs(P[1, 0]) > tol or abs(P[2, 0]) > tol or abs(P[2, 1]) > tol or abs(P[2, 2]-1) > tol:
        return None
    return P[0, 0], P[0, 2], P[1, 1], P[1, 2]


def linear_taps(scale, offset, n_out, n_in):
    # sample the input at scale*i + offset, neighbours outside the input have a zero weight
    coord = scale*np.arange(n_out) + offset
    lower = np.floor(coord)
    frac = coord - lower
    index = np.stack([lower, lower+1], axis=-1).astype(np.int64)
    weight = np.stack([1-frac, frac], axis=-1)
    weight[(index < 0) | (index >= n_in)] = 0
    return np.clip(index, 0, n_in-1), weight


def crop_taps(start, n_out, n_in):
    index = np.arange(start, start+n_out)[:, None]
    weight = np.ones((n_out, 1))
    weight[(index < 0) | (index >= n_in)] = 0
    return np.clip(index, 0, n_in-1), weight


def compose_taps(outer, inner):
    # taps of outer applied after inner
    outer_index, outer_weight = outer
    inner_index, inner_weight = inner
    n_out = outer_index.shape[0]
    index = inner_index[outer_index].reshape(n_out, -1)
    weight = (outer_weight[..., None]*inner_weight[outer_index]).reshape(n_out, -1)
    return index, weight


def stack_taps(taps_list):
    # (n_src, n_out, K) arrays, zero-weight taps moved last and dropped when unused by all the sources
    K = max(index.shape[1] for index, _ in taps_list)
    n_out = taps_list[0][0].shape[0]
    index = np.zeros((len(taps_list), n_out, K), dtype=np.int64)
    weight = np.zeros((len(taps_list), n_out, K))
    for i, (idx, w) in enumerate(taps_list):
        order = np.argsort(w == 0, axis=1, kind='stable')
        index[i, :, :idx.shape[1]] = np.take_along_axis(idx, order, axis=1)
        weight[i, :, :w.shape[1]] = np.take_along_axis(w, order, axis=1)
    keep = np.any(weight != 0, axis=(0, 1))
    keep[0] = True
    return index[:, :, keep], weight[:, :, keep]


def shift_crop_zoom_taps(n_in, rows, cols, shift_x, shift_y, magnification, n_out):
    """Taps of: sub-pixel shift of the (n_in, n_in) image, crop [rows, cols], zoom (cone effect)."""
    row_taps = crop_taps(rows.start, n_out, n_in)
    col_taps = crop_taps(cols.start, n_out, n_in)
    if shift_x != 0 or shift_y != 0:
        params = axis_aligned_params(geometric_transformation(n_in, n_in, 1, 1, shift=[shift_x, shift_y]))
        if params is None:
            return None
        sx, ox, sy, oy = params
        row_taps = linear_taps(sy, oy + sy*rows.start, n_out, n_in)
        col_taps = linear_taps(sx, ox + sx*cols.start, n_out, n_in)
    if magnification != 1:
        params = axis_aligned_params(geometric_transformation(n_out, n_out, 1, magnification))
        if params is None:
            return None
        sx, ox, sy, oy = params
        # the zoom is applied to the transposed crop (see Atmosphere.fill_OPD_support): x taps for the rows
        row_taps = compose_taps(linear_taps(sx, ox, n_out, n_out), row_taps)
        col_taps = compose_taps(linear_taps(sy, oy, n_out, n_out), col_taps)
    return row_taps, col_taps


class SeparableTaps:
    def __init__(self, row_taps, col_taps, backend, dtype):
        self.backend = backend
        self.row_index = backend.asarray(row_taps[0])
        self.row_weight = backend.asarray(row_taps[1].astype(dtype))
        self.col_index = backend.asarray(col_taps[0])
        self.col_weight = backend.asarray(col_taps[1].astype(dtype))

    def apply(self, image):
        """Interpolate the 2D image for every source: (n_src, n_out, n_out) array."""
        xp = self.backend
        out_rows = 0
        for k in range(self.row_index.shape[2]):
            out_rows = out_rows + image[self.row_index[:, :, k]] * self.row_weight[:, :, k, None]
        n_src, n_rows = out_rows.shape[:2]
        out = 0
        for k in range(self.col_index.shape[2]):
            index = xp.broadcast_to(self.col_index[:, None, :, k], (n_src, n_rows, self.col_index.shape[1]))
            out = out + xp.take_along_axis(out_rows, index, axis=2) * self.col_weight[:, None, :, k]
        return out
