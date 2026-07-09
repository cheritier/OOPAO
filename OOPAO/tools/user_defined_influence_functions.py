# -*- coding: utf-8 -*-
"""
Influence-function generation for various deformable mirrors.

Created on Wed Jul  8 16:14:04 2026

@author: cheritier
"""

import time

import numpy as np
import scipy.io
import skimage.transform as sk
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

from OOPAO.calibration.ao_cockpit_psim import mkp
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Telescope import Telescope
from OOPAO.tools.interpolateGeometricalTransformation import (
    anamorphosisImageMatrix,
    interpolate_cube,
    rotateImageMatrix,
    translationImageMatrix,
)
from OOPAO.tools.tools import OopaoError


# ---------------------------------------------------------------------------
# ELT - M4
# ---------------------------------------------------------------------------
def compute_ELT_M4_influence_functions(name_system: str,
                                       diameter: float,
                                       resolution: int,
                                       loc: str = None,
                                       mis_registration=None,
                                       flip_lr: bool = False,
                                       flip_ud: bool = False,
                                       specific_parameters: dict = None):
    """
    Generate the ELT-M4 influence functions from the reduced FEM data.

    Parameters
    ----------
    name_system : str
        Name of the system (unused, kept for interface consistency).
    diameter : float
        Telescope diameter in [m].
    resolution : int
        Number of pixels across the pupil.
    loc : str, optional
        Path to the M4 input data.
    mis_registration : MisRegistration, optional
        Mis-registration to apply (shift, rotation, scaling, anamorphosis).
    flip_lr : bool, optional
        Unused, kept for interface consistency.
    flip_ud : bool, optional
        Unused, kept for interface consistency.
    specific_parameters : dict, optional
        Keys: 'n_segments', 'new_arrangement', 'parallel', 'n_jobs'.
        Defaults: 6 segments, original arrangement, serial, 6 jobs.

    Raises
    ------
    OopaoError
        If the M4 input data cannot be found at `loc`.

    Returns
    -------
    IF_2D : np.ndarray (float32)
        Influence-function cube of shape [n_act, resolution, resolution].
    """
    # -------------------- input parameters --------------------
    if specific_parameters is None:
        n_segments = 6
        new_arrangement = False
        parallel = False
        n_jobs = 6
    else:
        n_segments = specific_parameters['n_segments']
        new_arrangement = specific_parameters['new_arrangement']
        parallel = specific_parameters['parallel']
        n_jobs = specific_parameters['n_jobs']

    pixel_size = diameter / resolution
    nact_per_seg = 892
    nact = 5352 * n_segments // 6
    n_nodes = 123903

    # magnification between M1 and M4 coordinate systems
    mag = 17.11994034069545

    if new_arrangement:
        idx = fits.getdata(loc + 'IDX_NEW.fits')

    # -------------------- FEM node coordinates --------------------
    try:
        node_coord = np.loadtxt(loc + 'M4coords/node_coord.txt', np.float64)
    except FileNotFoundError:
        raise OopaoError(
            'Could not find the ELT-M4 data. Make sure you indicated the '
            'correct path in the parameter file.\n'
            'If you do not have the input data, please contact '
            'cedric-taissir.heritier@lam.fr to get the access.')
    node_xy = node_coord[:, 0:2]

    # -------------------- FEM influence functions --------------------
    # file created with FEMreduce_4tuto.py
    print('LOADING FEM DATA...\n')
    IFMA = np.fromfile(loc + 'SAVED_DATA/IFMA_FEM_123903x892.raw',
                       dtype=np.float64).reshape(n_nodes, nact_per_seg)
    if new_arrangement:
        IFMA = IFMA[:, idx]

    # -------------------- coordinate system in M1 --------------------
    diam_max = (resolution - 1) * pixel_size  # largest diameter in array
    cobs = 0.  # central obstruction, not used
    GEO = mkp(diam_max, resolution, diam_max, cobs)
    XX = GEO.xx
    YY = GEO.yy

    # ---------- transformation from M4 to M1 with distortion ----------
    print('PREPARING COORDINATE SYSTEM INCLUDING DISTORTION...\n')

    # node coordinates scaled from M4 to M1
    xy_coord = node_xy * mag

    # rotated node positions for the 6 segments: xy_coords[n_nodes, 2, 6]
    theta = np.arange(6) * 2. * np.pi / 6
    xy_coords = np.zeros([n_nodes, 2, 6], dtype=np.float64)
    for k in range(6):
        rot = np.array([[np.cos(theta[k]), -np.sin(theta[k])],
                        [np.sin(theta[k]),  np.cos(theta[k])]])
        xy_coords[:, :, k] = xy_coord @ rot.T

    # coordinate system in M4 plane
    M4X = XX / mag
    M4Y = YY / mag

    # M1 as seen through M4, distortion by tilted converging beam
    # (coefficients from optical design fit)
    XXp = 17.0414 * M4X + 0.0587 * M4X**2
    YYp = -17.1984 * M4Y - 0.0592 * M4X * M4Y

    # correct flip due to reflection
    YYp = np.flip(YYp, axis=0)

    # -------------------- mis-registrations --------------------
    m_rad = 1 - mis_registration.radialScaling / 100
    m_tan = 1 - mis_registration.tangentialScaling / 100
    ana_angle = np.deg2rad(mis_registration.anamorphosisAngle)
    rot_angle = np.deg2rad(mis_registration.rotationAngle)

    # radial / tangential scaling and anamorphosis
    # NOTE: the updated XXp is reused in the YYp expression below,
    # reproducing the behaviour of the original implementation.
    XXp = (XXp * (m_rad * np.cos(ana_angle)**2 + m_tan * np.sin(ana_angle)**2)
           + YYp * (m_tan * np.sin(2 * ana_angle) / 2
                    - m_rad * np.sin(2 * ana_angle) / 2))
    YYp = (YYp * (m_rad * np.sin(ana_angle)**2 + m_tan * np.cos(ana_angle)**2)
           + XXp * (m_tan * np.sin(2 * ana_angle) / 2
                    - m_rad * np.sin(2 * ana_angle) / 2))

    # rotation
    XX_out = XXp * np.cos(rot_angle) - YYp * np.sin(rot_angle)
    YY_out = XXp * np.sin(rot_angle) + YYp * np.cos(rot_angle)

    # shift
    XX_out -= mis_registration.shiftX
    YY_out -= mis_registration.shiftY

    # query points (fixed for all segments and actuators)
    qpts = np.column_stack([XX_out.ravel(), YY_out.ravel()])

    # ---------- fast linear interpolation via precomputed sparse operator ----------
    def build_interp_matrix(points, query):
        """
        Build sparse matrix P [n_query, n_nodes] such that P @ values
        reproduces scipy.interpolate.griddata(points, values, query,
        method='linear'), with points outside the convex hull set to 0
        instead of NaN.
        """
        tri = Delaunay(points)
        simplex = tri.find_simplex(query)
        inside = simplex >= 0
        s = np.where(inside, simplex, 0)

        # barycentric coordinates
        T = tri.transform[s]                # [n_q, 3, 2]
        r = query - T[:, 2, :]              # offset from simplex origin
        b = np.einsum('nij,nj->ni', T[:, :2, :], r)
        w = np.column_stack([b, 1. - b.sum(axis=1)])   # [n_q, 3]
        w[~inside] = 0.

        verts = tri.simplices[s]            # [n_q, 3]
        rows = np.repeat(np.arange(len(query)), 3)
        return csr_matrix((w.ravel(), (rows, verts.ravel())),
                          shape=(len(query), points.shape[0]))

    def process_segment(k):
        """Interpolate all IFs of segment k.

        Returns an array [nact_per_seg, resolution, resolution] (float32).
        """
        P = build_interp_matrix(xy_coords[:, :, k], qpts)  # one triangulation
        seg = (P @ IFMA).astype(np.float32)                # [n_q, 892]
        return seg.T.reshape(nact_per_seg, resolution, resolution)

    # -------------------- run interpolations --------------------
    print('RUNNING INTERPOLATIONS...\n')
    t_start = time.time()
    IF_2D = np.zeros([nact, resolution, resolution], dtype=np.float32)

    if parallel and n_segments > 1:
        # One process per segment. Reduce n_jobs if RAM is tight; large
        # read-only arrays (IFMA, qpts) are memory-mapped to workers by
        # joblib, not fully copied.
        results = Parallel(n_jobs=min(n_jobs, n_segments), prefer='processes')(
            delayed(process_segment)(k) for k in range(n_segments))
        for k, seg in enumerate(results):
            IF_2D[k * nact_per_seg:(k + 1) * nact_per_seg] = seg
        del results
    else:
        for k in range(n_segments):
            print('Segment', k, ' ', end='\r', flush=True)
            IF_2D[k * nact_per_seg:(k + 1) * nact_per_seg] = process_segment(k)

    print('\nELAPSED TIME FOR FEM INTERPOLATION:', time.time() - t_start)
    return IF_2D


# ---------------------------------------------------------------------------
# RAMA - DM97
# ---------------------------------------------------------------------------
def compute_RAMA_DM97_influence_functions(name_system: str,
                                          diameter: float,
                                          resolution: int,
                                          loc: str = None,
                                          mis_registration=None,
                                          flip_lr: bool = False,
                                          flip_ud: bool = False,
                                          specific_parameters: dict = None):
    """
    Load and resample the RAMA DM97 influence functions.

    Returns
    -------
    IF_2D : np.ndarray
        Influence-function cube of shape [n_act, resolution, resolution].
    """
    try:
        IF_2D = np.load(loc + 'IF_97.npy')
    except FileNotFoundError:
        raise OopaoError(
            'Could not find the RAMA data. Make sure you downloaded it from '
            'https://nuage.osupytheas.fr/s/YRbHrHSQA9ZSiQP and indicated the '
            'correct path in the parameter file')

    pixel_size = diameter / resolution

    # re-order to put n_actuator first, crop extra pixels and re-center
    IF_2D = np.moveaxis(IF_2D, 2, 0)[:, 3:-3, 6:]
    n_if, n_px1, n_px2 = IF_2D.shape

    # pixel size projected on sky
    pixel_size_input = diameter / n_px1

    if resolution != n_px1:
        IF_2D = np.asarray(interpolate_cube(cube_in=IF_2D,
                                            pixel_size_in=pixel_size_input,
                                            pixel_size_out=pixel_size,
                                            resolution_out=resolution,
                                            mis_registration=mis_registration))
    return IF_2D


# ---------------------------------------------------------------------------
# LBT - ASM
# ---------------------------------------------------------------------------
def compute_LBT_ASM_influence_functions(name_system: str,
                                        diameter: float,
                                        resolution: int,
                                        loc: str = None,
                                        mis_registration=None,
                                        flip_lr: bool = False,
                                        flip_ud: bool = False,
                                        specific_parameters: dict = None):
    """
    Compute the LBT ASM influence functions from the mirror eigen modes.

    Returns
    -------
    influence_functions : np.ndarray
        Influence-function cube of shape [n_act, resolution, resolution].
    """
    if mis_registration is None:
        mis_registration = MisRegistration()

    filename_IF = loc + 'phase_matrix.sav'
    filename_coordinates = loc + 'act_coordinates.fits'
    filename_M2C = loc + 'm2c.fits'

    # -------------------- load eigen modes of the mirror --------------------
    try:
        tmp = scipy.io.readsav(filename_IF)
    except FileNotFoundError:
        raise OopaoError(
            'Could not find the LBT-ASM data. Make sure you indicated the '
            'correct path in the parameter file.\n'
            'If you do not have the input data, please contact '
            'cedric-taissir.heritier@lam.fr to get the access.')

    d_pix = tmp['dpix']
    n_modes = tmp['klmatrix'].shape[1]
    modes_ASM = np.zeros([d_pix * d_pix, n_modes])
    modes_ASM[tmp['idx_mask'], :] = tmp['klmatrix']
    modes_ASM = modes_ASM.reshape(d_pix, d_pix, n_modes).T

    n_act, nx, ny = modes_ASM.shape

    # -------------------- pixel scales --------------------
    # NOTE: physical pixel sizes are computed BEFORE any padding, since
    # padding adds pixels but does not change the pixel scale.
    diameter_ASM = 8.25  # [m]
    pixel_size_original = diameter_ASM / nx
    resolution_original = int(nx)
    pixel_size = diameter_ASM / resolution

    # ratio between both pixel scales
    ratio_ASM = pixel_size_original / pixel_size
    # after interpolation the image is shifted by a fraction of pixel
    # if ratio_ASM is not an integer
    extra = ratio_ASM % 1

    # difference in pixels between both resolutions
    n_pix = resolution_original - resolution

    # -------------------- upsampling case --------------------
    if n_pix < 0:
        pad = (-n_pix + 1) // 2 + 1  # small margin, symmetric padding
        modes_ASM = np.pad(modes_ASM, ((0, 0), (pad, pad), (pad, pad)))
        resolution_original += 2 * pad
        n_pix = resolution_original - resolution  # now >= 0
    if n_pix % 2 == 0:
        # even case: align the array with respect to the interpolation
        extra_x = extra / 2 - 0.5
        extra_y = extra / 2 - 0.5
        n_crop_x = n_pix // 2
        n_crop_y = n_pix // 2
    else:
        # uneven case: crop one extra pixel on one side
        extra_x = extra / 2 - 1.0
        extra_y = extra / 2 - 1.0
        n_crop_x = n_pix // 2
        n_crop_y = n_pix // 2 + 1

    # dummy map used only to define the transformation matrices
    influ_map = np.zeros([resolution_original, resolution_original])

    # ------------- transformations applied in the following order -------------
    # 1) down-scaling to get the right pixel size w.r.t. the M1 resolution
    down_scaling = anamorphosisImageMatrix(influ_map, 0, [ratio_ASM, ratio_ASM])

    # 2) transformations for the mis-registration
    anam_matrix = anamorphosisImageMatrix(
        influ_map,
        mis_registration.anamorphosisAngle,
        [1 + mis_registration.radialScaling, 1 + mis_registration.tangentialScaling])
    rot_matrix = rotateImageMatrix(influ_map, mis_registration.rotationAngle)
    shift_matrix = translationImageMatrix(
        influ_map,
        [-mis_registration.shiftX / pixel_size,
         -mis_registration.shiftY / pixel_size])  # units are in [m]

    # shift of half a pixel to center the images on an even number of pixels
    alignment_matrix = translationImageMatrix(influ_map, [extra_x, extra_y])

    # 3) global transformation matrix
    transformation_matrix = (down_scaling + anam_matrix + rot_matrix
                             + shift_matrix + alignment_matrix)

    def apply_transformation(image):
        # order=1 (bilinear): nearest-neighbor (order=0) produces blocky
        # results, especially when upsampling
        return sk.warp(image, transformation_matrix.inverse, order=1)

    # -------------------- apply the transformation to all modes --------------------
    warped = Parallel(n_jobs=4, prefer='threads')(
        delayed(apply_transformation)(mode) for mode in modes_ASM)
    warped = np.moveaxis(np.asarray(warped), 0, -1)
    # explicit end index: warped[a:-0] would return an empty array when the
    # crop is zero (i.e. resolution == resolution_original)
    warped = warped[n_crop_x:resolution_original - n_crop_y,
                    n_crop_x:resolution_original - n_crop_y, :]
    warped = -warped.reshape(warped.shape[0] * warped.shape[1], n_act)

    # -------------------- project from modal to zonal basis --------------------
    M2C = fits.getdata(filename_M2C)
    valid_act = np.where(M2C[:, 2] != 0)[0].astype(int)
    M2C = M2C[valid_act, :]

    influence_functions = warped @ np.linalg.pinv(M2C[:, :warped.shape[1]])
    influence_functions = np.moveaxis(
        influence_functions.reshape(resolution, resolution, M2C.shape[0]), 2, 0)

    return influence_functions


# ---------------------------------------------------------------------------
# GHOST - DM492
# ---------------------------------------------------------------------------
def compute_GHOST_DM492_influence_functions(name_system: str,
                                            diameter: float,
                                            resolution: int,
                                            loc: str = None,
                                            mis_registration=None,
                                            flip_lr: bool = False,
                                            flip_ud: bool = False,
                                            specific_parameters: dict = None):
    """
    Compute the GHOST DM492 influence functions from the actuator coordinates.

    Returns
    -------
    IF_2D : np.ndarray
        Influence-function cube of shape [n_valid_act, resolution, resolution].
    """
    try:
        dm_coord = scipy.io.loadmat(loc + 'dm_coord.mat')
    except FileNotFoundError:
        raise OopaoError(
            'Could not find the GHOST-DM data. Make sure you indicated the '
            'correct path in the parameter file.\n'
            'If you do not have the input data, please contact '
            'cedric-taissir.heritier@lam.fr to get the access.')

    diameter_dm_ghost = 6.9   # [m]
    telescope_diameter = 6.9  # [m]
    n_act_ghost = 24
    half_span = (n_act_ghost - 1) / 2

    coordinates = np.zeros([492, 2])
    coordinates[:, 0] = (dm_coord['x'] - half_span) * telescope_diameter / 2 / half_span
    coordinates[:, 1] = (dm_coord['y'] - half_span) * telescope_diameter / 2 / half_span

    tel = Telescope(resolution, diameter)
    dm = DeformableMirror(telescope=tel,
                          nSubap=23,
                          mechCoupling=0.15,
                          coordinates=coordinates,
                          misReg=mis_registration,
                          pitch=diameter_dm_ghost / n_act_ghost)

    IF_2D = np.moveaxis(dm.modes.reshape(resolution, resolution, dm.nValidAct), 2, 0)
    return IF_2D


# ---------------------------------------------------------------------------
# PAPYRUS - DM241
# ---------------------------------------------------------------------------
def compute_PAPYRUS_DM241_influence_functions(name_system: str,
                                              diameter: float,
                                              resolution: int,
                                              loc: str = None,
                                              mis_registration=None,
                                              flip_lr: bool = False,
                                              flip_ud: bool = False,
                                              specific_parameters: dict = None):
    """
    Compute the PAPYRUS DM241 influence functions from a Gaussian DM model.

    Returns
    -------
    IF_2D : np.ndarray
        Influence-function cube of shape [n_valid_act, resolution, resolution].
    """
    T152_on_DM_size = 37.5  # size of the T152 pupil on the DM [mm]

    pitch = 2.5             # actuator pitch [mm]
    n_act = 17
    DM_diag_size = n_act * pitch  # [mm]
    scale_T152_DM = DM_diag_size / T152_on_DM_size
    D_T152 = 1.52           # T152 telescope diameter [m]

    # cartesian actuator grid projected on the telescope pupil
    x = np.linspace(-scale_T152_DM * D_T152 / 2, scale_T152_DM * D_T152 / 2, n_act)
    X, Y = np.meshgrid(x, x)
    DM_coordinates = np.asarray([X.ravel(), Y.ravel()]).T

    # keep actuators within the pupil (+ margin of 2.2 pitch)
    dist = np.sqrt(DM_coordinates[:, 0]**2 + DM_coordinates[:, 1]**2)
    DM_coordinates = DM_coordinates[dist <= D_T152 / 2 + 2.2 * pitch * D_T152 / T152_on_DM_size, :]
    DM_pitch = pitch * D_T152 / T152_on_DM_size

    tel = Telescope(resolution, diameter)
    dm = DeformableMirror(telescope=tel,
                          nSubap=16,
                          mechCoupling=0.36,
                          misReg=mis_registration,
                          coordinates=DM_coordinates,
                          pitch=DM_pitch,
                          modes=None,
                          flip_lr=True,
                          print_dm_properties=False)

    IF_2D = np.moveaxis(dm.modes.reshape(resolution, resolution, dm.nValidAct), 2, 0)
    return IF_2D


# ---------------------------------------------------------------------------
# EKARUS - DM468
# ---------------------------------------------------------------------------
def compute_EKARUS_DM468_influence_functions(name_system: str,
                                             diameter: float,
                                             resolution: int,
                                             loc: str = None,
                                             mis_registration=None,
                                             flip_lr: bool = False,
                                             flip_ud: bool = False,
                                             specific_parameters: dict = None):
    """
    Load and resample the EKARUS DM468 influence functions.

    Returns
    -------
    IF : np.ndarray
        Influence-function cube of shape [n_act, resolution, resolution].
    """
    try:
        IF = np.load(loc + 'IF_zonal_cube.npy')
    except FileNotFoundError:
        raise OopaoError(
            'Could not find the EKARUS data. Make sure you downloaded it from '
            'https://drive.google.com/drive/folders/1WgqpqZjbBZd3WT_O_Em9wjlS7aQ41mON '
            'and indicated the correct path in the parameter file')

    n_if, n_px1, n_px2 = IF.shape
    pixel_size_input = diameter / n_px1  # pixel size projected on sky

    if resolution != n_px1:
        IF = np.asarray(interpolate_cube(cube_in=IF,
                                         pixel_size_in=pixel_size_input,
                                         pixel_size_out=diameter / resolution,
                                         resolution_out=resolution,
                                         mis_registration=mis_registration))
    return IF