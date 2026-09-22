# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:22:55 2020

@author: cheritie

Computation of the meta-sensitivity matrix used by SPRINT.

The mis-registrations are applied through DeformableMirror.apply_mis_registration,
so any DM (synthetic or built from user-defined InfluenceFunctions) is supported
without passing a parameter file or an extra "dm_input".
"""

import os

import numpy as np
import skimage.transform as sk
from astropy.io import fits as pfits

from ..MisRegistration import MisRegistration
from ..calibration.CalibrationVault import CalibrationVault
from ..calibration.InteractionMatrix import InteractionMatrix, InteractionMatrixFromPhaseScreen
from ..runtime import to_backend
from ..tools.interpolateGeometricalTransformation import (anamorphosisImageMatrix,
                                                          rotateImageMatrix,
                                                          translationImageMatrix)
from ..tools.tools import createFolder, warning

# ------------------------------------------------------------------------------------------------
# Parameters that SPRINT can identify, in the order indexed by `ind_mis_reg`
# ------------------------------------------------------------------------------------------------
MIS_REG_FIELDS = ('shiftX', 'shiftY', 'rotationAngle', 'magnification', 'radialScaling', 'tangentialScaling')
MIS_REG_SHORT_NAMES = ('dX', 'dY', 'dRot', 'dMagn', 'dmX', 'dmY')
MIS_REG_TITLES = ('Shift X', 'Shift Y', 'Rotation Angle', 'Magnification', 'radialScaling', 'tangentialScaling')
MIS_REG_UNITS = ('[m]', '[m]', '[deg]', '[%]', '[%]', '[%]')

# push-pull amplitude used for all interaction matrices [m]
STROKE = 1e-9


def select_mis_reg_indices(ind_mis_reg=None, n_mis_reg=None):
    """Indices (in MIS_REG_FIELDS) of the parameters to identify.

    ind_mis_reg defaults to the first n_mis_reg parameters. If both are given,
    ind_mis_reg is truncated to n_mis_reg entries (historical behaviour).
    """
    if ind_mis_reg is None:
        ind = np.arange(len(MIS_REG_FIELDS) if n_mis_reg is None else n_mis_reg)
    else:
        ind = np.atleast_1d(np.asarray(ind_mis_reg, dtype=int))
        if n_mis_reg is not None and len(ind) > n_mis_reg:
            warning('ind_mis_reg has ' + str(len(ind)) + ' entries but n_mis_reg = ' + str(n_mis_reg)
                    + ': only ' + str([MIS_REG_FIELDS[i] for i in ind[:n_mis_reg]]) + ' are identified.')
            ind = ind[:n_mis_reg]
    return ind


def warn_unused_arguments(func_name, kwargs):
    """Warn about arguments that became obsolete when the mis-registrations moved to the DM class."""
    unused = [key for key, val in kwargs.items() if val is not None]
    if unused:
        warning(func_name + ': argument(s) ' + ', '.join(unused) + ' are no longer used and are ignored. '
                'The mis-registrations are applied with DeformableMirror.apply_mis_registration.')


def same_mis_registration(m1, m2):
    return np.allclose(m1.as_array(), m2.as_array())


def as_2d(D):
    """Interaction matrix as [n_signal, n_modes], also for a single mode."""
    D = np.asarray(D)
    return D.reshape(D.shape[0], -1)


# ------------------------------------------------------------------------------------------------
# Models of the WFS signals
# ------------------------------------------------------------------------------------------------
def interaction_matrix(ngs, tel, dm, wfs, modes):
    return InteractionMatrix(ngs, tel, dm, wfs, modes, STROKE,
                             phaseOffset=0, nMeasurements=1, invert=False, print_time=False)


def dm_modes_opd(dm, modes):
    """OPD [n_pix, n_pix(, n_modes)] of the modes applied on the DM (used by the fast algorithm)."""
    dm.coefs = modes
    opd = np.array(to_backend(dm.OPD, np), copy=True)
    dm.coefs = 0
    return opd


def apply_mis_reg(tel, map_2d, misReg):
    """Interpolate a 2D map (or a cube [n_pix, n_pix, n]) to apply misReg (fast algorithm)."""
    pixelsize = tel.D / tel.resolution
    reference = np.zeros(map_2d.shape[:2])
    transformation = (anamorphosisImageMatrix(reference, misReg.anamorphosisAngle,
                                              [1 + misReg.radialScaling, 1 + misReg.tangentialScaling])
                      + rotateImageMatrix(reference, misReg.rotationAngle)
                      + translationImageMatrix(reference, [misReg.shiftY / pixelsize, misReg.shiftX / pixelsize]))

    def warp(image):
        return sk.warp(image, transformation.inverse, order=3)

    if np.ndim(map_2d) == 2:
        return warp(map_2d)
    return np.stack([warp(map_2d[:, :, i]) for i in range(map_2d.shape[2])], axis=2)


def interaction_matrix_fast(ngs, atm, tel, wfs, opd_modes, misReg):
    """Interaction matrix of interpolated DM modes (fast algorithm)."""
    pupil = tel.pupil if np.ndim(opd_modes) == 2 else tel.pupil[:, :, None]
    opd = pupil * apply_mis_reg(tel, opd_modes, misReg)
    is_paired = tel.isPaired  # InteractionMatrixFromPhaseScreen un-pairs the telescope
    try:
        return InteractionMatrixFromPhaseScreen(ngs, atm, tel, wfs, opd, STROKE,
                                                phaseOffset=0, nMeasurements=1, invert=False, print_time=False)
    finally:
        tel.isPaired = is_paired


# ------------------------------------------------------------------------------------------------
# Meta-sensitivity matrix
# ------------------------------------------------------------------------------------------------
def sensitivity_folder(nameFolder, nameSystem, misRegistrationZeroPoint, basis, dm):
    """Folder and file suffix where the sensitivity matrices of this configuration are stored."""
    modes = np.squeeze(basis.modes)
    zonal = modes.shape == (dm.nValidAct, dm.nValidAct) and np.array_equal(modes, np.eye(dm.nValidAct))
    folder = nameFolder + nameSystem + misRegistrationZeroPoint.misRegName + '/' + ('zon/' if zonal else 'mod/')
    suffix = '' if zonal else '_' + getattr(basis, 'extra', '')
    return folder, suffix


def computeMetaSensitivityMatrix(tel,
                                 ngs,
                                 dm_0,
                                 wfs,
                                 basis,
                                 misRegistrationZeroPoint,
                                 epsilonMisRegistration,
                                 nameFolder=None,
                                 nameSystem='',
                                 save_sensitivity_matrices=True,
                                 recompute_sensitivity=False,
                                 fast=False,
                                 atm=None,
                                 n_mis_reg=3,
                                 ind_mis_reg=None,
                                 **deprecated_kwargs):
    """Compute the set of sensitivity matrices required to identify the mis-registrations.

    Parameters
    ----------
    tel, ngs, wfs : Telescope, Source, WFS objects.
    dm_0 : DeformableMirror
        Reference DM. The perturbed DMs are generated with dm_0.apply_mis_registration.
    basis : object
        basis.modes : [n_act, n_modes] commands of the modal basis,
        basis.extra : string used to name the sensitivity matrices (e.g. 'KL').
    misRegistrationZeroPoint : MisRegistration
        Working point around which the sensitivity matrices are computed.
    epsilonMisRegistration : MisRegistration
        Amplitude of the +/- perturbation for each parameter.
    nameFolder, nameSystem : str, optional
        If nameFolder is given (and save_sensitivity_matrices is True), the matrices are
        loaded from / saved to nameFolder + nameSystem + <zero point name>/(zon|mod)/.
    recompute_sensitivity : bool
        Ignore existing files (they are overwritten).
    fast : bool
        Interpolate the DM modes instead of building new DMs (WARNING: not stable).
        Requires atm.
    n_mis_reg, ind_mis_reg :
        Parameters to identify, see select_mis_reg_indices.

    Returns
    -------
    metaSensitivityMatrix : CalibrationVault
        Each column is one sensitivity matrix reshaped as a vector. Pseudo-inverse in .M
    calib_0 : CalibrationVault
        Interaction matrix at misRegistrationZeroPoint.
    """
    warn_unused_arguments('computeMetaSensitivityMatrix', deprecated_kwargs)
    if fast and atm is None:
        raise ValueError('The fast algorithm requires the atm object.')

    modes = np.squeeze(basis.modes)
    ind = select_mis_reg_indices(ind_mis_reg, n_mis_reg)

    # ------------------ storage of the matrices ------------------
    use_files = save_sensitivity_matrices and nameFolder is not None
    if use_files:
        folder, suffix = sensitivity_folder(nameFolder, nameSystem, misRegistrationZeroPoint, basis, dm_0)
        createFolder(folder)
    loaded_files = []

    def load_or_compute(filename, title, compute):
        path = folder + 'im_' + filename + suffix + '.fits' if use_files else None
        if path is not None and not recompute_sensitivity and os.path.isfile(path):
            with pfits.open(path) as hdu:
                loaded_files.append(path)
                return CalibrationVault(hdu[1].data, invert=False)
        calib = compute()
        if path is not None:
            hdr = pfits.Header()
            hdr['TITLE'] = title
            pfits.HDUList([pfits.PrimaryHDU(header=hdr), pfits.ImageHDU(calib.D)]).writeto(path, overwrite=True)
        return calib

    # ------------------ model at a given mis-registration ------------------
    if fast:
        opd_modes = dm_modes_opd(dm_0, modes)

        def model(mis_reg):
            return interaction_matrix_fast(ngs, atm, tel, wfs, opd_modes, mis_reg)
    else:
        def model(mis_reg):
            return interaction_matrix(ngs, tel, dm_0.apply_mis_registration(mis_reg), wfs, modes)

    # ------------------ interaction matrix at the zero point ------------------
    def compute_calib_0():
        if fast:
            return model(misRegistrationZeroPoint)
        # rebuild the DM only if it is not already at the zero point
        if same_mis_registration(dm_0.misReg, misRegistrationZeroPoint):
            return interaction_matrix(ngs, tel, dm_0, wfs, modes)
        return model(misRegistrationZeroPoint)

    calib_0 = load_or_compute('0', 'INTERACTION MATRIX - INITIAL POINT', compute_calib_0)

    # ------------------ +/- epsilon for each parameter ------------------
    columns = []
    for i in ind:
        field = MIS_REG_FIELDS[i]
        epsilon = getattr(epsilonMisRegistration, field)
        calibs = []
        for sign, tag, label in [(+1, 'p', 'POSITIVE'), (-1, 'm', 'NEGATIVE')]:
            mis_reg = MisRegistration(misRegistrationZeroPoint)
            setattr(mis_reg, field, getattr(mis_reg, field) + sign * epsilon)
            calibs.append(load_or_compute(MIS_REG_SHORT_NAMES[i] + '_' + tag + '_' + str(np.abs(epsilon)),
                                          'INTERACTION MATRIX - ' + label + ' MIS-REGISTRATION',
                                          lambda m=mis_reg: model(m)))
        # centered finite difference, reshaped as a vector
        columns.append(((as_2d(calibs[0].D) - as_2d(calibs[1].D)) / (2 * epsilon)).reshape(-1))

    if loaded_files:
        warning('Existing sensitivity matrices loaded from\n' + os.path.dirname(loaded_files[0])
                + '\nMake sure that the loaded data correspond to your AO system!')

    return CalibrationVault(np.stack(columns, axis=1)), calib_0