# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:35:26 2020

@author: cheritie

Iterative estimation of the mis-registrations (SPRINT).
"""

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.tools.displayTools import cl_plot
from ..MisRegistration import MisRegistration
from ..tools.tools import warning
from .computeMetaSensitivyMatrix import (MIS_REG_FIELDS, MIS_REG_TITLES, MIS_REG_UNITS,
                                         apply_mis_reg,  # noqa: F401 (kept for backward compatibility)
                                         as_2d,
                                         computeMetaSensitivityMatrix,
                                         dm_modes_opd,
                                         interaction_matrix,
                                         interaction_matrix_fast,
                                         select_mis_reg_indices,
                                         warn_unused_arguments)


def compute_scaling_factor(D_model, D_in, precision):
    """Per-mode optical gain between the data and the model (least-squares)."""
    scaling = np.round(np.sum(D_model * D_in, axis=0) / np.sum(D_model * D_model, axis=0), precision)
    invalid = ~np.isfinite(scaling) | (scaling == 0)
    if invalid.any():
        warning('Invalid scaling factor for mode(s) ' + str(np.where(invalid)[0]) + ', set to 1.')
        scaling[invalid] = 1
    return scaling


def validity_tolerances(pitch, diameter, tolerance):
    """Convergence tolerance for each parameter of MIS_REG_FIELDS."""
    shift = pitch * tolerance
    rotation = np.rad2deg(np.arctan(shift / (diameter / 2)))
    return np.array([shift, shift, rotation, 0.05, 0.05, 0.05])


class _ConvergencePlot:
    """Live plot of the estimated parameters along the iterations."""

    def __init__(self, ind, reference):
        self.reference = reference
        n = len(ind)
        self.plot_obj = cl_plot(list_fig=[[[0, r], [0, r]] for r in reference],
                                type_fig=['plot'] * n,
                                list_title=[MIS_REG_TITLES[i] for i in ind],
                                list_legend=[None] * n,
                                list_label=[['Iteration Number', MIS_REG_UNITS[i]] for i in ind],
                                list_lim=[None] * n,
                                n_subplot=[n, 1],
                                list_display_axis=[True] * n,
                                list_ratio=[[0.95, 0.1], [1] * n], s=20)

    def update(self, history):
        """Returns False if the user asked to stop."""
        values = np.asarray(history)
        self.plot_obj.list_lim = [[-3 * r, 3 * r] for r in self.reference]
        cl_plot(list_fig=[[np.arange(len(values)), values[:, i]] for i in range(values.shape[1])],
                plt_obj=self.plot_obj)
        plt.pause(0.01)
        return self.plot_obj.keep_going is not False


def estimateMisRegistration(tel,
                            ngs,
                            dm_0,
                            wfs,
                            basis,
                            calib_in,
                            misRegistrationZeroPoint,
                            epsilonMisRegistration,
                            calib_0=None,
                            sensitivity_matrices=None,
                            nameFolder=None,
                            nameSystem='',
                            nIteration=3,
                            precision=3,
                            gainEstimation=1,
                            tolerance=1/50,
                            fast=False,
                            atm=None,
                            wfs_mis_registrated=None,
                            ind_mis_reg=None,
                            previous_estimate=None,
                            return_all=False,
                            display=True,
                            plot=True,
                            **deprecated_kwargs):
    """Iterative estimation of the mis-registrations from a set of WFS signals.

    Parameters
    ----------
    tel, ngs, wfs : Telescope, Source, WFS objects.
    dm_0 : DeformableMirror
        Reference DM, the models are built with dm_0.apply_mis_registration.
    basis : object with basis.modes [n_act, n_modes] (and basis.extra).
    calib_in : CalibrationVault
        Measured signals (e.g. on-sky), same shape as the model interaction matrix.
    misRegistrationZeroPoint : MisRegistration
        Starting point, around which the sensitivity matrices were computed.
    epsilonMisRegistration : MisRegistration
        Only used if the sensitivity matrices have to be computed.
    calib_0 : CalibrationVault, optional
        Model at the zero point (saves one interaction matrix at the first iteration).
    sensitivity_matrices : CalibrationVault, optional
        Meta-sensitivity matrix. Computed with computeMetaSensitivityMatrix if None.
    nIteration : int
        Number of iterations.
    precision : int
        Rounding of the estimates, np.round(value, precision).
    gainEstimation : float
        Gain applied to each update (avoids overshoots).
    tolerance : float
        Convergence tolerance in fraction of the DM pitch (for the shifts).
    fast : bool
        Interpolate the DM modes instead of building new DMs (WARNING: not stable). Requires atm.
    wfs_mis_registrated : optional
        Fast algorithm only: the shifts are applied on the WFS and the rotation on the DM.
    ind_mis_reg : list, optional
        Indices of the parameters (see MIS_REG_FIELDS), default: as many as sensitivity matrices.
    previous_estimate : list, optional
        History of a previous call, continued by this one.
    return_all : bool
        Return (mis_registration, scaling_factors, history, validity_flag, calib_last).

    Returns
    -------
    misRegistration_out : MisRegistration
        Absolute estimate. If the estimation did not converge (validity_flag False),
        a copy of misRegistrationZeroPoint is returned.
    """
    warn_unused_arguments('estimateMisRegistration', deprecated_kwargs)
    if fast and atm is None:
        raise ValueError('The fast algorithm requires the atm object.')
    modes = np.squeeze(basis.modes)

    # ---------- sensitivity matrices ----------
    if sensitivity_matrices is None:
        requested = np.arange(3) if ind_mis_reg is None else np.atleast_1d(ind_mis_reg)
        sensitivity_matrices, calib_0 = computeMetaSensitivityMatrix(tel=tel, ngs=ngs, dm_0=dm_0, wfs=wfs, basis=basis,
                                                                     misRegistrationZeroPoint=misRegistrationZeroPoint,
                                                                     epsilonMisRegistration=epsilonMisRegistration,
                                                                     nameFolder=nameFolder,
                                                                     nameSystem=nameSystem,
                                                                     fast=fast,
                                                                     atm=atm,
                                                                     n_mis_reg=len(requested),
                                                                     ind_mis_reg=requested)
    n_mis_reg = sensitivity_matrices.M.shape[0]
    ind = select_mis_reg_indices(ind_mis_reg, n_mis_reg)
    fields = [MIS_REG_FIELDS[i] for i in ind]

    def current_values(mis_reg):
        return np.array([getattr(mis_reg, f) for f in fields], dtype=float)

    # ---------- model at a given mis-registration ----------
    if fast:
        opd_modes = dm_modes_opd(dm_0, modes)

        def model(mis_reg):
            if wfs_mis_registrated is not None:
                # shifts applied in the WFS space, rotation on the DM
                wfs.apply_shift_wfs(mis_reg.shiftX, mis_reg.shiftY)
                mis_reg_dm = MisRegistration()
                mis_reg_dm.rotationAngle = mis_reg.rotationAngle
                mis_reg = mis_reg_dm
            return interaction_matrix_fast(ngs, atm, tel, wfs, opd_modes, mis_reg)
    else:
        def model(mis_reg):
            return interaction_matrix(ngs, tel, dm_0.apply_mis_registration(mis_reg), wfs, modes)

    # ---------- iterative estimation ----------
    misRegistration_out = MisRegistration(misRegistrationZeroPoint)
    D_in = as_2d(calib_in.D)
    scalingFactor_values = [np.ones(D_in.shape[1])]
    if previous_estimate is None:
        misRegistration_values = [current_values(misRegistration_out)]
    else:
        misRegistration_values = previous_estimate
    live_plot = _ConvergencePlot(ind, current_values(misRegistrationZeroPoint)) if plot else None

    flag_paired = tel.isPaired
    tel.isPaired = False
    try:
        for i_iter in range(nIteration):
            if i_iter == 0 and calib_0 is not None and not fast:
                calib_tmp = calib_0
            else:
                calib_tmp = model(misRegistration_out)
            D_model = as_2d(calib_tmp.D)

            scaling_factor = compute_scaling_factor(D_model, D_in, precision)
            residual = (D_in / scaling_factor - D_model).reshape(-1)
            update = np.round(gainEstimation * (sensitivity_matrices.M @ residual), precision)

            for field, delta in zip(fields, update):
                setattr(misRegistration_out, field, getattr(misRegistration_out, field) + delta)

            scalingFactor_values.append(scaling_factor)
            misRegistration_values.append(current_values(misRegistration_out))

            if display:
                misRegistration_out.print_()
            if live_plot is not None and not live_plot.update(misRegistration_values):
                break
    finally:
        tel.isPaired = flag_paired

    # ---------- rounding and validity ----------
    for field in fields:
        setattr(misRegistration_out, field, np.round(getattr(misRegistration_out, field), precision))

    if nIteration == 1 or len(misRegistration_values) < 2:
        validity_flag = True
    else:
        diff = np.abs(misRegistration_values[-1] - misRegistration_values[-2])
        diff[np.isnan(diff)] = np.inf
        validity_flag = bool(np.all(diff <= validity_tolerances(dm_0.pitch, tel.D, tolerance)[ind]))
    if not validity_flag:
        # not converged: keep the working point (absolute), i.e. no correction
        warning('The mis-registration estimation did not converge: returning the zero point.')
        misRegistration_out = MisRegistration(misRegistrationZeroPoint)

    if return_all:
        return misRegistration_out, scalingFactor_values, misRegistration_values, validity_flag, calib_tmp
    return misRegistration_out