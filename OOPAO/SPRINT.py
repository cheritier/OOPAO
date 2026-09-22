# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:30:51 2021

@author: cheritie
"""

import numpy as np

from .MisRegistration import MisRegistration
from .calibration.CalibrationVault import CalibrationVault
from .mis_registration_identification_algorithm.computeMetaSensitivyMatrix import (MIS_REG_FIELDS,
                                                                                   computeMetaSensitivityMatrix,
                                                                                   select_mis_reg_indices)
from .mis_registration_identification_algorithm.estimateMisRegistration import estimateMisRegistration
from .tools.tools import warning


class SPRINT:
    """System Parameters Recurrent INvasive Tracking: identification of the DM/WFS mis-registrations.

    The mis-registered models are generated with obj.dm.apply_mis_registration, so the
    same code handles synthetic DMs and DMs built from user-defined InfluenceFunctions.

    Parameters
    ----------
    obj : object
        Holds tel, ngs, dm, wfs (and atm for the fast algorithm; param for the default folder).
    basis : object
        basis.modes : [n_act, n_modes] commands of the modes used for the identification,
        basis.extra : string used to name the sensitivity matrices (e.g. 'KL').
    nameFolder : str, optional
        Folder where the sensitivity matrices are saved. Defaults to
        obj.param['pathInput'] + '/' + obj.param['name'] + '/s_mat/'; if unavailable,
        the matrices are not saved.
    nameSystem : str, optional
        Name of the system, used in the name of the saved matrices.
    mis_registration_zero_point : MisRegistration, optional
        Working point of the sensitivity matrices. Defaults to obj.dm.misReg.
    wfs_mis_registered : optional
        Fast algorithm only: the shifts are applied in the WFS space.
    fast_algorithm : bool
        Interpolate the DM modes instead of building new DMs (WARNING: not stable).
    n_mis_reg : int
        Number of parameters to identify (default: shiftX, shiftY, rotationAngle).
    recompute_sensitivity : bool
        Recompute (and overwrite) the saved sensitivity matrices.
    ind_mis_reg : list, optional
        Indices of the parameters to identify, among
        ['shiftX', 'shiftY', 'rotationAngle', 'magnification', 'radialScaling', 'tangentialScaling'].
    epsilon_mis_registration : MisRegistration, optional
        Amplitude of the perturbations used to compute the sensitivity matrices.
    dm_input : deprecated
        No longer needed, the mis-registrations are applied by obj.dm itself.
    """

    def __init__(self,
                 obj,
                 basis,
                 nameFolder=None,
                 nameSystem=None,
                 mis_registration_zero_point=None,
                 wfs_mis_registered=None,
                 fast_algorithm=False,
                 n_mis_reg=3,
                 recompute_sensitivity=False,
                 dm_input=None,
                 ind_mis_reg=None,
                 epsilon_mis_registration=None):
        print('Setting up SPRINT..')
        if dm_input is not None:
            warning('SPRINT: dm_input is no longer needed and is ignored, the mis-registrations are '
                    'applied with obj.dm.apply_mis_registration.')

        # modal basis considered
        self.basis = basis
        self.basis.modes = np.squeeze(self.basis.modes)

        # parameters to identify
        self.ind_mis_reg = select_mis_reg_indices(ind_mis_reg, n_mis_reg)
        self.n_mis_reg = len(self.ind_mis_reg)
        self.mis_reg_fields = [MIS_REG_FIELDS[i] for i in self.ind_mis_reg]

        self.wfs_mis_registered = wfs_mis_registered   # case where the shifts are applied in the WFS space
        self.fast_algorithm = fast_algorithm           # fast version of the algorithm (WARNING: not stable)
        if fast_algorithm:
            warning('SPRINT: fast_algorithm is experimental and currently not reliable '
                    '(interpolated modes do not follow the DM mis-registration conventions).')
        self.recompute_sensitivity = recompute_sensitivity
        self.name_system = '' if nameSystem is None else nameSystem
        self.nameFolder_sensitivity_matrice = self._default_folder(obj) if nameFolder is None else nameFolder

        # zero point for the sensitivity matrices
        if mis_registration_zero_point is None:
            warning('No input mis_registration_zero_point. Using the DM current mis-registration as the zero-point:')
            obj.dm.misReg.print_()
            mis_registration_zero_point = obj.dm.misReg
        self.mis_registration_zero_point = MisRegistration(mis_registration_zero_point)

        # epsilon mis-registration for the computation of the directional gradients
        if epsilon_mis_registration is None:
            epsilon_mis_registration = self.default_epsilon(obj.dm.pitch)
        self.epsilonMisRegistration = epsilon_mis_registration

        # pre-compute the sensitivity matrices
        self.metaMatrix, self.calib_0 = self._compute_sensitivity(obj, self.mis_registration_zero_point, save=True)

        # initial state, restored at every call of estimate
        self._initial_state = (MisRegistration(self.mis_registration_zero_point), self.metaMatrix, self.calib_0)
        print('Done!')

    # ------------------------------------------------------------------------------------------

    @staticmethod
    def default_epsilon(pitch):
        epsilon = MisRegistration()
        epsilon.shiftX = np.round(pitch / 10, 4)
        epsilon.shiftY = np.round(pitch / 10, 4)
        epsilon.rotationAngle = 0.1
        # the magnification also moves the radial/tangential scalings: set it first
        epsilon.magnification = 0.01
        epsilon.radialScaling = 0.01
        epsilon.tangentialScaling = 0.01
        return epsilon

    @staticmethod
    def _default_folder(obj):
        param = getattr(obj, 'param', None)
        try:
            return param['pathInput'] + '/' + param['name'] + '/s_mat/'
        except (TypeError, KeyError):
            warning('No nameFolder provided (and no obj.param): the sensitivity matrices will not be saved.')
            return None

    def _compute_sensitivity(self, obj, zero_point, save):
        return computeMetaSensitivityMatrix(tel=obj.tel,
                                            ngs=obj.ngs,
                                            dm_0=obj.dm,
                                            wfs=obj.wfs,
                                            basis=self.basis,
                                            misRegistrationZeroPoint=zero_point,
                                            epsilonMisRegistration=self.epsilonMisRegistration,
                                            nameFolder=self.nameFolder_sensitivity_matrice,
                                            nameSystem=self.name_system,
                                            save_sensitivity_matrices=save,
                                            recompute_sensitivity=self.recompute_sensitivity,
                                            fast=self.fast_algorithm,
                                            atm=getattr(obj, 'atm', None),
                                            n_mis_reg=self.n_mis_reg,
                                            ind_mis_reg=self.ind_mis_reg)

    # ------------------------------------------------------------------------------------------

    def estimate(self,
                 obj,
                 on_sky_slopes,
                 n_iteration=3,
                 n_update_zero_point=0,
                 precision=3,
                 gain_estimation=1,
                 dm_input=None,
                 tolerance=1/50,
                 display=True,
                 plot=True):
        """Estimate the mis-registration parameters.

        Parameters
        ----------
        obj : object holding tel, ngs, dm, wfs (and atm for the fast algorithm).
        on_sky_slopes : np.ndarray
            WFS signals of the modes of the basis, [n_signal, n_modes] (or [n_signal] for one mode).
        n_iteration : int
            Number of iterations for each zero point.
        n_update_zero_point : int
            Number of times the sensitivity matrices are re-computed around the current estimate.
        precision, gain_estimation, tolerance :
            See estimateMisRegistration.
        dm_input : deprecated, ignored.

        Example
        -------
            Sprint.estimate(obj, my_wfs_signal, n_iteration=3)
            Sprint.mis_registration_out.shiftX         # [m]
            Sprint.mis_registration_out.shiftY         # [m]
            Sprint.mis_registration_out.rotationAngle  # [deg]

        Returns
        -------
        MisRegistration (also stored in self.mis_registration_out)
        """
        if dm_input is not None:
            warning('SPRINT.estimate: dm_input is no longer needed and is ignored.')

        calib_in = CalibrationVault(on_sky_slopes, invert=False)

        # restart from the initial working point
        zero_point, self.metaMatrix, self.calib_0 = self._initial_state
        self.mis_registration_zero_point = MisRegistration(zero_point)
        print('SPRINT was setup around the following working point:')
        self.mis_registration_zero_point.print_()
        self.mis_registration_buffer = None

        for i_update in range(n_update_zero_point + 1):
            if i_update > 0:
                print('----------------------------------')
                print('Mis-Registrations Intermediate Value:')
                self.mis_registration_out.print_()
                print('----------------------------------')
                print('Updating the set of sensitivity matrices...', end=' ')
                self.mis_registration_zero_point = MisRegistration(self.mis_registration_out)
                self.metaMatrix, self.calib_0 = self._compute_sensitivity(obj, self.mis_registration_zero_point, save=False)
                print('Done!')

            (self.mis_registration_out,
             self.scaling_factor,
             self.mis_registration_buffer,
             self.validity_flag,
             self.calib_last) = estimateMisRegistration(tel=obj.tel,
                                                        ngs=obj.ngs,
                                                        dm_0=obj.dm,
                                                        wfs=obj.wfs,
                                                        basis=self.basis,
                                                        calib_in=calib_in,
                                                        misRegistrationZeroPoint=self.mis_registration_zero_point,
                                                        epsilonMisRegistration=self.epsilonMisRegistration,
                                                        calib_0=self.calib_0,
                                                        sensitivity_matrices=self.metaMatrix,
                                                        nIteration=n_iteration,
                                                        precision=precision,
                                                        gainEstimation=gain_estimation,
                                                        tolerance=tolerance,
                                                        fast=self.fast_algorithm,
                                                        atm=getattr(obj, 'atm', None),
                                                        wfs_mis_registrated=self.wfs_mis_registered,
                                                        ind_mis_reg=self.ind_mis_reg,
                                                        previous_estimate=self.mis_registration_buffer,
                                                        return_all=True,
                                                        display=display,
                                                        plot=plot)

        print('----------------------------------')
        print('Final Mis-Registrations identified:')
        self.mis_registration_out.print_()
        print('Mis-registration Validity Flag: ' + str(self.validity_flag))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return self.mis_registration_out