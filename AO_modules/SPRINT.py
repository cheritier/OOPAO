# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:30:51 2021

@author: cheritie
"""

from AO_modules.mis_registration_identification_algorithm.estimateMisRegistration import estimateMisRegistration
from AO_modules.mis_registration_identification_algorithm.computeMetaSensitivyMatrix import computeMetaSensitivityMatrix
from AO_modules.MisRegistration import MisRegistration
from AO_modules.calibration.CalibrationVault import calibrationVault
import numpy as np

class SPRINT:
    def __init__(self,obj, basis, nameFolder = None, nameSystem = None, mis_registration_zero_point = None, wfs_mis_registered= None, fast_algorithm = False, n_iteration = 3):
        print('Setting up SPRINT..')
        # modal basis considered
        self.basis              = basis
        # consider the case when only one signal is used
        if len(basis.indexModes)==1:
            self.basis.indexModes    = [basis.indexModes,basis.indexModes]
            self.basis.modes         = np.asarray([basis.modes,basis.modes]).T           
        
        # Case where the shifts are applied in the WFS space            
        self.wfs_mis_registered = wfs_mis_registered
        # fast version of the algorithm (WARNING: not stable)
        self.fast_algorithm     = fast_algorithm
        
        # zero point for the sensitivity matrices
        if mis_registration_zero_point is None:
            self.mis_registration_zero_point = MisRegistration()
        else:
            self.mis_registration_zero_point = mis_registration_zero_point

        # epsilon mis-registration for the computation of the directional gradients
        self.epsilonMisRegistration                  = MisRegistration()
        self.epsilonMisRegistration.shiftX           = np.round(obj.dm.pitch /10,4)
        self.epsilonMisRegistration.shiftY           = np.round(obj.dm.pitch /10,4)
        self.epsilonMisRegistration.rotationAngle    = np.round(np.rad2deg(np.arctan(self.epsilonMisRegistration.shiftX)/(obj.tel.D/2)),4)
        
        # folder name to save the sensitivity matrices
        if nameFolder is None:
            self.nameFolder_sensitivity_matrice = obj.param['pathInput'] +'/'+ obj.param['name']+'/s_mat/'
        else:
            self.nameFolder_sensitivity_matrice = nameFolder
        
        # name of the system considered
        if nameSystem is None:
            self.name_system = ''
        else:
            self.name_system = nameSystem

        # pre-compute the sensitivity matrices
        [self.metaMatrix,self.calib_0] = computeMetaSensitivityMatrix(nameFolder    = self.nameFolder_sensitivity_matrice,\
                                                         nameSystem                 = self.name_system,\
                                                         tel                        = obj.tel,\
                                                         atm                        = obj.atm,\
                                                         ngs                        = obj.ngs,\
                                                         dm_0                       = obj.dm,\
                                                         pitch                      = obj.dm.pitch,\
                                                         wfs                        = obj.wfs,\
                                                         basis                      = basis,\
                                                         misRegistrationZeroPoint   = self.mis_registration_zero_point,\
                                                         epsilonMisRegistration     = self.epsilonMisRegistration,\
                                                         param                      = obj.param,\
                                                         wfs_mis_registrated        = wfs_mis_registered)
        
        
        print('Done!')

    def estimate(self,obj,on_sky_slopes, n_iteration = 3):
        """
        Method of SPRINT to estimate the mis-registrations parameters
            - obj           : a class containing the different objects, tel, dm, atm, ngs and wfs
            _ on_sky_slopes : the wfs signal used to identify the mis-registration parameters
            _ n_iteration   : the number of iterations to consider
            
        exemple: 
            Sprint.estimate(obj, my_wfs_signal, n_iteration = 3)
        The estimation are available using:
            Sprint.mis_registration_out.shift_x  ---- shift in m
            Sprint.mis_registration_out.shift_y  ---- shift in m 
            Sprint.mis_registration_out.rotation ---- rotation in degree
        """
        if np.ndim(on_sky_slopes)==1:
            calib_misReg_in = calibrationVault(np.squeeze(np.asarray([on_sky_slopes.T,on_sky_slopes.T]).T))
        else:
            calib_misReg_in = calibrationVault(on_sky_slopes)
            

        [self.mis_registration_out ,self.scaling_factor ,self.mis_registration_buffer] = estimateMisRegistration( nameFolder                 = self.nameFolder_sensitivity_matrice,\
                                                             nameSystem                 = self.name_system,\
                                                             tel                        = obj.tel,\
                                                             atm                        = obj.atm,\
                                                             ngs                        = obj.ngs,\
                                                             dm_0                       = obj.dm,\
                                                             calib_in                   = calib_misReg_in,\
                                                             wfs                        = obj.wfs,\
                                                             basis                      = self.basis,\
                                                             misRegistrationZeroPoint   = self.mis_registration_zero_point,\
                                                             epsilonMisRegistration     = self.epsilonMisRegistration,\
                                                             param                      = obj.param,\
                                                             precision                  = 5,\
                                                             return_all                 = True,\
                                                             nIteration                 = n_iteration,\
                                                             fast                       = self.fast_algorithm,\
                                                             wfs_mis_registrated        = self.wfs_mis_registered,\
                                                             sensitivity_matrices       = self.metaMatrix )
        