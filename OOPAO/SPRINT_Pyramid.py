# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:59:56 2022

@author: cheritie
"""


from OOPAO.mis_registration_identification_algorithm.estimateMisRegistration_Pyramid import estimateMisRegistration_Pyramid
from OOPAO.mis_registration_identification_algorithm.computeMetaSensitivityMatrix_Pyramid import computeMetaSensitivityMatrix_Pyramid
from OOPAO.MisRegistration_Pyramid import MisRegistration_Pyramid
from OOPAO.calibration.CalibrationVault import CalibrationVault
import numpy as np

class SPRINT_Pyramid:
    def __init__(self, obj, basis,ind, nameFolder = None, nameSystem = None, mis_registration_zero_point = None, wfs_mis_registered= None, fast_algorithm = False, n_mis_reg = 3, recompute_sensitivity = False):
        print('Setting up SPRINT..')
        # modal basis considered
        self.basis              = basis
        self.n_mis_reg          = n_mis_reg
        self.recompute_sensitivity = recompute_sensitivity         
        
        # Case where the shifts are applied in the WFS space            
        self.wfs_mis_registered = wfs_mis_registered
        # fast version of the algorithm (WARNING: not stable)
        self.fast_algorithm     = fast_algorithm
        self.ind = ind
        # zero point for the sensitivity matrices
        if mis_registration_zero_point is None:
            self.mis_registration_zero_point = MisRegistration_Pyramid()
        else:
            self.mis_registration_zero_point = mis_registration_zero_point

        # epsilon mis-registration for the computation of the directional gradients
        self.epsilonMisRegistration                   = 1
        
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
        [self.metaMatrix,self.calib_0] = computeMetaSensitivityMatrix_Pyramid(nameFolder    = self.nameFolder_sensitivity_matrice,\
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
                                                         wfs_mis_registrated        = wfs_mis_registered,\
                                                         n_mis_reg                  = self.n_mis_reg,\
                                                         fast                       = self.fast_algorithm,\
                                                         recompute_sensitivity      = self.recompute_sensitivity, ind=self.ind    )

            
        self.metaMatrix_init = CalibrationVault(self.metaMatrix.D)
        self.mis_registration_zero_point_init = self.mis_registration_zero_point
        
        print('Done!')

    def estimate(self,obj,on_sky_slopes, n_iteration = 3, n_update_zero_point = 0 ,precision = 3, gain_estimation = 1,n_max=1):
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

        calib_misReg_in = CalibrationVault(on_sky_slopes,invert = False)
        # reinitialize the  meta matrix
        self.metaMatrix = CalibrationVault(self.metaMatrix_init.D)
        self.mis_registration_zero_point = self.mis_registration_zero_point_init 
        
        for i_update in range(n_update_zero_point+1):
            if i_update>0:
                print('----------------------------------')

                print('Mis-Registrations Intermediate Value:')
                self.mis_registration_out.print_()
                print('----------------------------------')

                print('Updating the set of sensitivity matrices...',end=' ')
                # update zero point:
                self.mis_registration_zero_point = self.mis_registration_out
                # pre-compute the sensitivity matrices
                [self.metaMatrix,self.calib_0] = computeMetaSensitivityMatrix_Pyramid(nameFolder    = self.nameFolder_sensitivity_matrice,\
                                                                 nameSystem                 = self.name_system,\
                                                                 tel                        = obj.tel,\
                                                                 atm                        = obj.atm,\
                                                                 ngs                        = obj.ngs,\
                                                                 dm_0                       = obj.dm,\
                                                                 pitch                      = obj.dm.pitch,\
                                                                 wfs                        = obj.wfs,\
                                                                 basis                      = self.basis,\
                                                                 misRegistrationZeroPoint   = self.mis_registration_zero_point,\
                                                                 epsilonMisRegistration     = self.epsilonMisRegistration,\
                                                                 param                      = obj.param,\
                                                                 wfs_mis_registrated        = self.wfs_mis_registered,\
                                                                 save_sensitivity_matrices  = False,\
                                                                 n_mis_reg                  = self.n_mis_reg,\
                                                                 fast                       = self.fast_algorithm,\
                                                                 recompute_sensitivity      = self.recompute_sensitivity, ind = self.ind    )

                print('Done!')

            # estimate mis-registrations
            [self.mis_registration_out ,self.scaling_factor ,self.mis_registration_buffer,self.validity_flag] = estimateMisRegistration_Pyramid( nameFolder  = self.nameFolder_sensitivity_matrice,\
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
                                                             return_all                 = True,\
                                                             nIteration                 = n_iteration,\
                                                             fast                       = self.fast_algorithm,\
                                                             wfs_mis_registrated        = self.wfs_mis_registered,\
                                                             sensitivity_matrices       = self.metaMatrix,\
                                                             precision                  = precision,\
                                                             gainEstimation             = gain_estimation, ind = self.ind,n_max =n_max)
                
                
        print('----------------------------------')

        print('Final Mis-Registrations identified:')
        self.mis_registration_out.print_()
        print('Mis-registration Validity Flag: '+ str(self.validity_flag))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
