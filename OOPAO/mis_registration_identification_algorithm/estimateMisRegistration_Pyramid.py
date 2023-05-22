# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:01:00 2022

@author: cheritie
"""


import matplotlib.pyplot as plt
import numpy             as np 
import scipy.ndimage as sp

from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.MisRegistration_Pyramid   import MisRegistration_Pyramid
from OOPAO.mis_registration_identification_algorithm.computeMetaSensitivityMatrix_Pyramid           import computeMetaSensitivityMatrix_Pyramid

"""
def estimateMisRegistration(nameFolder, nameSystem, tel, atm, ngs, dm_0, wfs, basis, calib_in, misRegistrationZeroPoint, epsilonMisRegistration, param, precision = 3, gainEstimation = 1, return_all = False):

    Compute the set of sensitivity matrices required to identify the mis-registrations. 
    
    %%%%%%%%%%%%%%%%   -- INPUTS -- %%%%%%%%%%%%%%%%
    _ nameFolder                : folder to store the sensitivity matrices.
    _ nameSystem                : name of the AO system considered. For instance 'ELT_96x96_R_band'
    _ tel                       : telescope object
    _ atm                       : atmosphere object
    _ ngs                       : source object
    _ dm_0                      : deformable mirror with reference configuration of mis-registrations
    _ pitch                     : pitch of the dm in [m]
    _ wfs                       : wfs object   
    _ basis                     : basis to use to compute the sensitivity matrices. Basis should be an object with the following fields: 
            
                    basis.modes      : [nActuator x nModes] matrix containing the commands to apply the modal basis on the dm
                    basis.indexModes : indexes of the modes considered in the basis. This is used to name the sensitivity matrices 
                    basis.extra      :  extra name to name the sensitivity matrices for instance 'KL'
                    
    _ precision                 : precision to round the parameter estimation. Equivalent to np.round(misReg_estimation,precision)
    _ gainEstimation            : gain to apply after one estimation. eventually allows to avoid overshoots. 
    _ return_all                : if true, returns all the estimations at every step of the algorithm                    
    _ misRegistrationZeroPoint  : mis-registration around which you want to compute the sensitivity matrices
    _ epsilonMisRegistration    : epsilon value to apply 
    _ param                     : dictionnary used as parameter file
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    The function returns the meta-sensitivity matrix that contains all the individual sensitivity matrices reshaped as a vector and concatenated. 
    
    %%%%%%%%%%%%%%%%   -- OUTPUTS -- %%%%%%%%%%%%%%%%
    _ misRegistration_out       : mis-registration object corresponding to the convergence value
    _ scalingFactor_values      : scaling factor (for each mode of the modal basis) for each iteration to take into consideration eventual gains variation between data and model
    _ misRegistration_values    : mis-registration values for each iteration
    
    
"""
def estimateMisRegistration_Pyramid(nameFolder, nameSystem, tel, atm, ngs, dm_0, wfs, basis, calib_in, misRegistrationZeroPoint, epsilonMisRegistration, param, precision = 3, gainEstimation = 1, sensitivity_matrices = None, return_all = False, fast = False, wfs_mis_registrated = None, nIteration = 3,ind = None,n_max = 1):
    
    #%%  ---------- LOAD/COMPUTE SENSITIVITY MATRICES --------------------
    # compute the sensitivity matrices. if the data already exits, the files will be loaded
    
    # WARNING: The data are loaded only if the name of the requeste files matches the ones in argument of this function.
    # make sure that these files are well corresponding to the system you are working with.
    
    if sensitivity_matrices is None:
        [metaMatrix,calib_0] = computeMetaSensitivityMatrix_Pyramid(nameFolder                 = nameFolder,\
                                         nameSystem                 = nameSystem,\
                                         tel                        = tel,\
                                         atm                        = atm,\
                                         ngs                        = ngs,\
                                         dm_0                       = dm_0,\
                                         pitch                      = dm_0.pitch,\
                                         wfs                        = wfs,\
                                         basis                      = basis,\
                                         misRegistrationZeroPoint   = misRegistrationZeroPoint,\
                                         epsilonMisRegistration     = epsilonMisRegistration,\
                                         param                      = param,\
                                         wfs_mis_registrated        = wfs_mis_registrated)
    else:
        metaMatrix = sensitivity_matrices
        
    
    #%%  ---------- ITERATIVE ESTIMATION OF THE PARAMETERS --------------------
    stroke                  = 1e-12
    criteria                = 0
    n_mis_reg               = metaMatrix.M.shape[0]
    misRegEstBuffer         = np.zeros(n_mis_reg)
    scalingFactor_values    = [1]
    misRegistration_values  = [np.zeros(n_mis_reg)]
    if ind is None:
        ind = np.arange(n_mis_reg)
    
    epsilonMisRegistration_f  = ['dX_1','dY_1','dX_2','dY_2','dX_3','dY_3','dX_4','dY_4']
    epsilonMisRegistration_field  = []
    [epsilonMisRegistration_field.append(epsilonMisRegistration_f[i]) for i in ind]


    i_iter=0
    tel.isPaired = False
    misRegistration_out = MisRegistration_Pyramid()+misRegistrationZeroPoint
    

    while criteria ==0:
        i_iter=i_iter+1
        # temporary deformable mirror
        wfs.apply_shift_wfs(sx=[misRegistration_out.dX_1,misRegistration_out.dX_2,misRegistration_out.dX_3,misRegistration_out.dX_4],sy=[misRegistration_out.dY_1,misRegistration_out.dY_2,misRegistration_out.dY_3,misRegistration_out.dY_4])
        # temporary interaction matrix
        calib_tmp =  InteractionMatrix(ngs,atm,tel,dm_0,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=50,invert=False,print_time=False)
        # temporary scaling factor            
        try:
            scalingFactor_tmp   = np.round(np.diag(calib_tmp.D.T@calib_in.D)/ np.diag(calib_tmp.D.T@calib_tmp.D),precision)
            contain_nan = False
            for i in scalingFactor_tmp:
                if(np.isnan(i)):
                    contain_nan = True
                    break
            if contain_nan:
                print('Nan Warning !!')
                scalingFactor_tmp[:] = 1 
            # temporary mis-registration
            misReg_tmp          = gainEstimation*np.matmul(metaMatrix.M,np.reshape( calib_in.D@np.diag(1/scalingFactor_tmp) - calib_tmp.D ,calib_in.D.shape[0]*calib_in.D.shape[1]))

        except:
            scalingFactor_tmp = np.round(np.sum(np.squeeze(calib_tmp.D)*np.squeeze(calib_in.D))/ np.sum(np.squeeze(calib_tmp.D)*np.squeeze(calib_tmp.D)),precision)    
            contain_nan = False
            if np.isnan(scalingFactor_tmp):
                scalingFactor_tmp = 1
                contain_nan = True
                print('Nan Warning !!')
            # temporary mis-registration 
            misReg_tmp          = gainEstimation*np.matmul(metaMatrix.M,np.squeeze((np.squeeze(calib_in.D)*(1/scalingFactor_tmp)) - np.squeeze(calib_tmp.D)))
        # cumulative mis-registration
        misRegEstBuffer+= np.round(misReg_tmp,precision)
        
        # define the next working point to adjust the scaling factor
        if contain_nan is False:
            for i_mis_reg in range(n_mis_reg):
                val = (getattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg]) + np.round(misReg_tmp[i_mis_reg],precision))
                if val>=0:
                    val=val%n_max
                else:
                    val = -(n_max-val%n_max)
                    
                setattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg],val)
                        
        
        # save the data for each iteration
        scalingFactor_values.append(np.copy(scalingFactor_tmp))
        misRegistration_values.append(np.copy(misRegEstBuffer))
        

        if i_iter==nIteration:
            criteria =1

    misRegistration_out.dX_1       = np.round(misRegistration_out.dX_1,precision)
    misRegistration_out.dX_2       = np.round(misRegistration_out.dX_2,precision)
    misRegistration_out.dX_3       = np.round(misRegistration_out.dX_3,precision)
    misRegistration_out.dX_4       = np.round(misRegistration_out.dX_4,precision)
    misRegistration_out.dY_1       = np.round(misRegistration_out.dY_1,precision)
    misRegistration_out.dY_2       = np.round(misRegistration_out.dY_2,precision)
    misRegistration_out.dY_3       = np.round(misRegistration_out.dY_3,precision)
    misRegistration_out.dY_4       = np.round(misRegistration_out.dY_4,precision)

    # values for validity
    
    tolerance = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]                
    
    diff      =  np.abs(misRegistration_values[-1]-misRegistration_values[-2])
    
    # in case of nan
    diff[np.where(np.isnan(diff))] = 10000
    if np.argwhere(diff-tolerance[:n_mis_reg]>0).size==0 or contain_nan is False:    
        # validity of the mis-reg
        validity_flag = True
    else:
        validity_flag = False
        misRegistration_out.dX_1       = 0*np.round(misRegistration_out.dX_1,precision)
        misRegistration_out.dX_2       = 0*np.round(misRegistration_out.dX_2,precision)
        misRegistration_out.dX_3       = 0*np.round(misRegistration_out.dX_3,precision)
        misRegistration_out.dX_4       = 0*np.round(misRegistration_out.dX_4,precision)
        misRegistration_out.dY_1       = 0*np.round(misRegistration_out.dY_1,precision)
        misRegistration_out.dY_2       = 0*np.round(misRegistration_out.dY_2,precision)
        misRegistration_out.dY_3       = 0*np.round(misRegistration_out.dY_3,precision)
        misRegistration_out.dY_4       = 0*np.round(misRegistration_out.dY_4,precision)
    
    if return_all:
        return misRegistration_out, scalingFactor_values, misRegistration_values,validity_flag
    else:
        return misRegistration_out

