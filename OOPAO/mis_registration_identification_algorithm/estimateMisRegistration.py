# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:35:26 2020

@author: cheritie
"""

import numpy as np
import skimage.transform as sk

from ..MisRegistration import MisRegistration
from ..calibration.InteractionMatrix import InteractionMatrix, InteractionMatrixFromPhaseScreen
from ..mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
from ..mis_registration_identification_algorithm.computeMetaSensitivyMatrix import computeMetaSensitivityMatrix
from ..tools.interpolateGeometricalTransformation import (anamorphosisImageMatrix,
                                                          rotateImageMatrix,
                                                          translationImageMatrix)

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


def estimateMisRegistration(nameFolder, nameSystem, tel, atm, ngs, dm_0, wfs, basis, calib_in, misRegistrationZeroPoint, epsilonMisRegistration, param, precision = 3, gainEstimation = 1, sensitivity_matrices = None, return_all = False, fast = False, wfs_mis_registrated = None, nIteration = 3):
    
    #%%  ---------- LOAD/COMPUTE SENSITIVITY MATRICES --------------------
    # compute the sensitivity matrices. if the data already exits, the files will be loaded
    
    # WARNING: The data are loaded only if the name of the requeste files matches the ones in argument of this function.
    # make sure that these files are well corresponding to the system you are working with.
    
    if sensitivity_matrices is None:
        [metaMatrix,calib_0] = computeMetaSensitivityMatrix(nameFolder                 = nameFolder,\
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
    
    epsilonMisRegistration_field = ['shiftX','shiftY','rotationAngle','radialScaling','tangentialScaling']

    i_iter=0
    tel.isPaired = False
    misRegistration_out = MisRegistration(misRegistrationZeroPoint)
    
    if fast:

        dm_0.coefs = np.squeeze(basis.modes)
        tel*dm_0
        input_modes_0 = dm_0.OPD
        input_modes_cp = input_modes_0.copy()
        while criteria ==0:
            i_iter=i_iter+1
            # temporary deformable mirror
            if np.ndim(input_modes_0)==2:
                if wfs_mis_registrated is not None:
                    misRegistration_wfs                 = MisRegistration()
                    misRegistration_wfs.shiftX          = misRegistration_out.shiftX
                    misRegistration_wfs.shiftY          = misRegistration_out.shiftY
                    
                    misRegistration_dm               = MisRegistration()
                    misRegistration_dm.rotationAngle = misRegistration_out.rotationAngle
                    
                    wfs.apply_shift_wfs( misRegistration_wfs.shiftX , misRegistration_wfs.shiftY)
                    
                    input_modes_cp =  tel.pupil*apply_mis_reg(tel,input_modes_0, misRegistration_dm)   
                else:
                    input_modes_cp = tel.pupil*apply_mis_reg(tel,input_modes_0, misRegistration_out) 
            else:
                    
                for i_modes in range(input_modes_0.shape[2]):
                    if wfs_mis_registrated is not None:
    
                        misRegistration_wfs                 = MisRegistration()
                        misRegistration_wfs.shiftX          = misRegistration_out.shiftX
                        misRegistration_wfs.shiftY          = misRegistration_out.shiftY
                        
                        misRegistration_dm               = MisRegistration()
                        misRegistration_dm.rotationAngle = misRegistration_out.rotationAngle
                        
                        wfs.apply_shift_wfs(misRegistration_wfs.shiftX, misRegistration_wfs.shiftY)
                        
                        input_modes_cp[:,:,i_modes] = tel.pupil*apply_mis_reg(tel,input_modes_0[:,:,i_modes], misRegistration_dm)   
                    else:
                        input_modes_cp[:,:,i_modes] = tel.pupil*apply_mis_reg(tel,input_modes_0[:,:,i_modes], misRegistration_out)   
                
        
            # temporary interaction matrix
            calib_tmp =  InteractionMatrixFromPhaseScreen(ngs,atm,tel,wfs,input_modes_cp,stroke,phaseOffset=0,nMeasurements=5,invert=False,print_time=False)
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
            for i_mis_reg in range(n_mis_reg):
                    setattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg], getattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg]) + np.round(misReg_tmp[i_mis_reg],precision))                    
                            
            # save the data for each iteration
            scalingFactor_values.append(np.copy(scalingFactor_tmp))
            misRegistration_values.append(np.copy(misRegEstBuffer))


            if i_iter==nIteration:
                criteria =1

    else:
        while criteria ==0:
            i_iter=i_iter+1
            # temporary deformable mirror
            dm_tmp = applyMisRegistration(tel,misRegistration_out,param, wfs = wfs_mis_registrated,print_dm_properties=False,floating_precision=dm_0.floating_precision)
        
            # temporary interaction matrix
            calib_tmp =  InteractionMatrix(ngs,atm,tel,dm_tmp,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=50,invert=False,print_time=False)
            # erase dm_tmp to free memory
            del dm_tmp
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
                    setattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg], getattr(misRegistration_out, epsilonMisRegistration_field[i_mis_reg]) + np.round(misReg_tmp[i_mis_reg],precision))
                            
            
            # save the data for each iteration
            scalingFactor_values.append(np.copy(scalingFactor_tmp))
            misRegistration_values.append(np.copy(misRegEstBuffer))
            

            if i_iter==nIteration:
                criteria =1

    misRegistration_out.shiftX              = np.round(misRegistration_out.shiftX,precision)
    misRegistration_out.shiftY              = np.round(misRegistration_out.shiftY,precision)
    misRegistration_out.rotationAngle       = np.round(misRegistration_out.rotationAngle,precision)
    misRegistration_out.radialScaling       = np.round(misRegistration_out.radialScaling,precision)
    misRegistration_out.tangentialScaling   = np.round(misRegistration_out.tangentialScaling,precision)

    # values for validity
    
    tolerance = [dm_0.pitch/50,dm_0.pitch/50,np.rad2deg(np.arctan((dm_0.pitch/50)/(tel.D/2))),0.05,0.05]                
    
    diff      =  np.abs(misRegistration_values[-1]-misRegistration_values[-2])
    
    # in case of nan
    diff[np.where(np.isnan(diff))] = 10000
    if np.argwhere(diff-tolerance[:n_mis_reg]>0).size==0 or contain_nan:    
        # validity of the mis-reg
        validity_flag = True
    else:
        validity_flag = False
        misRegistration_out.shiftX              = 0*np.round(misRegistration_out.shiftX,precision)
        misRegistration_out.shiftY              = 0*np.round(misRegistration_out.shiftY,precision)
        misRegistration_out.rotationAngle       = 0*np.round(misRegistration_out.rotationAngle,precision)
        misRegistration_out.radialScaling       = 0*np.round(misRegistration_out.radialScaling,precision)
        misRegistration_out.tangentialScaling   = 0*np.round(misRegistration_out.tangentialScaling,precision)
    
    if return_all:
        return misRegistration_out, scalingFactor_values, misRegistration_values,validity_flag
    else:
        return misRegistration_out



def apply_mis_reg(tel,map_2d, misReg):
    pixelsize = tel.D/tel.resolution
    tel.resetOPD()
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(tel.OPD,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
    rotMatrix               = rotateImageMatrix(tel.OPD,misReg.rotationAngle)
    shiftMatrix             = translationImageMatrix(tel.OPD,[misReg.shiftY/pixelsize,misReg.shiftX/pixelsize]) #units are in m
          
    # 3) Global transformation matrix
    transformationMatrix    =  anamMatrix + rotMatrix + shiftMatrix 
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,order=3)
            return output
    out = globalTransformation(map_2d)
    return out

