# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:51:25 2021

@author: cheritie
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:35:26 2020

@author: cheritie
"""
import matplotlib.pyplot as plt
import numpy             as np 
import scipy.ndimage as sp

from AO_modules.calibration.InteractionMatrix import interactionMatrix
from AO_modules.MisRegistration   import MisRegistration

from AO_modules.mis_registration_identification_algorithm.computeMetaSensitivyMatrix_full           import computeMetaSensitivityMatrix
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration                 import applyMisRegistration, apply_shift_wfs
from AO_modules.tools.interpolateGeometricalTransformation                                     import rotateImageMatrix,rotation,translationImageMatrix,translation,anamorphosis,anamorphosisImageMatrix

import skimage.transform as sk
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
def estimateMisRegistration(nameFolder, nameSystem, tel, atm, ngs, dm_0, wfs, basis, calib_in, misRegistrationZeroPoint, epsilonMisRegistration, param, precision = 3, gainEstimation = 1, sensitivity_matrices = None, return_all = False, fast = False,nIteration = 3, wfs_mis_registrated = None, extra_dm_mis_registration=None):
    
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
                                         wfs_mis_registrated        = wfs_mis_registrated,\
                                         extra_dm_mis_registration = extra_dm_mis_registration)
    else:
        metaMatrix = sensitivity_matrices
        
    
    #%%  ---------- ITERATIVE ESTIMATION OF THE PARAMETERS --------------------
    stroke                  = 1e-12
    criteria                = 0
    misRegEstBuffer         = np.zeros(5)
    scalingFactor_values    = []
    misRegistration_values  = []
    i=0
    tel-atm
    misRegistration_out = MisRegistration(misRegistrationZeroPoint)
    
    if fast:
        from AO_modules.calibration.InteractionMatrix import interactionMatrixFromPhaseScreen

        dm_0.coefs = basis.modes
        tel*dm_0
        input_modes_0 = tel.OPD
        input_modes_cp = input_modes_0.copy()
        while criteria ==0:
            i=i+1
            # temporary deformable mirror
            for i_modes in range(input_modes_0.shape[2]):
                if wfs_mis_registrated is None:

                    misRegistration_wfs                 = MisRegistration()
                    misRegistration_wfs.shiftX          = misRegistration_out.shiftX
                    misRegistration_wfs.shiftY          = misRegistration_out.shiftY
                    
                    misRegistration_dm                      = MisRegistration()
                    misRegistration_dm.rotationAngle        = misRegistration_out.rotationAngle
                    misRegistration_dm.tangentialScaling    = misRegistration_out.tangentialScaling
                    misRegistration_dm.radialScaling        = misRegistration_out.radialScaling
                    
                    apply_shift_wfs(wfs, misRegistration_wfs.shiftX / (wfs.nSubap/wfs.telescope.D), misRegistration_wfs.shiftY/ (wfs.nSubap/wfs.telescope.D))
                    
                    input_modes_cp[:,:,i_modes] = apply_mis_reg(tel,input_modes_0[:,:,i_modes], misRegistration_dm)   
                else:
                    input_modes_cp[:,:,i_modes] = apply_mis_reg(tel,input_modes_0[:,:,i_modes], misRegistration_out)   
            
        
            # temporary interaction matrix
            calib_tmp =  interactionMatrixFromPhaseScreen(ngs,atm,tel,wfs,input_modes_cp,stroke,phaseOffset=0,nMeasurements=50,invert = False)
            # temporary scaling factor    
            scalingFactor_tmp   = np.round(np.diag(calib_tmp.D.T@calib_in.D)/ np.diag(calib_tmp.D.T@calib_tmp.D),precision)
            
            # temporary mis-registration
            misReg_tmp          = gainEstimation*np.matmul(metaMatrix.M,np.reshape( calib_in.D@np.diag(1/scalingFactor_tmp) - calib_tmp.D ,calib_in.D.shape[0]*calib_in.D.shape[1]))
            
            # cumulative mis-registration
            misRegEstBuffer+= np.round(misReg_tmp,precision)
            
            # define the next working point to adjust the scaling factor
            misRegistration_out.rotationAngle        += np.round(misReg_tmp[0],precision)
            misRegistration_out.shiftX               += np.round(misReg_tmp[1],precision)
            misRegistration_out.shiftY               += np.round(misReg_tmp[2],precision)
            misRegistration_out.tangentialScaling    += np.round(misReg_tmp[3],precision)
            misRegistration_out.radialScaling        += np.round(misReg_tmp[4],precision)           # save the data for each iteration

            scalingFactor_values.append(np.copy(scalingFactor_tmp))
            misRegistration_values.append(np.copy(misRegEstBuffer))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Mis-Registrations identified:')
            print('Rotation [deg] \t Shift X [m] \t Shift Y [m]')
            print(str(misRegistration_out.rotationAngle)   + '\t\t' +str(misRegistration_out.shiftX)+'\t\t' + str(misRegistration_out.shiftY)+ '\t\t' + str(misRegistration_out.tangentialScaling)+ '\t\t' +str(misRegistration_out.radialScaling))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            if i==nIteration:
                criteria =1

    else:
        while criteria ==0:
            i=i+1
            # temporary deformable mirror
            dm_tmp = applyMisRegistration(tel,misRegistration_out,param, wfs = wfs_mis_registrated, extra_dm_mis_registration = extra_dm_mis_registration)
        
            # temporary interaction matrix
            calib_tmp =  interactionMatrix(ngs,atm,tel,dm_tmp,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=50, invert = False)
            # erase dm_tmp to free memory
            del dm_tmp
            # temporary scaling factor    
            scalingFactor_tmp   = np.round(np.diag(calib_tmp.D.T@calib_in.D)/ np.diag(calib_tmp.D.T@calib_tmp.D),precision)
            
            # temporary mis-registration
            misReg_tmp          = gainEstimation*np.matmul(metaMatrix.M,np.reshape( calib_in.D@np.diag(1/scalingFactor_tmp) - calib_tmp.D ,calib_in.D.shape[0]*calib_in.D.shape[1]))

            # cumulative mis-registration
            misRegEstBuffer+= np.round(misReg_tmp,precision)
            
            # define the next working point to adjust the scaling factor
            misRegistration_out.rotationAngle        += np.round(misReg_tmp[0],precision)
            misRegistration_out.shiftX               += np.round(misReg_tmp[1],precision)
            misRegistration_out.shiftY               += np.round(misReg_tmp[2],precision)
            misRegistration_out.tangentialScaling    += np.round(misReg_tmp[3],precision)
            misRegistration_out.radialScaling        += np.round(misReg_tmp[4],precision)
            
            # save the data for each iteration
            scalingFactor_values.append(np.copy(scalingFactor_tmp))
            misRegistration_values.append(np.copy(misRegEstBuffer))
            
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Mis-Registrations identified:')
            print('Rotation [deg] \t Shift X [m] \t Shift Y [m]')
            print(str(misRegistration_out.rotationAngle)   + '\t\t' +str(misRegistration_out.shiftX)+'\t\t' + str(misRegistration_out.shiftY)+ '\t\t' + str(misRegistration_out.tangentialScaling)+ '\t\t' +str(misRegistration_out.radialScaling))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            if i==nIteration:
                criteria =1

    if return_all:
        return misRegistration_out, scalingFactor_values, misRegistration_values
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

