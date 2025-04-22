# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:35:26 2020

@author: cheritie
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sk
from OOPAO.tools.displayTools import cl_plot
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


def estimateMisRegistration(nameFolder,
                            nameSystem,
                            tel,
                            atm,
                            ngs,
                            dm_0,
                            wfs,
                            basis,
                            calib_in,
                            misRegistrationZeroPoint,
                            epsilonMisRegistration,
                            param,
                            precision=3,
                            gainEstimation=1,
                            sensitivity_matrices=None,
                            return_all=False,
                            fast=False,
                            wfs_mis_registrated=None,
                            nIteration=3,
                            dm_input=None,
                            display=True,
                            tolerance=1/50,
                            plot = True,
                            previous_estimate=None,
                            ind_mis_reg = None):

    # ---------- LOAD/COMPUTE SENSITIVITY MATRICES --------------------
    # compute the sensitivity matrices. if the data already exits, the files will be loaded

    # WARNING: The data are loaded only if the name of the requeste files matches the ones in argument of this function.
    # make sure that these files are well corresponding to the system you are working with.

    if sensitivity_matrices is None:
        [metaMatrix,calib_0] = computeMetaSensitivityMatrix(nameFolder=nameFolder,
                                                            nameSystem=nameSystem,
                                                            tel=tel,
                                                            atm=atm,
                                                            ngs=ngs,
                                                            dm_0=dm_0,
                                                            pitch=dm_0.pitch,
                                                            wfs=wfs,
                                                            basis=basis,
                                                            misRegistrationZeroPoint=misRegistrationZeroPoint,
                                                            epsilonMisRegistration=epsilonMisRegistration,
                                                            param=param,
                                                            wfs_mis_registrated=wfs_mis_registrated,
                                                            ind_mis_reg = ind_mis_reg)
    else:
        metaMatrix = sensitivity_matrices

    #  ---------- ITERATIVE ESTIMATION OF THE PARAMETERS --------------------
    epsilonMisRegistration_field = ['shiftX', 'shiftY', 'rotationAngle', 'radialScaling', 'tangentialScaling']
    epsilonMisRegistration_field = list(np.asarray(epsilonMisRegistration_field)[ind_mis_reg])
    stroke = 1e-12
    criteria = 0
    n_mis_reg = metaMatrix.M.shape[0]
    misRegEstBuffer = np.zeros(n_mis_reg)
    misRegEstBuffer_ref = np.zeros(n_mis_reg)
    for i in range(n_mis_reg):
        misRegEstBuffer_ref[i] = getattr(misRegistrationZeroPoint, epsilonMisRegistration_field[i])

    scalingFactor_values = [np.ones(basis.modes.shape[1])]

    if previous_estimate is None:
        misRegistration_values = [np.zeros(n_mis_reg)]
    else:
        misRegistration_values = previous_estimate    

    
    
    if plot:
        units = ['[m]','[m]','[deg]','[%]','[%]']
        units = list(np.asarray(units)[ind_mis_reg])
        list_title = ['Shift X','Shift Y','Rotation Angle','radialScaling','tangentialScaling']
        list_title = list(np.asarray(list_title)[ind_mis_reg])
        list_label = []
        list_inp = []
        for i_mis_reg in range(n_mis_reg):
            list_inp.append([[0,misRegEstBuffer_ref[i_mis_reg]],[0,misRegEstBuffer_ref[i_mis_reg]]])
            list_label.append(['Iteration Number',units[i_mis_reg]])
        plot_obj = cl_plot(list_fig          = list_inp,
                           type_fig          = ['plot']*(n_mis_reg),
                           list_title        = list_title,
                           list_legend       = [None]*(n_mis_reg),
                           list_label        = list_label,
                           list_lim          = [None]*(n_mis_reg),
                           n_subplot         = [n_mis_reg,1],
                           list_display_axis = [True]*(n_mis_reg),
                           list_ratio        = [[0.95,0.1],[1]*(n_mis_reg)], s=20)
        
        
    i_iter=0
    flag_paired = tel.isPaired
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
            calib_tmp =  InteractionMatrixFromPhaseScreen(ngs,atm,tel,wfs,input_modes_cp,stroke,phaseOffset=0,nMeasurements=1,invert=False,print_time=False)
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
            misRegistration_values.append(np.copy(misRegEstBuffer) + misRegEstBuffer_ref)
            if display:
                print('----------------------------------------------------------------------------')
                misRegistration_out.print_()
            if plot:
                
                list_inp = []
                list_lim = []
                # list_inp.append([np.arange(i_iter+1),np.mean(np.asarray(scalingFactor_values),axis=1)])
                for i_mis_reg in range(n_mis_reg):
                    global_value = np.asarray(misRegistration_values)[1:,i_mis_reg]
                    list_inp.append([np.arange(len(misRegistration_values)-1),
                                     global_value])
                    list_lim.append([0.8*misRegEstBuffer_ref[i_mis_reg],1.2*misRegEstBuffer_ref[i_mis_reg]])

                plot_obj.list_lim = list_lim
                cl_plot(list_fig = list_inp,
                         plt_obj = plot_obj)
                plt.pause(0.01)
                if plot_obj.keep_going is False:
                    break
            if i_iter==nIteration:
                criteria =1

    else:
        while criteria ==0:
            i_iter=i_iter+1
            # temporary deformable mirror
            dm_tmp = applyMisRegistration(tel,misRegistration_out,param, wfs = wfs_mis_registrated,print_dm_properties=False,floating_precision=dm_0.floating_precision,dm_input=dm_input)
            
            # temporary interaction matrix
            calib_tmp =  InteractionMatrix(ngs,atm,tel,dm_tmp,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=1,invert=False,print_time=False)
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
            misRegistration_values.append(np.copy(misRegEstBuffer)+misRegEstBuffer_ref)
            if display:
                misRegistration_out.print_()
            if plot:
                list_inp = []
                list_lim = []
                # list_inp.append([np.arange(i_iter+1),np.mean(np.asarray(scalingFactor_values),axis=1)])
                for i_mis_reg in range(n_mis_reg):
                    global_value = np.asarray(misRegistration_values)[1:,i_mis_reg]
                    list_inp.append([np.arange(len(misRegistration_values)-1),
                                     global_value])
                    list_lim.append([0.8*misRegEstBuffer_ref[i_mis_reg],1.2*misRegEstBuffer_ref[i_mis_reg]])

                plot_obj.list_lim = list_lim
                cl_plot(list_fig = list_inp,
                         plt_obj = plot_obj)
                plt.pause(0.01)
                if plot_obj.keep_going is False:
                    break
                
            if i_iter==nIteration:
                criteria =1

    misRegistration_out.shiftX              = np.round(misRegistration_out.shiftX,precision)
    misRegistration_out.shiftY              = np.round(misRegistration_out.shiftY,precision)
    misRegistration_out.rotationAngle       = np.round(misRegistration_out.rotationAngle,precision)
    misRegistration_out.radialScaling       = np.round(misRegistration_out.radialScaling,precision)
    misRegistration_out.tangentialScaling   = np.round(misRegistration_out.tangentialScaling,precision)

    # values for validity
    
    tolerance = [np.rad2deg(np.arctan((dm_0.pitch*tolerance)/(tel.D/2))),dm_0.pitch*tolerance,dm_0.pitch*tolerance,0.05,0.05]                
    
    diff      =  np.abs(misRegistration_values[-1]-misRegistration_values[-2])
    
    # in case of nan
    diff[np.where(np.isnan(diff))] = 10000
    # print(tolerance)
    # print(diff)
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
    
    tel.isPaired = flag_paired
    # set back the system to its initial working point
    dm_0 = applyMisRegistration(tel,misRegistrationZeroPoint,param, wfs = wfs_mis_registrated,print_dm_properties=False, floating_precision=dm_0.floating_precision, dm_input = dm_input)
    
    if return_all:
        return misRegistration_out, scalingFactor_values, misRegistration_values,validity_flag,calib_tmp
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

