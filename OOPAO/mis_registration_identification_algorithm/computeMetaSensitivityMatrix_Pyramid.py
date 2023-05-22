# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:38:57 2022

@author: cheritie
"""


from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix,InteractionMatrixFromPhaseScreen
from OOPAO.MisRegistration  import MisRegistration
from astropy.io import fits as pfits
import numpy as np
from OOPAO.tools.tools import createFolder
from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
from OOPAO.tools.interpolateGeometricalTransformation                                     import rotateImageMatrix,rotation,translationImageMatrix,translation,anamorphosis,anamorphosisImageMatrix
import skimage.transform as sk
"""
def computeMetaSensitivityMatrix(nameFolder,nameSystem,tel,atm,ngs,dm_0,pitch,wfs,basis,misRegistrationZeroPoint,epsilonMisRegistration,param):

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
                    basis.extra      :  extra name to name the sensitivity matrices for instance 'KL'
    _ misRegistrationZeroPoint  : mis-registration around which you want to compute the sensitivity matrices
    _ epsilonMisRegistration    : epsilon value to apply 
    _ param                     : dictionnary used as parameter file
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    The function returns the meta-sensitivity matrix that contains all the individual sensitivity matrices reshaped as a vector and concatenated. 
    
    %%%%%%%%%%%%%%%%   -- OUTPUTS -- %%%%%%%%%%%%%%%%
    _ metaSensitivityMatrix     : meta-sensitivity matrix stored as a calibration object. the matrix is accessible with metaSensitivityMatrix.D and its pseudo inverse with metaSensitivityMatrix.M
    _ calib_0                   : interaction matrix corresponding to misRegistrationZeroPoint stored as a calibration object.

"""
def computeMetaSensitivityMatrix_Pyramid(nameFolder, nameSystem, tel, atm, ngs, dm_0, pitch, wfs, basis, misRegistrationZeroPoint, epsilonMisRegistration, param, wfs_mis_registrated = None,save_sensitivity_matrices=True,fast = False, n_mis_reg = 8, recompute_sensitivity = False, ind = None):
    #%% --------------------  CREATION OF THE DESTINATION FOLDER --------------------
    homeFolder = misRegistrationZeroPoint.misRegName+'/'
    intMat_name = 'im'
    if basis.modes.shape == np.shape(np.eye(dm_0.nValidAct)):
        
        comparison = basis.modes == np.eye(dm_0.nValidAct)
        if comparison.all():
            foldername  = nameFolder+nameSystem+homeFolder+'zon/'
            extraName   = '' 

        else:
            foldername  = nameFolder+nameSystem+homeFolder+'mod/'
            extraName   = '_'+basis.extra
    else:
        foldername  = nameFolder+nameSystem+homeFolder+'mod/'
        extraName   = '_'+basis.extra  
    
    if save_sensitivity_matrices:
        createFolder(foldername)
    

    
    #%% --------------------  COMPUTATION OF THE INTERACTION MATRICES FOLDER --------------------
    epsilonMisRegistration_n  = ['dX_1','dY_1','dX_2','dY_2','dX_3','dY_3','dX_4','dY_4']
    if ind is None:
        ind = np.arange(n_mis_reg)

    v_x = [[1,0,0,0],\
           [0,0,0,0],\
           [0,1,0,0],\
           [0,0,0,0],\
           [0,0,1,0],\
           [0,0,0,0],\
           [0,0,0,1],\
           [0,0,0,0]]
        
    v_y = [[0,0,0,0],\
           [1,0,0,0],\
           [0,0,0,0],\
           [0,1,0,0],\
           [0,0,0,0],\
           [0,0,1,0],\
           [0,0,0,0],\
           [0,0,0,1]]
        
    val_x = []
    val_y = []    
    epsilonMisRegistration_name = []

    [val_x.append(v_x[i]) for i in ind]
    [val_y.append(v_y[i]) for i in ind]
    [epsilonMisRegistration_name.append(epsilonMisRegistration_n[i]) for i in ind]
    epsilon = epsilonMisRegistration
    try:
        meta_matrix =np.zeros([wfs.nSignal*basis.modes.shape[1],int(len(epsilonMisRegistration_name))])
    except:
        meta_matrix =np.zeros([wfs.nSignal,int(len(epsilonMisRegistration_name))])
            
    for i in range(n_mis_reg):
        # name for the matrices
        name_0 = foldername + intMat_name +'_0'+ extraName+'.fits'
        name_p = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_p_'+str(np.abs(epsilon))+ extraName+'.fits'
        name_n = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_m_'+str(np.abs(epsilon))+ extraName+'.fits'
        
        stroke = 1e-12


    #%% --------------------  CENTERED INTERACTION MATRIX --------------------            
        try:
            if recompute_sensitivity is False:

                hdu = pfits.open(name_0)
                calib_0 = CalibrationVault(hdu[1].data,invert=False)
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('WARNING: you are loading existing data from \n')
                print(str(name_0)+'/n')
                print('Make sure that the loaded data correspond to your AO system!')
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            else:
                hdu = pfits.open(name_0+'_volontary_error')                

        except:
            calib_0 = InteractionMatrix(ngs, atm, tel, dm_0, wfs, basis.modes ,stroke, phaseOffset=0, nMeasurements=50,invert=False,print_time=False)
            
            # save output in fits file
            if save_sensitivity_matrices:
                hdr=pfits.Header()
                hdr['TITLE']    = 'INTERACTION MATRIX - INITIAL POINT'
                empty_primary   = pfits.PrimaryHDU(header=hdr)
                primary_hdu     = pfits.ImageHDU(calib_0.D)
                hdu             = pfits.HDUList([empty_primary, primary_hdu])
                hdu.writeto(name_0,overwrite=True)

    #%% --------------------  POSITIVE MIS-REGISTRATION --------------------
        try:
            if recompute_sensitivity is False:
                hdu = pfits.open(name_p)
                calib_tmp_p = CalibrationVault(hdu[1].data,invert=False)
            else:
                hdu = pfits.open(name_0+'_volontary_error')       
        except:
            value_x = []
            value_y = []

            [value_x.append(val_x[i][j]*epsilon + getattr(misRegistrationZeroPoint,epsilonMisRegistration_n[j])) for j in range(4)] 
            [value_y.append(val_y[i][j]*epsilon + getattr(misRegistrationZeroPoint,epsilonMisRegistration_n[j+4])) for j in range(4)] 

            # compute new deformable mirror
            wfs.apply_shift_wfs(sx=value_x,sy=value_y)
            # compute the interaction matrix for the positive mis-registration
            calib_tmp_p = InteractionMatrix(ngs, atm, tel, dm_0, wfs, basis.modes, stroke, phaseOffset=0, nMeasurements=50,invert=False,print_time=False)

            # save output in fits file
            if save_sensitivity_matrices:
                hdr=pfits.Header()
                hdr['TITLE']    = 'INTERACTION MATRIX - POSITIVE MIS-REGISTRATION'
                empty_primary   = pfits.PrimaryHDU(header=hdr)
                primary_hdu     = pfits.ImageHDU(calib_tmp_p.D)
                hdu             = pfits.HDUList([empty_primary, primary_hdu])
                hdu.writeto(name_p,overwrite=True)

    #%% --------------------  NEGATIVE MIS-REGISTRATION --------------------
        try:
            if recompute_sensitivity is False:
                hdu = pfits.open(name_n)
                calib_tmp_n = CalibrationVault(hdu[1].data,invert=False)
            else:
                hdu = pfits.open(name_0+'_volontary_error')    
        except:
            # compute new mask 
            value_x = []
            value_y = []

            [value_x.append(-val_x[i][j]*epsilon + getattr(misRegistrationZeroPoint,epsilonMisRegistration_n[j])) for j in range(4)] 
            [value_y.append(-val_y[i][j]*epsilon + getattr(misRegistrationZeroPoint,epsilonMisRegistration_n[j+4])) for j in range(4)] 

            # compute new deformable mirror
            wfs.apply_shift_wfs(sx=value_x,sy=value_y)            # compute the interaction matrix for the negative mis-registration
            calib_tmp_n = InteractionMatrix(ngs, atm, tel, dm_0, wfs, basis.modes, stroke, phaseOffset=0, nMeasurements=50,invert=False,print_time=False)

            # save output in fits file
            if save_sensitivity_matrices:
                hdr=pfits.Header()
                hdr['TITLE']    = 'INTERACTION MATRIX - NEGATIVE MIS-REGISTRATION'
                empty_primary   = pfits.PrimaryHDU(header=hdr)
                primary_hdu     = pfits.ImageHDU(calib_tmp_n.D)
                hdu             = pfits.HDUList([empty_primary, primary_hdu])
                hdu.writeto(name_n,overwrite=True)                
       
    #%% --------------------  COMPUTE THE META SENSITIVITY MATRIX --------------------
    # reshape the matrix as a row vector
#        print('The value of the epsilon mis-registration is '+ str(getattr(epsilonMisRegistration,epsilonMisRegistration_field[i])))
        try:
            row_meta_matrix   = np.reshape(((calib_tmp_p.D -calib_tmp_n.D)/2.)/epsilon,[calib_tmp_p.D.shape[0]*calib_tmp_p.D.shape[1]])
        except:
            row_meta_matrix   = np.reshape(((calib_tmp_p.D -calib_tmp_n.D)/2.)/epsilon,[calib_tmp_p.D.shape[0]])            
        meta_matrix[:,i]  = row_meta_matrix 
        
    metaSensitivityMatrix = CalibrationVault(meta_matrix)
    
    return metaSensitivityMatrix, calib_0

