# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:21:18 2020

@author: cheritie
"""





# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:22:55 2020

@author: cheritie
"""

from AO_modules.CalibrationVault import calibrationVault
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.InteractionMatrix import interactionMatrix
from AO_modules.MisRegistration   import MisRegistration
from astropy.io import fits as pfits
import numpy as np
from AO_modules.tools.tools import createFolder

def computePCAModes(nameFolder,nameSystem,tel,atm,ngs,dm_0,pitch,wfs,basis,misRegistrationZeroPoint,epsilonMisRegistration,param):
    
    #%% --------------------  CREATION OF THE DESTINATION FOLDER --------------------
    homeFolder = misRegistrationZeroPoint.misRegName+'/'
    intMat_name = 'interactionMatrix'
    
    
    if basis.modes.shape == np.shape(np.eye(dm_0.nValidAct)):
        
        comparison = basis.modes == np.eye(dm_0.nValidAct)
        if comparison.all():
            foldername  = nameFolder+nameSystem+homeFolder+'zonal/'
            extraName   = '' 
            createFolder(foldername)

        else:
            foldername  = nameFolder+nameSystem+homeFolder+'modal/'
            extraName   = '_'+str(basis.indexModes[0])+'-'+str(basis.indexModes[-1])+'_'+basis.extra
            createFolder(foldername)
    else:
        foldername  = nameFolder+nameSystem+homeFolder+'modal/'
        extraName   = '_'+str(basis.indexModes[0])+'-'+str(basis.indexModes[-1])+'_'+basis.extra    
        createFolder(foldername)
    

    
    #%% --------------------  COMPUTATION OF THE INTERACTION MATRICES FOLDER --------------------
    epsilonMisRegistration_name = ['dRot','dX','dY']
    epsilonMisRegistration_field = ['rotationAngle','shiftX','shiftY']
    
    M2C_PCA = np.zeros([basis.modes.shape[0],basis.modes.shape[1],len(epsilonMisRegistration_name)])

    for i in range(len(epsilonMisRegistration_name)):
        # name for the matrices
        name_p = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_p_'+str(np.abs(epsilonMisRegistration.shiftX))+ extraName+'.fits'
        name_n = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_m_'+str(np.abs(epsilonMisRegistration.shiftX))+ extraName+'.fits'
        
        
        stroke = 1e-12
    #%% --------------------  POSITIVE MIS-REGISTRATION --------------------
        try:
            hdu = pfits.open(name_p)
            calib_tmp_p = calibrationVault(hdu[1].data)
        except:
            # set the mis-registration value
            misRegistration_tmp = MisRegistration(misRegistrationZeroPoint)
            setattr(misRegistration_tmp,epsilonMisRegistration_field[i],getattr(epsilonMisRegistration,epsilonMisRegistration_field[i]))
            # compute new deformable mirror

            dm_tmp = DeformableMirror(telescope    = tel,\
                          nSubap       = param['nSubaperture'],\
                          misReg       = misRegistration_tmp,\
                          M4_param     = param)
            # compute the interaction matrix for the positive mis-registration
            calib_tmp_p = interactionMatrix(ngs,atm,tel,dm_tmp,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=50)
            # save output in fits file
            hdr=pfits.Header()
            hdr['TITLE'] = 'INTERACTION MATRIX - POSITIVE MIS-REGISTRATION'
            empty_primary = pfits.PrimaryHDU(header=hdr)
            primary_hdu = pfits.ImageHDU(calib_tmp_p.D)
            hdu = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(name_p,overwrite=True)
            del dm_tmp

    #%% --------------------  NEGATIVE MIS-REGISTRATION --------------------
        try:
            hdu = pfits.open(name_n)
            calib_tmp_n = calibrationVault(hdu[1].data)
        except:
            # set the mis-registration value
            misRegistration_tmp = MisRegistration(misRegistrationZeroPoint)
            setattr(misRegistration_tmp,epsilonMisRegistration_field[i], -getattr(epsilonMisRegistration,epsilonMisRegistration_field[i]))
            # compute new deformable mirror
            dm_tmp = DeformableMirror(telescope    = tel,\
                          nSubap       = param['nSubaperture'],\
                          misReg       = misRegistration_tmp,\
                          M4_param     = param)
            # compute the interaction matrix for the negative mis-registration
            calib_tmp_n = interactionMatrix(ngs,atm,tel,dm_tmp,wfs,basis.modes,stroke,phaseOffset=0,nMeasurements=50)
            # save output in fits file
            hdr=pfits.Header()
            hdr['TITLE'] = 'INTERACTION MATRIX - NEGATIVE MIS-REGISTRATION'
            empty_primary = pfits.PrimaryHDU(header=hdr)
            primary_hdu = pfits.ImageHDU(calib_tmp_n.D)
            hdu = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(name_n,overwrite=True)                
            del dm_tmp
       
    #%% --------------------  COMPUTE THE META SENSITIVITY MATRIX --------------------
    # reshape the matrix as a row vector
        derivative_matrix     = (calib_tmp_p.D -calib_tmp_n.D)/2/getattr(epsilonMisRegistration,epsilonMisRegistration_field[i])
        
        tmp_cal = calibrationVault(derivative_matrix)
        
        M2C =  tmp_cal.V.T
       
        tmp =  dm_0.modes@basis.modes@M2C*2
        norma = np.std(np.squeeze(tmp[tel.pupilLogical,:]),axis =0)
        
        M2C_PCA[:,:,i] = basis.modes@M2C@np.diag(1/norma)

        
    return M2C_PCA
