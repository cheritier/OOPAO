# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:49:46 2021

@author: cheritie
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:22:55 2020

@author: cheritie
"""

from AO_modules.calibration.CalibrationVault import calibrationVault
from AO_modules.calibration.InteractionMatrix import interactionMatrix
from AO_modules.MisRegistration  import MisRegistration
from astropy.io import fits as pfits
import numpy as np
from AO_modules.tools.tools import createFolder
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration

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
                    basis.indexModes : indexes of the modes considered in the basis. This is used to name the sensitivity matrices 
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
def computeMetaSensitivityMatrix(nameFolder, nameSystem, tel, atm, ngs, dm_0, pitch, wfs, basis, misRegistrationZeroPoint, epsilonMisRegistration, param, wfs_mis_registrated = None, extra_dm_mis_registration = None):
    #%% --------------------  CREATION OF THE DESTINATION FOLDER --------------------
    homeFolder = misRegistrationZeroPoint.misRegName+'/'
    intMat_name = 'IM'
    if extra_dm_mis_registration is None:
        print('No extra mis-registration added to the dm')
        extra_misReg_name = ''
    else:
        print('Warning, an extra mis-registration is added to the dm')
        extra_misReg_name = '_extra_dm_rot_'   + str('%.2f' %extra_dm_mis_registration.rotationAngle)            +'_deg_'\
                            'sX_'             + str('%.2f' %(extra_dm_mis_registration.shiftX))                 +'_m_'\
                            'sY_'             + str('%.2f' %(extra_dm_mis_registration.shiftY))                 +'_m_'

        
    if basis.modes.shape == np.shape(np.eye(dm_0.nValidAct)):
        
        comparison = basis.modes == np.eye(dm_0.nValidAct)
        if comparison.all():
            foldername  = nameFolder+nameSystem+homeFolder+'zonal/'
            extraName   = ''+extra_misReg_name 
            createFolder(foldername)

        else:
            foldername  = nameFolder+nameSystem+homeFolder+'modal/'
            extraName   = '_'+str(basis.indexModes[0])+'-'+str(basis.indexModes[-1])+'_'+basis.extra+extra_misReg_name
            createFolder(foldername)
    else:
        foldername  = nameFolder+nameSystem+homeFolder+'modal/'
        extraName   = '_'+str(basis.indexModes[0])+'-'+str(basis.indexModes[-1])+'_'+basis.extra+extra_misReg_name    
        createFolder(foldername)
    

    
    #%% --------------------  COMPUTATION OF THE INTERACTION MATRICES FOLDER --------------------
    epsilonMisRegistration_name  = ['dRot','dX','dY','dTanScal','dRadScal']
    epsilonMisRegistration_field = ['rotationAngle','shiftX','shiftY','tangentialScaling','radialScaling']
    meta_matrix =np.zeros([wfs.nSignal*basis.modes.shape[1],int(len(epsilonMisRegistration_name))])
    for i in range(len(epsilonMisRegistration_name)):
        # name for the matrices
        name_0 = foldername + intMat_name +'_0'+ extraName+'.fits'
        name_p = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_p_'+str(np.abs(epsilonMisRegistration.shiftX))+ extraName+'.fits'
        name_n = foldername + intMat_name +'_'+ epsilonMisRegistration_name[i]+'_m_'+str(np.abs(epsilonMisRegistration.shiftX))+ extraName+'.fits'
        
        stroke = 1e-12


    #%% --------------------  CENTERED INTERACTION MATRIX --------------------
        try:
            hdu = pfits.open(name_0)
            calib_0 = calibrationVault(hdu[1].data)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('WARNING: you are loading existing data from \n')
            print(str(name_0)+'/n')
            print('Make sure that the loaded data correspond to your AO system!')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        except:
            calib_0 = interactionMatrix(ngs, atm, tel, dm_0, wfs, basis.modes ,stroke, phaseOffset=0, nMeasurements=50)
            # save output in fits file
            hdr=pfits.Header()
            hdr['TITLE']    = 'INTERACTION MATRIX - INITIAL POINT'
            empty_primary   = pfits.PrimaryHDU(header=hdr)
            primary_hdu     = pfits.ImageHDU(calib_0.D)
            hdu             = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(name_0,overwrite=True)

    #%% --------------------  POSITIVE MIS-REGISTRATION --------------------
        try:
            hdu = pfits.open(name_p)
            calib_tmp_p = calibrationVault(hdu[1].data)
        except:
            # set the mis-registration value
            misRegistration_tmp = MisRegistration(misRegistrationZeroPoint)
            
            setattr(misRegistration_tmp,epsilonMisRegistration_field[i],getattr(misRegistration_tmp,epsilonMisRegistration_field[i]) + getattr(epsilonMisRegistration,epsilonMisRegistration_field[i]))
            
            # compute new deformable mirror
            dm_tmp      = applyMisRegistration(tel,misRegistration_tmp,param, wfs = wfs_mis_registrated, extra_dm_mis_registration = extra_dm_mis_registration)
            # compute the interaction matrix for the positive mis-registration
            calib_tmp_p = interactionMatrix(ngs, atm, tel, dm_tmp, wfs, basis.modes, stroke, phaseOffset=0, nMeasurements=50)
            # save output in fits file
            hdr=pfits.Header()
            hdr['TITLE']    = 'INTERACTION MATRIX - POSITIVE MIS-REGISTRATION'
            empty_primary   = pfits.PrimaryHDU(header=hdr)
            primary_hdu     = pfits.ImageHDU(calib_tmp_p.D)
            hdu             = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(name_p,overwrite=True)
            del dm_tmp

    #%% --------------------  NEGATIVE MIS-REGISTRATION --------------------
        try:
            hdu = pfits.open(name_n)
            calib_tmp_n = calibrationVault(hdu[1].data)
        except:
            # set the mis-registration value
            misRegistration_tmp = MisRegistration(misRegistrationZeroPoint)
            setattr(misRegistration_tmp,epsilonMisRegistration_field[i], getattr(misRegistration_tmp,epsilonMisRegistration_field[i]) - getattr(epsilonMisRegistration,epsilonMisRegistration_field[i]))
            # compute new deformable mirror
            dm_tmp = applyMisRegistration(tel,misRegistration_tmp,param, wfs = wfs_mis_registrated,extra_dm_mis_registration = extra_dm_mis_registration)
            # compute the interaction matrix for the negative mis-registration
            calib_tmp_n = interactionMatrix(ngs, atm, tel, dm_tmp, wfs, basis.modes, stroke, phaseOffset=0, nMeasurements=50)
            # save output in fits file
            hdr=pfits.Header()
            hdr['TITLE']    = 'INTERACTION MATRIX - NEGATIVE MIS-REGISTRATION'
            empty_primary   = pfits.PrimaryHDU(header=hdr)
            primary_hdu     = pfits.ImageHDU(calib_tmp_n.D)
            hdu             = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(name_n,overwrite=True)                
            del dm_tmp
       
    #%% --------------------  COMPUTE THE META SENSITIVITY MATRIX --------------------
    # reshape the matrix as a row vector
#        print('The value of the epsilon mis-registration is '+ str(getattr(epsilonMisRegistration,epsilonMisRegistration_field[i])))
       
        row_meta_matrix   = np.reshape(((calib_tmp_p.D -calib_tmp_n.D)/2.)/getattr(epsilonMisRegistration,epsilonMisRegistration_field[i]),[calib_tmp_p.D.shape[0]*calib_tmp_p.D.shape[1]])
        meta_matrix[:,i]  = row_meta_matrix 
        
    metaSensitivityMatrix = calibrationVault(meta_matrix)
    
    return metaSensitivityMatrix, calib_0
