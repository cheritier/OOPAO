# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:52:11 2020

@author: cheritie
"""
from astropy.io import fits as pfits
import numpy as np
from AO_modules.tools.tools import createFolder
from AO_modules.calibration.InteractionMatrix                  import interactionMatrix


def computeZonalInteractionMatrix_from_ao_obj(ao_obj, nameFolder = None, nameFile = None, nameExtra = '', phaseOffset = 0, amplitude = 1e-9, nMeasurements = 100):

    # name of the output file
    if nameFile is None:
        try:
            # case where the system name has an extra attribute
            nameFile = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.param['modulation'])+'_mod_'+str(ao_obj.param['postProcessing'])+'_psfCentering_'+str(ao_obj.param['psfCentering'])+ao_obj.param['extra']
        except:
            nameFile = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.param['modulation'])+'_mod_'+str(ao_obj.param['postProcessing'])+'_psfCentering_'+str(ao_obj.param['psfCentering'])
    nameFile = nameFile + nameExtra
    if nameFolder is None:
        nameFolder = ao_obj.param['pathInput']+'/'+ao_obj.param['name']+'/'
    
    createFolder(nameFolder)
    
    print('Saving the interaction matrix file as '+nameFile)
    
    print('File will be located in ' +nameFolder)
    
    # basis for the zonal interaction matrix
    M2C    = np.eye(ao_obj.dm.nValidAct)
    
    # calibration
    calib = interactionMatrix(ngs           = ao_obj.ngs,\
                              atm           = ao_obj.atm,\
                              tel           = ao_obj.tel,\
                              dm            = ao_obj.dm,\
                              wfs           = ao_obj.wfs,\
                              M2C           = M2C,\
                              stroke        = amplitude,\
                              phaseOffset   = phaseOffset,\
                              nMeasurements = nMeasurements)


    # save output in fits file
    hdr             = pfits.Header()
    hdr['TITLE']    = 'zonal_interaction_matrix'
    empty_primary   = pfits.PrimaryHDU(header=hdr)
    primary_hdu     = pfits.ImageHDU(calib.D.astype(np.float32))
    hdu             = pfits.HDUList([empty_primary, primary_hdu])
    # save output
    hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
    
    return calib


def computeZonalInteractionMatrix(ngs, atm, tel, dm, wfs,param, nameFolder = None, nameFile = None, nameExtra = '', phaseOffset = 0, amplitude = 1e-9, nMeasurements = 100):

    # name of the output file
    if nameFile is None:
        try:
            # case where the system name has an extra attribute
            nameFile = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(param['modulation'])+'_mod_'+str(param['postProcessing'])+'_psfCentering_'+str(param['psfCentering'])+param['extra']
        except:
            nameFile = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(param['modulation'])+'_mod_'+str(param['postProcessing'])+'_psfCentering_'+str(param['psfCentering'])
    nameFile = nameFile + nameExtra
    if nameFolder is None:
        nameFolder = param['pathInput']+'/'+param['name']+'/'
    
    createFolder(nameFolder)
    
    print('Saving the interaction matrix file as '+nameFile)
    
    print('File will be located in ' +nameFolder)
    
    # basis for the zonal interaction matrix
    M2C    = np.eye(dm.nValidAct)
    
    # calibration
    calib = interactionMatrix(ngs           = ngs,\
                              atm           = atm,\
                              tel           = tel,\
                              dm            = dm,\
                              wfs           = wfs,\
                              M2C           = M2C,\
                              stroke        = amplitude,\
                              phaseOffset   = phaseOffset,\
                              nMeasurements = nMeasurements)


    # save output in fits file
    hdr             = pfits.Header()
    hdr['TITLE']    = 'zonal_interaction_matrix'
    empty_primary   = pfits.PrimaryHDU(header=hdr)
    primary_hdu     = pfits.ImageHDU(calib.D)
    # primary_hdu     = pfits.ImageHDU(calib.D.astype(np.float32))
    hdu             = pfits.HDUList([empty_primary, primary_hdu])
    # save output
    hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
    
    return calib