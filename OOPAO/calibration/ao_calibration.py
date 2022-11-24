# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:47:17 2020

@author: cheritie
"""

import time

import numpy as np
from astropy.io import fits as pfits

from .CalibrationVault import CalibrationVault
from .InteractionMatrix import InteractionMatrix
from ..tools.tools import createFolder, emptyClass, read_fits


def ao_calibration_from_ao_obj(ao_obj, nameFolderIntMat = None, nameIntMat = None, nameFolderBasis = None, nameBasis = None, nMeasurements=50, index_modes = None, get_basis = True):
    
    
    # check if the name of the basis is specified otherwise take the nominal name
    if nameBasis is None:
        if ao_obj.dm.isM4:
            initName = 'M2C_M4_'
        else:
            initName = 'M2C_'
        try:
            nameBasis = initName+str(ao_obj.param['resolution'])+'_res'+ao_obj.param['extra']
        except:
            nameBasis = initName+str(ao_obj.param['resolution'])+'_res'
    ao_calib_object             = emptyClass()
    
        # check if a name for the origin folder is specified
    if nameFolderBasis is None:
        nameFolderBasis = ao_obj.param['pathInput']
    createFolder(nameFolderBasis)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
#    get the modal basis : 
    
    try:
        print('Loading the KL Modal Basis from: ' + nameFolderBasis+nameBasis )
        M2C = read_fits(nameFolderBasis+ nameBasis+'.fits')
        if index_modes is None:
            M2C = M2C[:,:ao_obj.param['nModes']]
        else:
            M2C = M2C[:,index_modes]
        
        ao_obj.param['modal_basis_filename'] = nameFolderBasis+ nameBasis+'.fits'

        if get_basis:
            ao_obj.dm.coefs = M2C
            ao_obj.tel*ao_obj.dm
        
            basis = np.reshape(ao_obj.tel.OPD,[ao_obj.tel.resolution**2,M2C.shape[1]])
            ao_calib_object.basis = basis
        
        if ao_obj.param['getProjector']:

            print('Computing the pseudo-inverse of the modal basis...')

            cross_product_basis = np.matmul(basis.T,basis) 
            
            non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
            criteria = np.abs(1-np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis))
            if criteria <= 1e-3:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the fast computation')
                projector = np.diag(1/np.diag(cross_product_basis))@basis.T
            else:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the slow computation')
                projector = np.linalg.pinv(basis)  
            print('saving for later..')
            print('Done!')
                
            ao_calib_object.projector   = projector
        
    except:
        print('ERROR: No file found! Taking a zonal basis instead..' )
        M2C = np.eye(ao_obj.dm.nValidAct)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    if nameFolderIntMat is None:
        nameFolderIntMat = ao_obj.param['pathInput']+ao_obj.param['name']+'/'
    createFolder(nameFolderIntMat)

#%    get the interaction matrix : 
        
    if nameIntMat is None:
        if ao_obj.wfs.tag == 'pyramid' or ao_obj.wfs.tag == 'double_wfs':
            try:
                # case where the system name has an extra attribute
                nameIntMat = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.param['modulation'])+'_mod_'+str(ao_obj.param['postProcessing'])+'_psfCentering_'+str(ao_obj.param['psfCentering'])+ao_obj.param['extra']
            except:
                nameIntMat = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.param['modulation'])+'_mod_'+str(ao_obj.param['postProcessing'])+'_psfCentering_'+str(ao_obj.param['psfCentering'])
       
        if ao_obj.wfs.tag == 'shackHartmann':
            if ao_obj.wfs.is_geometric: 
                nature = 'geometric'
            else:
                nature = 'diffractive'
            try:
                # case where the system name has an extra attribute
                nameIntMat = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.wfs.nValidSubaperture)+'_subap_'+nature+'_'+ao_obj.param['extra']
            except:
                nameIntMat = 'zonal_interaction_matrix_'+str(ao_obj.param['resolution'])+'_res_'+str(ao_obj.wfs.nValidSubaperture)+'_subap_'+nature      
    try:
        print('Loading Interaction matrix '+nameIntMat+'...')
        imat = read_fits(nameFolderIntMat+nameIntMat+'.fits')
        calib = CalibrationVault(imat@M2C)    
        print('Done!')
        ao_obj.param['interaction_matrix_filename'] = nameFolderIntMat+nameIntMat+'.fits'


        
    except:  
        print('ERROR! Computingh the zonal interaction matrix')
        print('ERROR: No file found! computing imat..' )
        time.sleep(5)
        M2C_zon = np.eye(ao_obj.dm.nValidAct)

        stroke =1e-9 # 1 nm amplitude
        calib = InteractionMatrix(ao_obj.ngs,ao_obj.atm,ao_obj.tel,ao_obj.dm,ao_obj.wfs,M2C_zon,stroke,phaseOffset = 0,nMeasurements = nMeasurements)
        # save output in fits file
        hdr=pfits.Header()
        hdr['TITLE'] = 'INTERACTION MATRIX'
        empty_primary = pfits.PrimaryHDU(header=hdr)
        # primary_hdu = pfits.ImageHDU(calib.D.astype(np.float32))
        primary_hdu = pfits.ImageHDU(calib.D)

        hdu = pfits.HDUList([empty_primary, primary_hdu])
        hdu.writeto(nameFolderIntMat + nameIntMat + '.fits', overwrite=True)
        calib = CalibrationVault(calib.D@M2C)    

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
   
#%    get the modal gains matrix :   
    nameExtra = '_r0_'+str(100*ao_obj.atm.r0)+'_cm_'+ao_obj.param['opticalBand']+'_band_fitting_'+str(ao_obj.param['nModes'])+'_KL'
    try:
        nameModalGains = 'modal_gains'+ao_obj.param['extra']+nameExtra
    except:
        nameModalGains = 'modal_gains'+nameExtra    
    
    try:
        data_gains  = read_fits(nameFolderIntMat+ nameModalGains+'.fits')

        print('Using Modal Gains loaded from '+str(nameFolderIntMat+ nameModalGains+'.fits'))

    except:
        data_gains      = np.ones(M2C.shape[1]) 
        print('No Modal Gains found. All gains set to 1')   
    
    ao_calib_object.gOpt        = np.diag(1/data_gains)
    ao_calib_object.M2C         = M2C
    ao_calib_object.calib       = calib

    return ao_calib_object

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# same function using an ao object as an input

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

def ao_calibration(ngs, tel, atm, dm, wfs, param, nameFolderIntMat = None, nameIntMat = None, nameFolderBasis = None, nameBasis = None, nMeasurements=50, index_modes = None, get_basis = True,input_basis = None):
    
    # check if the name of the basis is specified otherwise take the nominal name
    if nameBasis is None:
        if dm.isM4:
            initName = 'M2C_M4_'
        else:
            initName = 'M2C_'
    
        try:
            nameBasis = initName+str(param['resolution'])+'_res'+param['extra']
        except:
            nameBasis = initName+str(param['resolution'])+'_res'
    
    ao_calib_object             = emptyClass()
    
        # check if a name for the origin folder is specified
    if nameFolderBasis is None:
        nameFolderBasis = param['pathInput']
    createFolder(nameFolderBasis)
        
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
#    get the modal basis : 
    if input_basis is None:
        try:
            print('Loading the KL Modal Basis from: ' + nameFolderBasis+nameBasis )
           
            M2C = read_fits(nameFolderBasis+ nameBasis+'.fits')
            param['modal_basis_filename'] = nameFolderBasis+ nameBasis+'.fits'
            if index_modes is None:
                M2C = M2C[:,:param['nModes']]
            else:
                M2C = M2C[:,index_modes]
                
            if get_basis or param['getProjector']:
                dm.coefs = M2C
                tel*dm
        
                basis = np.reshape(tel.OPD,[tel.resolution**2,M2C.shape[1]])
                ao_calib_object.basis       = basis
    
            if param['getProjector']:
                print('Computing the pseudo-inverse of the modal basis...')
                cross_product_basis = np.matmul(basis.T,basis) 
                non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
                criteria = 1-np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis)
                if criteria <= 1e-3:
                    print('Diagonality criteria: ' + str(criteria) + ' -- using the fast computation')
                    projector = np.diag(1/np.diag(cross_product_basis))@basis.T
                else:
                    print('Diagonality criteria: ' + str(criteria) + ' -- using the slow computation')
                    projector = np.linalg.pinv(basis)  
                ao_calib_object.projector   = projector        
        except:
            print('ERROR: No file found! Taking a zonal basis instead..' )
            M2C = np.eye(dm.nValidAct)
    else:
        M2C = input_basis
        if get_basis or param['getProjector']:
            dm.coefs = M2C
            tel*dm
    
            basis = np.reshape(tel.OPD,[tel.resolution**2,M2C.shape[1]])
            ao_calib_object.basis       = basis

        if param['getProjector']:
            print('Computing the pseudo-inverse of the modal basis...')
            cross_product_basis = np.matmul(basis.T,basis) 
            non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
            criteria = 1-np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis)
            if criteria <= 1e-3:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the fast computation')
                projector = np.diag(1/np.diag(cross_product_basis))@basis.T
            else:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the slow computation')
                projector = np.linalg.pinv(basis)  
            ao_calib_object.projector   = projector     
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    if nameFolderIntMat is None:
        nameFolderIntMat = param['pathInput']+param['name']+'/'
    createFolder(nameFolderIntMat)
#%    get the interaction matrix : 
        
    if nameIntMat is None:
        
        
        if wfs.tag == 'pyramid' or  wfs.tag == 'double_wfs' :
            try:
                # case where the system name has an extra attribute
                nameIntMat = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(param['modulation'])+'_mod_'+str(param['postProcessing'])+'_psfCentering_'+str(param['psfCentering'])+param['extra']
            except:
                nameIntMat = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(param['modulation'])+'_mod_'+str(param['postProcessing'])+'_psfCentering_'+str(param['psfCentering'])
       
        if wfs.tag == 'shackHartmann':
            if wfs.is_geometric: 
                nature = 'geometric'
            else:
                nature = 'diffractive'
            try:
                # case where the system name has an extra attribute
                nameIntMat = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(wfs.nValidSubaperture)+'_subap_'+nature+'_'+param['extra']
            except:
                nameIntMat = 'zonal_interaction_matrix_'+str(param['resolution'])+'_res_'+str(wfs.nValidSubaperture)+'_subap_'+nature     

            
            
            
    try:
        print('Loading Interaction matrix '+nameFolderIntMat+nameIntMat+'...')
        imat = read_fits(nameFolderIntMat+nameIntMat+'.fits')
        calib = CalibrationVault(imat@M2C)      
        print('Done!')
        param['interaction_matrix_filename'] = nameFolderIntMat+nameIntMat+'.fits'

        
    except:  
        
        print('ERROR: No file found! computing imat..' )
        time.sleep(5)
        M2C_zon = np.eye(dm.nValidAct)
        stroke =1e-9 # 1 nm amplitude
        calib = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C_zon, stroke, phaseOffset = 0, nMeasurements = nMeasurements)
        # save output in fits file
        hdr=pfits.Header()
        hdr['TITLE'] = 'INTERACTION MATRIX'
        empty_primary = pfits.PrimaryHDU(header=hdr)
        # primary_hdu = pfits.ImageHDU(calib.D.astype(np.float32))
        primary_hdu = pfits.ImageHDU(calib.D)
        hdu = pfits.HDUList([empty_primary, primary_hdu])
        hdu.writeto(nameFolderIntMat+nameIntMat+'.fits',overwrite=True)
        calib = CalibrationVault(calib.D@M2C)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
   
#%    get the modal gains matrix :   
    nameExtra = '_r0_'+str(100*atm.r0)+'_cm_'+param['opticalBand']+'_band_fitting_'+str(param['nModes'])+'_KL'
    try:
        nameModalGains = 'modal_gains'+param['extra']+nameExtra
    except:
        nameModalGains = 'modal_gains'+nameExtra    
    
    try:
        data_gains  = read_fits(nameFolderIntMat+ nameModalGains+'.fits')

        print('Using Modal Gains loaded from '+str(nameFolderIntMat+ nameModalGains+'.fits'))

    except:
        data_gains      = np.ones(M2C.shape[1]) 
        print('No Modal Gains found. All gains set to 1')

    
    ao_calib_object.gOpt        = np.diag(1/data_gains)        
    
    ao_calib_object.M2C         = M2C
    ao_calib_object.calib       = calib

    
    
    return ao_calib_object

def get_modal_gains_from_ao_obj(ao_obj, nameFolderIntMat = None):
    
    if nameFolderIntMat is None:
        nameFolderIntMat = ao_obj.param['pathInput']+ao_obj.param['name']+'/'
    createFolder(nameFolderIntMat)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
   
#%    get the modal gains matrix :   
    nameExtra = '_r0_'+str(100*ao_obj.atm.r0)+'_cm_'+ao_obj.param['opticalBand']+'_band_fitting_'+str(ao_obj.param['nModes'])+'_KL'
    try:
        nameModalGains = 'modal_gains'+ao_obj.param['extra']+nameExtra
    except:
        nameModalGains = 'modal_gains'+nameExtra    
    print('Looking for Modal Gains loaded from '+str(nameFolderIntMat+ nameModalGains+'.fits'))
    try:
        data_gains  = read_fits(nameFolderIntMat+ nameModalGains+'.fits')
        print('Using Modal Gains loaded from '+str(nameFolderIntMat+ nameModalGains+'.fits'))

    except:
        data_gains      = np.ones(ao_obj.param['nModes']) 
        print('No Modal Gains found. All gains set to 1')   
    
    gOpt        = np.diag(1/data_gains)

    return gOpt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# same function using an ao object as an input

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

def get_modal_gains(param, nameFolderIntMat = None,r0 =None):
    if r0 is None:
        r0 = param['r0']
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    if nameFolderIntMat is None:
        nameFolderIntMat = param['pathInput']+param['name']+'/'
    createFolder(nameFolderIntMat)
#%    get the modal gains matrix :   
    nameExtra = '_r0_'+str(100*r0)+'_cm_'+param['opticalBand']+'_band_fitting_'+str(param['nModes'])+'_KL'
    try:
        nameModalGains = 'modal_gains'+param['extra']+nameExtra
    except:
        nameModalGains = 'modal_gains'+nameExtra    
    
    try:
        data_gains  = read_fits(nameFolderIntMat+ nameModalGains+'.fits')
        print('Using Modal Gains loaded from '+str(nameFolderIntMat+ nameModalGains+'.fits'))

    except:
        data_gains      = np.ones(param['nModes']) 
        print('No Modal Gains found. All gains set to 1')
    gOpt        = np.diag(1/data_gains)        
    return gOpt