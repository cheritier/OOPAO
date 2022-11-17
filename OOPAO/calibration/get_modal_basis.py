# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:21:00 2021

@author: cheritie
"""

import numpy as np
from astropy.io import fits as pfits

from ..tools.tools import createFolder, emptyClass


def get_modal_basis_from_ao_obj(ao_obj, nameFolderBasis = None, nameBasis = None):
    
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
       
        hdu=pfits.open(nameFolderBasis+ nameBasis+'.fits')
        try:
            M2C = hdu[1].data
        except:
            M2C = hdu[0].data
            
        M2C = M2C[:,:ao_obj.param['nModes']]
        
        ao_obj.dm.coefs = M2C
        ao_obj.tel*ao_obj.dm
    
        basis = np.reshape(ao_obj.tel.OPD,[ao_obj.tel.resolution**2,M2C.shape[1]])
        ao_calib_object.M2C   = M2C
        ao_calib_object.basis = basis
        
        if ao_obj.param['getProjector']:
            print('Computing the pseudo-inverse of the modal basis...')

            cross_product_basis = np.matmul(basis.T,basis) 
            
            non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
            criteria = np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis)
            if criteria <= 1e-3:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the fast computation')
                projector = np.diag(1/np.diag(cross_product_basis))@basis.T
            else:
                print('Diagonality criteria: ' + str(criteria) + ' -- using the slow computation')
                projector = np.linalg.pinv(basis)  
            ao_calib_object.projector   = projector  
        
    except:
        print('ERROR: No file found! Taking a zonal basis instead..' )
        M2C = np.eye(ao_obj.dm.nValidAct)
        ao_calib_object.M2C   = M2C

    return ao_calib_object
#%%
#
#
#    Same functions without ao-object
#    
#    
#%%
    
def get_modal_basis(ngs, tel, atm, dm, wfs, param, nameFolderBasis = None, nameBasis = None):
    
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
    
    try:
        print('Loading the KL Modal Basis from: ' + nameFolderBasis+nameBasis )
       
        hdu=pfits.open(nameFolderBasis+ nameBasis+'.fits')
        M2C = hdu[1].data

        M2C = M2C[:,:param['nModes']]
        
        dm.coefs = M2C
        tel*dm
    
        basis = np.reshape(tel.OPD,[tel.resolution**2,M2C.shape[1]])
        ao_calib_object.basis       = basis
        ao_calib_object.M2C   = M2C

        if param['getProjector']:
            print('Computing the pseudo-inverse of the modal basis...')

            cross_product_basis = np.matmul(basis.T,basis) 
            
            non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
            criteria = np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis)
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
        
        
    return ao_calib_object


def get_projector(basis):
        print('Computing the pseudo-inverse of the modal basis...')

        cross_product_basis = np.matmul(basis.T,basis) 
        
        non_diagonal_elements = np.sum(np.abs(cross_product_basis))-np.trace(cross_product_basis)
        criteria = np.abs(np.trace(cross_product_basis)-non_diagonal_elements)/np.trace(cross_product_basis)
        if criteria <= 1e-3:
            print('Diagonality criteria: ' + str(criteria) + ' -- using the fast computation')
            projector = np.diag(1/np.diag(cross_product_basis))@basis.T
        else:
            print('Diagonality criteria: ' + str(criteria) + ' -- using the slow computation')
            projector = np.linalg.pinv(basis)  
        
        return projector