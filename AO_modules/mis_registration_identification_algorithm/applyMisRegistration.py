# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:03:13 2020

@author: cheritie
"""

import numpy as np
from AO_modules.DeformableMirror import DeformableMirror
from astropy.io import fits as pfits
from joblib import Parallel, delayed
from AO_modules.tools.interpolateGeometricalTransformation import rotateImageMatrix,rotation,translationImageMatrix,translation,anamorphosis,anamorphosisImageMatrix
import time
from AO_modules.MisRegistration  import MisRegistration
import skimage.transform as sk
import scipy.ndimage as sp


def applyMisRegistration(tel,misRegistration_tmp,param, wfs = None):
        
        if wfs is None:
            
            # case synthetic DM - with user-defined coordinates
            try:
                if param['dm_coordinates'] is None:
                    coordinates = 0
                else:
                    coordinates = param['dm_coordinates'] 
            except:
                coordinates = 0
                
            
                
            try:
                if param['isM4'] is True:
                    # case with M4
                    dm_tmp = DeformableMirror(telescope    = tel,\
                                        nSubap       = param['nSubaperture'],\
                                        mechCoupling = param['mechanicalCoupling'],\
                                        coordinates  = coordinates,\
                                        pitch        = 0,\
                                        misReg       = misRegistration_tmp,\
                                        M4_param     = param)
                    
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Mis-Registrations Applied on M4!')
            
                else:
                    param['isM4'] = False
                
                    modes, coord, M2C, validAct =  get_influence_functions(telescope             = tel,\
                                                                            misReg               = misRegistration_tmp,\
                                                                            filename_IF          = param['filename_if'],\
                                                                            filename_mir_modes   = param['filename_mir_modes'],\
                                                                            filename_coordinates = param['filename_coord'],\
                                                                            filename_M2C         = param['filename_m2c'])
                    
                    dm_tmp = DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        coordinates  = coord,\
                        pitch        = 0,\
                        misReg       = misRegistration_tmp,\
                        modes        = np.reshape(modes,[tel.resolution**2,modes.shape[2]]),\
                        M4_param     = param)
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Mis-Registrations Applied on user-defined DM!')
                        
                    
            except:
                    # default case
                        
                param['isM4'] = False
                dm_tmp = DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        coordinates  = coordinates,\
                        pitch        = 0,\
                        misReg       = misRegistration_tmp,\
                        M4_param     = param)
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Mis-Registrations Applied on Synthetic DM!')
        else:
            if wfs.tag == 'pyramid':
                misRegistration_wfs                 = MisRegistration()
                misRegistration_wfs.shiftX          = misRegistration_tmp.shiftX
                misRegistration_wfs.shiftY          = misRegistration_tmp.shiftY
                
                apply_shift_wfs(wfs, misRegistration_wfs.shiftX / (wfs.nSubap/wfs.telescope.D), misRegistration_wfs.shiftY/ (wfs.nSubap/wfs.telescope.D))

                
                misRegistration_dm                   = MisRegistration()
                misRegistration_dm.rotationAngle     = misRegistration_tmp.rotationAngle
                misRegistration_dm.tangentialScaling = misRegistration_tmp.tangentialScaling
                misRegistration_dm.radialScaling     = misRegistration_tmp.radialScaling
                
                dm_tmp = applyMisRegistration(tel,misRegistration_dm, param, wfs = None)
                
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Mis-Registrations Applied on both DM and WFS!')
            else:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Wrong object passed as a wfs.. aplying the mis-registrations to the DM only')
                dm_tmp = applyMisRegistration(tel,misRegistration_tmp, param, wfs = None)
                             
        return dm_tmp
    
    
    
    
    

def apply_shift_wfs(wfs,sx,sy):
    
    tmp                             = np.ones([wfs.nRes,wfs.nRes])
    tmp[:,0]                        = 0
    Tip                             = (sp.morphology.distance_transform_edt(tmp))
    Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
    
    # normalize the TT to apply the modulation in terms of lambda/D
    Tip                        = (wfs.telRes/wfs.nSubap)*(((Tip/Tip.max())-0.5)*2*np.pi)
    Tilt                       = (wfs.telRes/wfs.nSubap)*(((Tilt/Tilt.max())-0.5)*2*np.pi)
    
    wfs.mask = np.exp(1j*(wfs.initial_m+sx*Tip+sy*Tilt))
    
    
    
def get_influence_functions(telescope, misReg, filename_IF,filename_mir_modes,filename_coordinates, filename_M2C):

    #% Load eigen modes of the mirror
    hdu = pfits.open(filename_IF)

    F = hdu[0].data

    influenceFunctions_ASM = F[:,30:470,19:459]
    
    a= time.time()
    nAct,nx, ny = influenceFunctions_ASM.shape
    
    pixelSize_M4_original = 8.25/nx
    
             
    # size of the influence functions maps
    resolution_M4_original       = int(nx)   
    
    # resolution of the M1 pupil
    resolution_M1                = telescope.pupil.shape[1]
    
    # compute the pixel scale of the M1 pupil
    pixelSize_M1                 = 8.25/resolution_M1
    
    # compute the ratio_M4_M1 between both pixel scale.
    ratio_M4_M1                  = pixelSize_M4_original/pixelSize_M1
    # after the interpolation the image will be shifted of a fraction of pixel extra if ratio_M4_M1 is not an integer
    extra = (ratio_M4_M1)%1 
    
    # difference in pixels between both resolutions    
    nPix = resolution_M4_original-resolution_M1
    
    if nPix%2==0:
        # case nPix is even
        # alignement of the array with respect to the interpolation 
        # (ratio_M4_M1 is not always an integer of pixel)
        extra_x = extra/2 -0.5
        extra_y = extra/2 -0.5
        
        # crop one extra pixel on one side
        nCrop_x = nPix//2
        nCrop_y = nPix//2
    else:
        # case nPix is uneven
        # alignement of the array with respect to the interpolation 
        # (ratio_M4_M1 is not always an integer of pixel)
        extra_x = extra/2 -0.5 -0.5
        extra_y = extra/2 -0.5 -0.5
        # crop one extra pixel on one side
        nCrop_x = nPix//2
        nCrop_y = (nPix//2)+1
           
    # allocate memory to store the influence functions
    influMap = np.zeros([resolution_M4_original,resolution_M4_original])  
    
    #-------------------- The Following Transformations are applied in the following order -----------------------------------
       
    # 1) Down scaling to get the right pixel size according to the resolution of M1
    downScaling     = anamorphosisImageMatrix(influMap,0,[ratio_M4_M1,ratio_M4_M1])
    
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(influMap,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
    rotMatrix               = rotateImageMatrix(influMap,misReg.rotationAngle)
    shiftMatrix             = translationImageMatrix(influMap,[misReg.shiftY/pixelSize_M1,misReg.shiftX/pixelSize_M1]) #units are in m
    
    # Shift of half a pixel to center the images on an even number of pixels
    alignmentMatrix         = translationImageMatrix(influMap,[extra_x,extra_y])
        
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,order=3)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction_IF(influMap):
        output = globalTransformation(influMap)  
        return output
    
    
    
    print('Reconstructing the influence functions ... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs=4,prefer='threads')(delayed(reconstruction_IF)(i) for i in influenceFunctions_ASM)
        return Q 
    
    
    influenceFunctions_tmp =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
    influenceFunctions_tmp = influenceFunctions_tmp  [nCrop_x:-nCrop_y,nCrop_x:-nCrop_y,:]
    
    
    b= time.time()
    
    print(b-a)
    
    
    if filename_mir_modes is None:
        print('Influence functions already in the zonal basis')
        influenceFunctions = influenceFunctions_tmp
    else:
        hdu = pfits.open(filename_mir_modes)
        
        U = hdu[0].data
        
        mod2zon = np.reshape(U[np.where(np.abs(U)!=0)],[663,663])
        
        influenceFunctions = influenceFunctions_tmp@ mod2zon.T
    
    hdu = pfits.open(filename_M2C)

    M2C = hdu[0].data
    
    validAct = np.where(M2C[:,2]!=0)
    validAct = validAct[0].astype(int)
    
    M2C = M2C[validAct,:]
    
    hdu = pfits.open(filename_coordinates)

    coordinates =   hdu[0].data[validAct,:]/100
    
    return influenceFunctions, coordinates, M2C, validAct