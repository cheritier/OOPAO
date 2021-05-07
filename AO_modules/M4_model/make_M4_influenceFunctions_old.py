# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:14:59 2020

@author: cheritie
"""

import numpy as np
import skimage.transform as sk
import ctypes
import time

from astropy.io import fits as pfits
from joblib import Parallel, delayed

from AO_modules.MisRegistration import MisRegistration
from AO_modules.tools.interpolateGeometricalTransformation import rotateImageMatrix,rotation,translationImageMatrix,translation,anamorphosis,anamorphosisImageMatrix

try : 
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_set_num_threads(8)
except:
    try:
        mkl_rt = ctypes.CDLL('./mkl_rt.dll')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_set_num_threads(8)
    except:
        print('Could not optimize the parallelisation of the code ')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF THE FUNCTION   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def makeM4influenceFunctions(pup,filename,misReg=0,nAct=0,nJobs=30,nThreads=20):
    a = time.time()
    try:
        mkl_set_num_threads(nThreads)
    except:
         print('Could not optimize the parallelisation of the code ')
    # read the FITS file of the cropped influence functions.
    hdul = pfits.open( filename )
    print("Read influence function from fits file : ", filename)
    
    xC                          = hdul[0].header['XCENTER']
    pixelSize_M4_original       = hdul[0].header['pixsize']
    i1, j1                      = hdul[1].data
    influenceFunctions_M4 = hdul[2].data
    xpos, ypos                  = hdul[3].data
    
    # size of influence functions and number of actuators
    nx, ny, nActu = influenceFunctions_M4.shape
    
    # create a MisReg object to store the different mis-registration
    if np.isscalar(misReg):
        if misReg==0:
            misReg=MisRegistration()
        else:
            print('ERROR: wrong value for the mis-registrations')
   
    # case with less actuators (debug purposes)
    if nAct!=0:
        nActu=nAct
        
    # coordinates of M4 before the interpolation        
    coordinates_M4_original      = np.zeros([nAct,2])    
    coordinates_M4_original[:,0] = xpos[:nAct]
    coordinates_M4_original[:,1] = ypos[:nAct]
    
    # size of the influence functions maps
    resolution_M4_original       = int(np.ceil(2*xC))   
    
    # resolution of the M1 pupil
    resolution_M1                = pup.shape[1]
    
    # compute the pixel scale of the M1 pupil
    pixelSize_M1                 = 40/resolution_M1
    
    # compute the ratio_M4_M1 between both pixel scale.
    ratio_M4_M1                  = pixelSize_M4_original/pixelSize_M1
    
    # crop the IFs support to get an integer value of pixel to consider for the resampling. 
    # if no integer is possible, the smallest value is taken
    
    out = np.zeros(30)
    n1  = np.zeros(30)
    n2  = np.zeros(30)
    
    for i in range(30):
        sizeTest    = resolution_M4_original-2*i
        n1[i]       = np.abs(np.fix(sizeTest*ratio_M4_M1)-sizeTest*ratio_M4_M1)
        n2[i]       = np.abs(np.ceil(sizeTest*ratio_M4_M1)-sizeTest*ratio_M4_M1)        
        out[i]      = np.min([n1[i],n2[i]])
    
    nCrop = np.argmin(out)
    
    
    if out.min()==0.:
        print('Down sampling the IFs without approximation... ')
    else:
        print('Down sampling the IFs with an approximation... ')
    
    # size of the cropped IFs
    resolution_M4_cropped = resolution_M4_original-2*nCrop
    
    # size of the down-scaled IFs    
    resolution_M4_scaled = int(resolution_M4_cropped*ratio_M4_M1)
       
    # allocate memory to store the influence functions
    influMap = np.zeros((resolution_M4_cropped,resolution_M4_cropped))  

#-------------------- The Following Transformations are applied in the following order -----------------------------------
   
    # 1) Down scaling to get the right pixel size according to the resolution of M1
    downScaling     = anamorphosisImageMatrix(influMap,0,[ratio_M4_M1,ratio_M4_M1])
    
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(influMap,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
    rotMatrix               = rotateImageMatrix(influMap,misReg.rotationAngle)
    shiftMatrix             = translationImageMatrix(influMap,[misReg.shiftY/pixelSize_M1,misReg.shiftX/pixelSize_M1]) #units are in m
    
    # Shift of half a pixel to center the images on an even number of pixels
    alignmentMatrix         = translationImageMatrix(influMap,[-0.5,-0.5])
    
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,order=3)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction_IF(i,j,k,n):
        n=int(n)
        # support of the IFS
        influMap = np.zeros((941,941)) 
        i=int(i)
        j=int(j)
        # fill up the support with the 2D IFS [120x120]
        influMap[i:i+120, j:j+120] = k  
        output = globalTransformation(influMap[n:-n,n:-n])
        del influMap
        
        return output
    # permute the dimensions for joblib
    influenceFunctions_M4 = np.moveaxis(influenceFunctions_M4,-1,0)
    # only consider nActu influence functions
    influenceFunctions_M4 = influenceFunctions_M4[:nActu,:,:]
    i1                          = i1[:nActu]
    j1                          = j1[:nActu]
    
    # number of pixel to crop as a vector for joblib
    nCrop_vect                  = np.zeros(nActu)
    nCrop_vect [:]              = nCrop 
    
    print('Reconstructing the influence functions ... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs=nJobs,prefer='threads')(delayed(reconstruction_IF)(i,j,k,n) for i,j,k,n in zip(i1,j1,influenceFunctions_M4,nCrop_vect))
        return Q 
    # output of the reconstruction
    influenceFunctions_M4 =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
        
    # number of pixel to be cropped after the transformation
    dN = (resolution_M4_cropped-1-resolution_M4_scaled)//2+(resolution_M4_scaled-resolution_M1)//2

    # crop the influence functions to the right support
    influenceFunctions_M4 = influenceFunctions_M4[dN:-dN-1,dN:-dN-1,:]

    # recenter the initial coordinates_M4_originalinates around 0
    coordinates_M4 = ((coordinates_M4_original-resolution_M4_original/2)*ratio_M4_M1)    
    
    # apply the transformations and re-center them for the new resolution resolution_M1
    coordinates_M4 = translation(rotation(anamorphosis(coordinates_M4,misReg.anamorphosisAngle*np.pi/180,misReg.radialScaling,misReg.tangentialScaling),-misReg.rotationAngle*np.pi/180),[misReg.shiftX/pixelSize_M1,misReg.shiftY/pixelSize_M1])+resolution_M1/2
    
    # store the influence function in sparse matrices 
    influenceFunctions_M4=np.reshape(influenceFunctions_M4,[resolution_M1*resolution_M1,nActu])
    
    b = time.time()

    print('Done! M4 influence functions computed in ' + str(b-a) + ' s!')
    
    return influenceFunctions_M4,coordinates_M4,nActu

def makeM4influenceFunctions_noPrint(pup,filename,misReg=0,nAct=0,nJobs=30,nThreads=20):
    try:
        mkl_set_num_threads(nThreads)
    except:
         print('Could not optimize the parallelisation of the code ')
    # read the FITS file of the cropped influence functions.
    hdul = pfits.open( filename )
    
    xC                          = hdul[0].header['XCENTER']
    pixelSize_M4_original       = hdul[0].header['pixsize']
    i1, j1                      = hdul[1].data
    influenceFunctions_M4 = hdul[2].data
    xpos, ypos                  = hdul[3].data
    
    # size of influence functions and number of actuators
    nx, ny, nActu = influenceFunctions_M4.shape
    
    # create a MisReg object to store the different mis-registration
    if np.isscalar(misReg):
        if misReg==0:
            misReg=MisRegistration()
        else:
            print('ERROR: wrong value for the mis-registrations')
   
    # case with less actuators (debug purposes)
    if nAct!=0:
        nActu=nAct
        
    # coordinates of M4 before the interpolation        
    coordinates_M4_original      = np.zeros([nAct,2])    
    coordinates_M4_original[:,0] = xpos[:nAct]
    coordinates_M4_original[:,1] = ypos[:nAct]
    
    # size of the influence functions maps
    resolution_M4_original       = int(np.ceil(2*xC))   
    
    # resolution of the M1 pupil
    resolution_M1                = pup.shape[1]
    
    # compute the pixel scale of the M1 pupil
    pixelSize_M1                 = 40/resolution_M1
    
    # compute the ratio_M4_M1 between both pixel scale.
    ratio_M4_M1                  = pixelSize_M4_original/pixelSize_M1
    
    # crop the IFs support to get an integer value of pixel to consider for the resampling. 
    # if no integer is possible, the smallest value is taken
    
    out = np.zeros(30)
    n1  = np.zeros(30)
    n2  = np.zeros(30)
    
    for i in range(30):
        sizeTest    = resolution_M4_original-2*i
        n1[i]       = np.abs(np.fix(sizeTest*ratio_M4_M1)-sizeTest*ratio_M4_M1)
        n2[i]       = np.abs(np.ceil(sizeTest*ratio_M4_M1)-sizeTest*ratio_M4_M1)        
        out[i]      = np.min([n1[i],n2[i]])
    
    nCrop = np.argmin(out)
    
  
    # size of the cropped IFs
    resolution_M4_cropped = resolution_M4_original-2*nCrop
    
    # size of the down-scaled IFs    
    resolution_M4_scaled = int(resolution_M4_cropped*ratio_M4_M1)
       
    # allocate memory to store the influence functions
    influMap = np.zeros((resolution_M4_cropped,resolution_M4_cropped))  

#-------------------- The Following Transformations are applied in the following order -----------------------------------
   
    # 1) Down scaling to get the right pixel size according to the resolution of M1
    downScaling     = anamorphosisImageMatrix(influMap,0,[ratio_M4_M1,ratio_M4_M1])
    
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(influMap,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
    rotMatrix               = rotateImageMatrix(influMap,misReg.rotationAngle)
    shiftMatrix             = translationImageMatrix(influMap,[misReg.shiftY/pixelSize_M1,misReg.shiftX/pixelSize_M1]) #units are in m
    
    # Shift of half a pixel to center the images on an even number of pixels
    alignmentMatrix         = translationImageMatrix(influMap,[-0.5,-0.5])
    
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,order=3)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction_IF(i,j,k,n):
        n=int(n)
        # support of the IFS
        influMap = np.zeros((941,941)) 
        i=int(i)
        j=int(j)
        # fill up the support with the 2D IFS [120x120]
        influMap[i:i+120, j:j+120] = k  
        output = globalTransformation(influMap[n:-n,n:-n])
        del influMap
        
        return output
    # permute the dimensions for joblib
    influenceFunctions_M4 = np.moveaxis(influenceFunctions_M4,-1,0)
    # only consider nActu influence functions
    influenceFunctions_M4 = influenceFunctions_M4[:nActu,:,:]
    i1                          = i1[:nActu]
    j1                          = j1[:nActu]
    
    # number of pixel to crop as a vector for joblib
    nCrop_vect                  = np.zeros(nActu)
    nCrop_vect [:]              = nCrop 
    
    def joblib_reconstruction():
        Q=Parallel(n_jobs=nJobs,prefer='threads')(delayed(reconstruction_IF)(i,j,k,n) for i,j,k,n in zip(i1,j1,influenceFunctions_M4,nCrop_vect))
        return Q 
    # output of the reconstruction
    influenceFunctions_M4 =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
        
    # number of pixel to be cropped after the transformation
    dN = (resolution_M4_cropped-1-resolution_M4_scaled)//2+(resolution_M4_scaled-resolution_M1)//2

    # crop the influence functions to the right support
    influenceFunctions_M4 = influenceFunctions_M4[dN:-dN-1,dN:-dN-1,:]

    # recenter the initial coordinates_M4_originalinates around 0
    coordinates_M4 = ((coordinates_M4_original-resolution_M4_original/2)*ratio_M4_M1)    
    
    # apply the transformations and re-center them for the new resolution resolution_M1
    coordinates_M4 = translation(rotation(anamorphosis(coordinates_M4,misReg.anamorphosisAngle*np.pi/180,misReg.radialScaling,misReg.tangentialScaling),-misReg.rotationAngle*np.pi/180),[misReg.shiftX/pixelSize_M1,misReg.shiftY/pixelSize_M1])+resolution_M1/2
    
    # store the influence function in sparse matrices 
    influenceFunctions_M4=np.reshape(influenceFunctions_M4,[resolution_M1*resolution_M1,nActu])
    
    
    return influenceFunctions_M4,coordinates_M4,nActu


def getPetalModes(tel,dm,i_petal):
    
    if np.isscalar(i_petal):
        #    initiliaze dm coef
        petal = np.zeros(dm.nValidAct)
        petal[(i_petal-1)*892:i_petal*892]=-np.ones(892)
        dm.coefs = petal
        tel*dm
        tmp = np.abs(tel.OPD) > 1
        out = tmp  
    else:
        out=np.zeros([tel.resolution,tel.resolution,len(i_petal)])
        for i in range(len(i_petal)):
            petal = np.zeros(dm.nValidAct)
            petal[(i_petal[i]-1)*892:i_petal[i]*892]=-np.ones(892)
            dm.coefs = petal
            tel*dm
            tmp = np.abs(tel.OPD) > 1
            out[:,:,i] = tmp
    out_float = out.astype(float)
    return out, out_float