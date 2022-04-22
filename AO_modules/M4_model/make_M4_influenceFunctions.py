# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:31:25 2020

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
        import mkl
        mkl_set_num_threads = mkl.set_num_threads#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF THE FUNCTION   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def makeM4influenceFunctions(pup, filename, misReg = 0, dm = None, nAct = 0, nJobs = 30, nThreads = 20, order =1,floating_precision=64, D = 40):
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
    influenceFunctions_M4       = hdul[2].data
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
    pixelSize_M1                 = D/resolution_M1
    
    # compute the ratio_M4_M1 between both pixel scale.
    ratio_M4_M1                  = pixelSize_M4_original/pixelSize_M1
    # after the interpolation the image will be shifted of a fraction of pixel extra if ratio_M4_M1 is not an integer
    extra = (ratio_M4_M1)%1 
    
    nPix = (resolution_M4_original-resolution_M1)
    
    extra = extra/2 + (np.floor(ratio_M4_M1)-1)*0.5
    nCrop =  (nPix/2)
    
               
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
    alignmentMatrix         = translationImageMatrix(influMap,[-nCrop + extra,-nCrop + extra])
        
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_M1,resolution_M1],order=order)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction_IF(i,j,k):
  
        i=int(i)
        j=int(j)      
        
        if floating_precision==32:
            # support of the IFS
            influMap = np.zeros([resolution_M4_original,resolution_M4_original],dtype=np.float32) 
            # fill up the support with the 2D IFS [120x120]
            influMap[i:i+nx, j:j+ny] = np.float32(k)  
        else:
            influMap = np.zeros([resolution_M4_original,resolution_M4_original],dtype=np.float64) 
            # fill up the support with the 2D IFS [120x120]
            influMap[i:i+nx, j:j+ny] = k  

        output = globalTransformation(influMap)

        del influMap
        return output
    
    # permute the dimensions for joblib
    influenceFunctions_M4 = np.moveaxis(influenceFunctions_M4,-1,0)
    # only consider nActu influence functions
    influenceFunctions_M4 = influenceFunctions_M4[:nActu,:,:]
    i1                          = i1[:nActu]
    j1                          = j1[:nActu] 

    
    print('Reconstructing the influence functions ... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs=nJobs,prefer='threads')(delayed(reconstruction_IF)(i,j,k) for i,j,k in zip(i1,j1,influenceFunctions_M4))
        return Q 
    # output of the reconstruction
    influenceFunctions_M4 =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
    
    # recenter the initial coordinates_M4_originalinates around 0
    coordinates_M4 = ((coordinates_M4_original-resolution_M4_original/2)*ratio_M4_M1)    
    
    # apply the transformations and re-center them for the new resolution resolution_M1
    coordinates_M4 = translation(rotation(anamorphosis(coordinates_M4,misReg.anamorphosisAngle*np.pi/180,misReg.radialScaling,misReg.tangentialScaling),-misReg.rotationAngle*np.pi/180),[misReg.shiftX/pixelSize_M1,misReg.shiftY/pixelSize_M1])+resolution_M1/2
    
    # store the influence function in sparse matrices 
    if dm is None:
        influenceFunctions_M4 = np.reshape(influenceFunctions_M4,[resolution_M1*resolution_M1,nActu])
        return influenceFunctions_M4,coordinates_M4,nActu
    else:
        dm.modes = np.reshape(influenceFunctions_M4,[resolution_M1*resolution_M1,nActu])
        return coordinates_M4
        


def getPetalModes(tel,dm,i_petal):
    try:
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
    except:
        print('Error, the numbe or actuator does not match 892 actuator per petal.. returning an empty list.')
        out = []
        out_float = []
    return out, out_float