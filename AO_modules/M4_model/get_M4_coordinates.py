# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:11:45 2020

@author: cheritie
"""


import numpy as np
import ctypes
from astropy.io import fits as pfits
from AO_modules.MisRegistration import MisRegistration
from AO_modules.tools.interpolateGeometricalTransformation import rotation,translation,anamorphosis

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

def get_M4_coordinates(pup,filename,misReg=0,nAct=0,nJobs=30,nThreads=20):
    
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
    coordinates_M4_original      = np.zeros([nActu,2])    
    coordinates_M4_original[:,0] = xpos[:nActu]
    coordinates_M4_original[:,1] = ypos[:nActu]
    
    # size of the influence functions maps
    resolution_M4_original       = int(np.ceil(2*xC))   
    
    # resolution of the M1 pupil
    resolution_M1                = pup.shape[1]
    
    # compute the pixel scale of the M1 pupil
    pixelSize_M1                 = 40/resolution_M1
    
    # compute the ratio_M4_M1 between both pixel scale.
    ratio_M4_M1                  = pixelSize_M4_original/pixelSize_M1
   
    # recenter the initial coordinates_M4_originalinates around 0
    coordinates_M4 = ((coordinates_M4_original-resolution_M4_original/2)*ratio_M4_M1)    
    
    # apply the transformations and re-center them for the new resolution resolution_M1
    coordinates_M4_m = translation(rotation(anamorphosis(coordinates_M4,misReg.anamorphosisAngle*np.pi/180,misReg.radialScaling,misReg.tangentialScaling),-misReg.rotationAngle*np.pi/180),[misReg.shiftX/pixelSize_M1,misReg.shiftY/pixelSize_M1])+resolution_M1/2
    
    coordinates_M4_pix = ((coordinates_M4_m-resolution_M1//2)/(resolution_M1//2))*20

    return np.flip(coordinates_M4_m), np.flip(coordinates_M4_pix)

