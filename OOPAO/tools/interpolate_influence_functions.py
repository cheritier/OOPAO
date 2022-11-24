# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:04:46 2021

@author: cheritie
"""

import numpy as np
import skimage.transform as sk
from joblib import Parallel, delayed

from .interpolateGeometricalTransformation import (anamorphosis,
                                                   anamorphosisImageMatrix,
                                                   rotateImageMatrix,
                                                   rotation,
                                                   translation,
                                                   translationImageMatrix)


def interpolate_influence_functions(influence_functions_in, pixel_size_in, pixel_size_out, resolution_out, mis_registration, coordinates_in = None, order = 1):
    
    nAct,nx, ny = influence_functions_in.shape  
             
    # size of the influence functions maps
    resolution_in       = int(nx)   
        
    # compute the ratio between both pixel scale.
    ratio                  = pixel_size_in/pixel_size_out
    
    # after the interpolation the image will be shifted of a fraction of pixel extra if ratio is not an integer
    extra = (ratio)%1 
    
    # difference in pixels between both resolutions    
    nPix = resolution_in-resolution_out
    
    if nPix%2==0:
        # case nPix is even
        # alignement of the array with respect to the interpolation 
        # (ratio is not always an integer of pixel)
        extra_x = extra/2 -0.5
        extra_y = extra/2 -0.5
        
        # crop one extra pixel on one side
        nCrop_x = nPix//2
        nCrop_y = nPix//2
    else:
        # case nPix is uneven
        # alignement of the array with respect to the interpolation 
        # (ratio is not always an integer of pixel)
        extra_x = extra/2 -0.5 -0.5
        extra_y = extra/2 -0.5 -0.5
        # crop one extra pixel on one side
        nCrop_x = nPix//2
        nCrop_y = (nPix//2)+1
           
    # allocate memory to store the influence functions
    influMap = np.zeros([resolution_in,resolution_in])  
    
    #-------------------- The Following Transformations are applied in the following order -----------------------------------
       
    # 1) Down scaling to get the right pixel size according to the resolution of M1
    downScaling     = anamorphosisImageMatrix(influMap,0,[ratio,ratio])
    
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(influMap,mis_registration.anamorphosisAngle,[1+mis_registration.radialScaling,1+mis_registration.tangentialScaling])
    rotMatrix               = rotateImageMatrix(influMap,mis_registration.rotationAngle)
    shiftMatrix             = translationImageMatrix(influMap,[mis_registration.shiftY/pixel_size_out,mis_registration.shiftX/pixel_size_out]) #units are in m
    
    # Shift of half a pixel to center the images on an even number of pixels
    alignmentMatrix         = translationImageMatrix(influMap,[extra_x-nCrop_x,extra_y-nCrop_y])
        
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=order)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction_IF(influMap):
        output = globalTransformation(influMap)  
        return output
    
    print('Reconstructing the influence functions ... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs=4,prefer='threads')(delayed(reconstruction_IF)(i) for i in influence_functions_in)
        return Q 
    
    influence_functions_out =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
    if coordinates_in is not None:
        # recenter the initial coordinates_ininates around 0
        coordinates_out = ((coordinates_in-resolution_in/2)*ratio)    
        
        # apply the transformations and re-center them for the new resolution resolution_out
        coordinates_out = translation(rotation(anamorphosis(coordinates_out,mis_registration.anamorphosisAngle*np.pi/180,mis_registration.radialScaling,mis_registration.tangentialScaling),-mis_registration.rotationAngle*np.pi/180),[mis_registration.shiftX/pixel_size_out,mis_registration.shiftY/pixel_size_out])+resolution_out/2
        
        
        return influence_functions_out, coordinates_out
    else:
        return influence_functions_out