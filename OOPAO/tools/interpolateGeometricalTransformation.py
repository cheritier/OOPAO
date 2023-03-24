# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:35:31 2020

@author: cheritie
"""

import numpy as np
import skimage.transform as sk
from joblib import Parallel, delayed

from .tools import bin_ndarray
from ..MisRegistration import MisRegistration


# Rotation with respect to te center of the image
def rotateImageMatrix(image,angle):
    # compute the shift value to center the image around 0
    shift_y, shift_x = np.array(image.shape[:2]) / 2.
    # apply the rotation
    tf_rotate = sk.SimilarityTransform(rotation=np.deg2rad(angle))
    # center the image around 0
    tf_shift = sk.SimilarityTransform(translation=[-shift_x, -shift_y])
    # re-center the image
    tf_shift_inv = sk.SimilarityTransform(translation=[shift_x, shift_y])
    
    return ((tf_shift + (tf_rotate + tf_shift_inv)))


# Differential scaling in X and Y with respect to the center of the image
def scalingImageMatrix(image,scaling):
    # compute the shift value to center the image around 0
    shift_y, shift_x = np.array(image.shape[:2]) / 2.
    # apply the scaling in X and Y
    tf_scaling = sk.SimilarityTransform(scale=scaling)
    # center the image around 0
    tf_shift = sk.SimilarityTransform(translation=[-shift_x, -shift_y])
    # re-center the image
    tf_shift_inv = sk.SimilarityTransform(translation=[shift_x, shift_y])
    
    return ((tf_shift + (tf_scaling + tf_shift_inv)))

# Shift in X and Y
def translationImageMatrix(image,shift):
    # translate the image with the corresponding shift value
    tf_shift = sk.SimilarityTransform(translation=shift)    
    return tf_shift

# Anamorphosis = composition of a rotation, a scaling in X and Y and an anti-rotation
def anamorphosisImageMatrix(image,direction,scale):
    # Rotate the image
    matRot  = rotateImageMatrix(image,direction)
    # Apply the X and Y scaling
    matShearing = scalingImageMatrix(image,scaling=scale)
    # De-Rotate the image
    matAntiRot  = rotateImageMatrix(image,-direction)  
        
    return matRot+matShearing+matAntiRot

def translation(coord,shift):
    x=coord[:,0]
    y=coord[:,1]
    xOut =   x+shift[0]
    yOut =   y+shift[1]
    
    coordOut=np.copy(coord)
    coordOut[:,0]=xOut
    coordOut[:,1]=yOut
    
    return coordOut

def rotation(coord,angle):
    x=coord[:,0]
    y=coord[:,1]
    xOut =   x*np.cos(angle)-y*np.sin(angle)
    yOut =   y*np.cos(angle)+x*np.sin(angle)
    coordOut=np.copy(coord)
    coordOut[:,0] = xOut
    coordOut[:,1] = yOut
    
    return coordOut

def anamorphosis(coord,angle,mNorm,mRad):
    x       = coord[:,0]
    y       = coord[:,1]
    mRad   += 1
    mNorm  += 1
    xOut    =   x * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  y * (mNorm*np.sin(2*angle)/2  -mRad*np.sin(2*angle)/2)
    yOut    =   y * (mNorm*np.cos(angle)**2  + mRad* np.sin(angle)**2)  +  x * (mNorm*np.sin(2*angle)/2  -mRad*np.sin(2*angle)/2)
    
    coordOut      = np.copy(coord)
    coordOut[:,0] = xOut
    coordOut[:,1] = yOut
    
    return coordOut





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% START OF THE FUNCTION   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def interpolateGeometricalTransformation(data,misReg=0, order =3):

    # size of influence functions and number of actuators
    nx, ny, nData = data.shape
    
    # create a MisReg object to store the different mis-registration
    if np.isscalar(misReg):
        if misReg==0:
            misReg=MisRegistration()
        else:
            print('ERROR: wrong value for the mis-registrations')
    
   
    # 2) transformations for the mis-registration
    anamMatrix  = anamorphosisImageMatrix(data,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
    rotMatrix   = rotateImageMatrix(data,misReg.rotationAngle)
    shiftMatrix = translationImageMatrix(data,[misReg.shiftX,misReg.shiftY]) #units are in pixel of the M1
    
    # Global transformation matrix
    transformationMatrix =  anamMatrix + rotMatrix + shiftMatrix 
    
    data = np.moveaxis(np.asarray(data),-1,0)

    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,order=order,mode='constant',cval = 0)
            return output
    
    def joblib_transformation():
        Q=Parallel(n_jobs=8,prefer='threads')(delayed(globalTransformation)(i) for i in data)
        return Q 
    
    print('Applying the transformations ... ')

    maps = joblib_transformation()
    # change axis order
    dataOut = np.moveaxis(np.asarray(np.asarray(maps)),0,-1)

    print('Done! ')

    
    return dataOut
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:04:46 2021

@author: cheritie
"""


def interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out, shape_out = None, mis_registration = None, order = 1, joblib_prefer = 'threads', joblib_nJobs = 4):
    if mis_registration is None:
        mis_registration = MisRegistration()
    nAct,nx, ny = cube_in.shape  
             
    # size of the influence functions maps
    resolution_in       = int(nx)   
        
    # compute the ratio between both pixel scale.
    ratio                  = pixel_size_in/pixel_size_out
    # after the interpolation the image will be shifted of a fraction of pixel extra if ratio is not an integer
    extra = (ratio)%1 
    
    # difference in pixels between both resolutions    
    nPix = resolution_in-resolution_out
    
    
    extra = extra/2 + (np.floor(ratio)-1)*0.5
    nCrop =  (nPix/2)
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
    alignmentMatrix         = translationImageMatrix(influMap,[extra-nCrop,extra-nCrop])
        
    # 3) Global transformation matrix
    transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    def globalTransformation(image):
            output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=order)
            return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction(map_2D):
        output = globalTransformation(map_2D)  
        return output
    
    # print('interpolating... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs = joblib_nJobs,prefer = joblib_prefer)(delayed(reconstruction)(i) for i in cube_in)
        return Q 
    
    cube_out =  np.asarray(joblib_reconstruction())
    # print('...Done!')    

    return cube_out

def interpolate_image(image_in, pixel_size_in, pixel_size_out,resolution_out, rotation_angle = 0, shift_x = 0,shift_y = 0,anamorphosisAngle=0,tangentialScaling=0,radialScaling=0, shape_out = None, order = 1):

        nx, ny = image_in.shape  
                 
        # size of the influence functions maps
        resolution_in       = int(nx)   
            
        # compute the ratio between both pixel scale.
        ratio                  = pixel_size_in/pixel_size_out
        # after the interpolation the image will be shifted of a fraction of pixel extra if ratio is not an integer
        extra = (ratio)%1 
        
        # difference in pixels between both resolutions    
        nPix = resolution_in-resolution_out
        
        
        extra = extra/2 + (np.floor(ratio)-1)*0.5
        nCrop =  (nPix/2)
        # allocate memory to store the influence functions
        influMap = np.zeros([resolution_in,resolution_in])  
        
        #-------------------- The Following Transformations are applied in the following order -----------------------------------
           
        # 1) Down scaling to get the right pixel size according to the resolution of M1
        downScaling     = anamorphosisImageMatrix(influMap,0,[ratio,ratio])
        
        # 2) transformations for the mis-registration
        anamMatrix              = anamorphosisImageMatrix(influMap,anamorphosisAngle,[1+radialScaling,1+tangentialScaling])
        rotMatrix               = rotateImageMatrix(influMap,rotation_angle)
        shiftMatrix             = translationImageMatrix(influMap,[shift_x/pixel_size_out,shift_y/pixel_size_out]) #units are in m
        
        # Shift of half a pixel to center the images on an even number of pixels
        alignmentMatrix         = translationImageMatrix(influMap,[extra-nCrop,extra-nCrop])
            
        # 3) Global transformation matrix
        transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
        
        def globalTransformation(image):
                output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=order)
                return output
        
        # definition of the function that is run in parallel for each 
        def reconstruction(map_2D):
            output = globalTransformation(map_2D)  
            return output
                
        image_out =  reconstruction(image_in)
        # print('...Done!')    
    
        return image_out
    
    
    
    
    
def binning_optimized(cube_in,binning_factor):
    n_im, nx,ny = np.shape(cube_in)
    
    if nx%binning_factor==0 and binning_factor%1==0:
        # in case the binning factor gives an integer number of pixels
        cube_out =  bin_ndarray(cube_in,[n_im, nx//binning_factor,ny//binning_factor], operation='sum')        
    else:
        # size of the cube maps
        resolution_in       = int(nx)   
        resolution_out = int(np.ceil(resolution_in/binning_factor))

        pixel_size_in   = resolution_in
        pixel_size_out   = resolution_out
        
        # compute the ratio between both pixel scale.
        ratio                  = pixel_size_in/pixel_size_out
        # after the interpolation the image will be shifted of a fraction of pixel extra if ratio is not an integer
        extra = (ratio)%1 
        
        # difference in pixels between both resolutions    
        nPix = resolution_in-resolution_out
        
        
        extra = extra/2 + (np.floor(ratio)-1)*0.5
        nCrop =  (nPix/2)
        # allocate memory to store the influence functions
        influMap = np.zeros([resolution_in,resolution_in])  
        
        #-------------------- The Following Transformations are applied in the following order -----------------------------------
           
        # 1) Down scaling to get the right pixel size according to the resolution of M1
        downScaling     = anamorphosisImageMatrix(influMap,0,[ratio,ratio])
        
        # 2) transformations for the mis-registration
        anamMatrix              = anamorphosisImageMatrix(influMap,1,[1,1])

        # Shift of half a pixel to center the images on an even number of pixels
        alignmentMatrix         = translationImageMatrix(influMap,[extra-nCrop,extra-nCrop])
            
        # 3) Global transformation matrix
        transformationMatrix    = downScaling + anamMatrix + alignmentMatrix
        
        def globalTransformation(image):
                output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=1)
                return output
        
        # definition of the function that is run in parallel for each 
        def reconstruction(map_2D):
            output = globalTransformation(map_2D)  
            return output
        
        # print('interpolating... ')    
        def joblib_reconstruction():
            Q=Parallel(n_jobs = 4,prefer = 'threads')(delayed(reconstruction)(i) for i in cube_in)
            return Q 
        
        cube_out =  np.asarray(joblib_reconstruction())
    
    return cube_out
        
        
        
     

def interpolate_cube_special(cube_in, sx, sy, pixel_size_in, pixel_size_out, resolution_out, shape_out = None, mis_registration = None, order = 1, joblib_prefer = 'threads', joblib_nJobs = 4):
    if mis_registration is None:
        mis_registration = MisRegistration()
    nAct,nx, ny = cube_in.shape  
             
    sx_vect = sx
    sy_vect = sy
    # size of the influence functions maps
    resolution_in       = int(nx)   
        
    # compute the ratio between both pixel scale.
    ratio                  = pixel_size_in/pixel_size_out
    # after the interpolation the image will be shifted of a fraction of pixel extra if ratio is not an integer
    extra = (ratio)%1 
    
    # difference in pixels between both resolutions    
    nPix = resolution_in-resolution_out
    
    
    extra = extra/2 + (np.floor(ratio)-1)*0.5
    nCrop =  (nPix/2)
    # allocate memory to store the influence functions
    influMap = np.zeros([resolution_in,resolution_in])  
    
    #-------------------- The Following Transformations are applied in the following order -----------------------------------
       
    # 1) Down scaling to get the right pixel size according to the resolution of M1
    downScaling     = anamorphosisImageMatrix(influMap,0,[ratio,ratio])
    
    # 2) transformations for the mis-registration
    anamMatrix              = anamorphosisImageMatrix(influMap,mis_registration.anamorphosisAngle,[1+mis_registration.radialScaling,1+mis_registration.tangentialScaling])
    rotMatrix               = rotateImageMatrix(influMap,mis_registration.rotationAngle)
    # shiftMatrix             = translationImageMatrix(influMap,[mis_registration.shiftY/pixel_size_out,mis_registration.shiftX/pixel_size_out]) #units are in m
    
    # Shift of half a pixel to center the images on an even number of pixels
    alignmentMatrix         = translationImageMatrix(influMap,[extra-nCrop,extra-nCrop])
        
    # 3) Global transformation matrix
    # transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
    
    # def globalTransformation(image,sx,sy):
    #         output  = sk.warp(image,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=order)
    #         return output
    
    # definition of the function that is run in parallel for each 
    def reconstruction(map_2D,sx,sy):
        mis_registration.shiftX = sx    
        mis_registration.shiftY = sy

        shiftMatrix             = translationImageMatrix(influMap,[mis_registration.shiftY/pixel_size_out,mis_registration.shiftX/pixel_size_out]) #units are in m
        transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
        output =  sk.warp(map_2D,(transformationMatrix).inverse,output_shape = [resolution_out,resolution_out],order=order)
        return output
    
    # print('interpolating... ')    
    def joblib_reconstruction():
        Q=Parallel(n_jobs = joblib_nJobs,prefer = joblib_prefer)(delayed(reconstruction)(i,j,k) for i,j,k in zip(cube_in,sx_vect,sy_vect))
        return Q 
    
    cube_out =  np.asarray(joblib_reconstruction())
    # print('...Done!')    

    return cube_out   
        
        