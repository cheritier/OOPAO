# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:35:31 2020

@author: cheritie
"""

import numpy as np
import skimage.transform as sk
from joblib import Parallel, delayed
from AO_modules.MisRegistration import MisRegistration

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

def interpolateGeometricalTransformation(data,misReg=0):

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
            output  = sk.warp(image,(transformationMatrix).inverse,order=3,mode='constant',cval = 0)
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
        