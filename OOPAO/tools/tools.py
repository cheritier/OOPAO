# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:33:20 2020

@author: cheritie
"""

import json
import os
import subprocess

import jsonpickle
import numpy as np
import skimage.transform as sk
from astropy.io import fits as pfits


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USEFUL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def print_(input_text,condition):
    if condition:
        print(input_text)
        

def createFolder(path):
    
    if path.rfind('.') != -1:
        path = path[:path.rfind('/')+1]
        
    try:
        os.makedirs(path)
    except OSError:
        if path:
            path =path
        else:
            print ("Creation of the directory %s failed:" % path)
            print('Maybe you do not have access to this location.')
    else:
        print ("Successfully created the directory %s !" % path)

def emptyClass():
    class nameClass:
        pass
    return nameClass

def bsxfunMinus(a,b):      
    A =np.tile(a[...,None],len(b))
    B =np.tile(b[...,None],len(a))
    out = A-B.T
    return out

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def translationImageMatrix(image,shift):
    # translate the image with the corresponding shift value
    tf_shift = sk.SimilarityTransform(translation=shift)    
    return tf_shift

def globalTransformation(image,shiftMatrix,order=3):
        output  = sk.warp(image,(shiftMatrix).inverse,order=order)
        return output


def reshape_2D(A,axis = 2, pupil=False ):
    if axis ==2:
        out = np.reshape(A,[A.shape[0]*A.shape[1],A.shape[2]])
    else:
        out = np.reshape(A,[A.shape[0],A.shape[1]*A.shape[2]])
    if pupil:
        out = np.squeeze(out[pupil,:])
    return out


def reshape_3D(A,axis = 1 ):
    if axis ==1:
        dim_rep =np.sqrt(A.shape[0]) 
        out = np.reshape(A,[dim_rep,dim_rep,A.shape[1]])
    else:
        dim_rep =np.sqrt(A.shape[1]) 
        out = np.reshape(A,[A.shape[0],dim_rep,dim_rep])    
    return out        


def read_json(filename):
    with open(filename ) as f:
        C = json.load(f)
    
    data = jsonpickle.decode(C) 

    return data

def read_fits(filename , dim = 0):
    hdu  = pfits.open(filename)
    if dim == 0:
        try:
            data = np.copy(hdu[1].data)
        except:
            data = np.copy(hdu[0].data)
    else:
        
        data = hdu[dim].data
    hdu.close()
    del hdu[0].data
    
    
    return data

    
def write_fits(data, filename , header_name = '',overwrite=True):
    
    hdr = pfits.Header()
    hdr['TITLE'] = header_name
    
    empty_primary = pfits.PrimaryHDU(header = hdr)
    
    primary_hdu = pfits.ImageHDU(data)
    
    hdu = pfits.HDUList([empty_primary, primary_hdu])
    
    hdu.writeto(filename,overwrite = overwrite)
    hdu.close()
    del hdu[0].data
    
def findNextPowerOf2(n):
 
    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
 
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1       # unset rightmost bit
 
    # `n` is now a power of two (less than `n`)
 
    # return next power of 2
    return n << 1
      
def centroid(image, threshold = 0):
    im = np.copy(image)
    im[im<threshold]=0
    x = 0
    y = 0
    s = im.sum()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x+=im[i,j]*j/s
            y+=im[j,i]*j/s
            
    return x,y    


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def compute_fourier_mode(pupil,spatial_frequency,angle_deg,zeropadding = 2):
    N = pupil.shape[0]
    mode = np.zeros([N,N])
    
    t = spatial_frequency*zeropadding/2
    Z = np.zeros([N,N],'complex')

    thet = angle_deg
    
    Z[N//2+int(t*np.cos(np.deg2rad(thet)+np.pi)),N//2+int(t*np.sin(np.deg2rad(thet)+np.pi))]=1
    Z[N//2-int(t*np.cos(np.deg2rad(thet)+np.pi)),N//2-int(t*np.sin(np.deg2rad(thet)+np.pi))]=-100000
    
    support = np.zeros([N*zeropadding,N*zeropadding],dtype='complex')
    
    
    center = zeropadding*N//2
    support [center-N//2:center+N//2,center-N//2:center+N//2]=Z
    F = np.fft.ifft2(support)
    F= F[center-N//2:center+N//2,center-N//2:center+N//2]    
    # normalisation
    S= np.abs(F)/np.max(np.abs(F))
    S=S - np.mean([S.max(), S.min()])
    mode = S/S.std()
    
    return mode

#%%
def circularProfile(img):
    # Compute circular average profile from an image, reference to center of image
    # Get image parameters
    a = img.shape[0]
    b = img.shape[1]
    # Image center
    cen_x = a//2
    cen_y = b//2
    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(b) - cen_x, np.arange(a) - cen_y)
    R = np.sqrt(np.square(X) + np.square(Y))
    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))
    index = 0
    bin_size = 1
    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = img[mask]
        intensity[index] = np.mean(values)
        index += 1
    return intensity

