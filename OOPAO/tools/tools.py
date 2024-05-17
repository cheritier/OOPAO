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
from OOPAO.tools import *
import matplotlib.pyplot as plt


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USEFUL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def crop(imageArray, size, axis, maximum = 0):
    """
    

    Parameters
    ----------
    imageArray : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.
    maximum : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # returns an subimage centered on CENTER (assumed to be the center of image), 
    # of size SIZE (integer), 
    # considering any datacube IMAGEARRAY (np array) of size NxMxL
    if imageArray.ndim == 2:
        sizeImage = imageArray.shape[0]
        center = np.array((sizeImage//2, sizeImage//2))
        if maximum == 1:
            center = np.int16(centroid(imageArray)) + np.int16(sizeImage/2)
            print(center)
        return imageArray[center[1]-size//2:center[1]+size//2,center[0]-size//2:center[0]+size//2]

    if imageArray.ndim == 3:
        if axis == 0:
            sizeImage = imageArray.shape[1]
            center = np.array((sizeImage//2, sizeImage//2))
            return imageArray[:,center[0]-size//2:center[0]+size//2,center[1]-size//2:center[1]+size//2]
        if axis == 1:
            sizeImage = imageArray.shape[0]
            center = np.array((sizeImage//2, sizeImage//2))
            return imageArray[center[0]-size//2:center[0]+size//2,:,center[1]-size//2:center[1]+size//2]
        if axis == 2:
            sizeImage = imageArray.shape[0]
            center = np.array((sizeImage//2, sizeImage//2))
            return imageArray[center[0]-size//2:center[0]+size//2,center[1]-size//2:center[1]+size//2, :]

def zero_pad_array(array,padding:int):
    """
    

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    padding : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    support : TYPE
        DESCRIPTION.

    """
    if np.ndim(array)!=2:
        raise ValueError('Input array should be of shape 2D')
    else:
        support = np.zeros([int(2*padding + array.shape[0]),int(2*padding + array.shape[1])],dtype =array.dtype )
        center_x = support.shape[0]//2
        center_y = support.shape[1]//2
        support[center_x-array.shape[0]//2:center_x+array.shape[0]//2,center_y-array.shape[1]//2:center_y+array.shape[1]//2] = array.copy()
        return support
    # else:
    #     if axis is None or len(axis)!=0:
    #         raise ValueError('The axis on which the padding should be applied is not specified or not properly set, specify 2 axis aver which apply the padding. For instance, axis = [0,1] to pad the 2 first dimensions')
    #     else: 
    #         if axis == [0,1]:
    #             np.tile(array[])
    #         elif
            
                
                
    
    


def strehlMeter(PSF, tel, zeroPaddingFactor = 2, display = True, title = ''):
    # Measures the Strehl ratio from a focal plane image PSF
    # Method : compute the ratio of the OTF on the OTF of Airy pattern
    # Airy pattern is computed from tel.pupil function, with a sampling of zeroPaddingFactor
    
    # Compute Airy pattern : zero OPD PSF from tel objet
    tel.resetOPD()
    tel.computePSF(zeroPaddingFactor)    
    Airy = tel.PSF
    sizeAiry = Airy.shape[0]
    sizePSF  = PSF.shape[0]
    sizeMin  = np.min((sizeAiry, sizePSF))
    Airy = crop(Airy, np.int16(sizeMin), axis = 3)
    PSF  = crop(PSF,  np.int16(sizeMin), axis = 3)
        
    # Compute OTF for PSF and Airy
    OTF  = np.abs(np.fft.fftshift(np.fft.fft2(PSF)))
    OTFa = np.abs(np.fft.fftshift(np.fft.fft2(Airy)))
    OTF = OTF / np.max(OTF)
    OTFa = OTFa / np.max(OTFa)
    # Compute intensity profiles
    profile  = circularProfile(OTF)
    profilea = circularProfile(OTFa)
    profilea = profilea / np.max(profilea)
    profile  = profile / np.max(profile)
    profilea = profilea[0:np.int64(tel.resolution * zeroPaddingFactor/2)]
    profile  = profile[0:np.int64(tel.resolution * zeroPaddingFactor/2)]

    if display:        
        # plot OTF profiles for visualization
        plt.figure()
        xArray  = np.linspace(0,1, np.int64(tel.resolution * zeroPaddingFactor/2))
        plt.semilogy(xArray, profile, label = 'OTF', linewidth = '2')
        plt.semilogy(xArray, profilea, label = 'Perfect OTF')
        plt.title(title + ' Strehl ' + str(np.round(np.sum(OTF) / np.sum(OTFa) * 100, 1)))
        plt.legend()
        plt.ylim((1e-4,1))
        plt.xlim((0, 1))
        plt.xlabel('Spatial frequency in the pupil [D/Lambda]', fontsize = 15)
        plt.ylabel('OTF profile [normalized to peak]', fontsize = 15)
        plt.pause(0.01)
        plt.show()
        
        # plt.yscale('log')
        # plt.imshow(tel.pupil, extent = [0.6, 0.8, 0.6, 0.8])
        # plt.text(0.65,0.575, 'Pupil')
        # plt.imshow(np.log(crop(PSF, np.int16(8 * zeroPaddingFactor), 3, 1)), extent = [0.75, 0.95, 0.3, 0.5])
        # plt.text(0.8,0.275, 'PSF')
        # plt.imshow(np.log(crop(Airy, np.int16(8 * zeroPaddingFactor), 3)), extent = [0.5, 0.7, 0.3, 0.5])
        # plt.text(0.55,0.275, 'Airy')
        # plt.title('Strehl ' + np.str_(round(np.sum(OTF) / np.sum(OTFa) * 100)) + '%')
    print('Strehl ratio [%] : ', np.sum(OTF) / np.sum(OTFa) * 100)
    return np.sum(OTF) / np.sum(OTFa) * 100

        
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

def circularProfile(img, maximum = False):
    # Compute circular average profile from an image, reference to center of image
    # Get image parameters
    a = img.shape[0]
    b = img.shape[1]
    # Image center
    cen_x = a//2
    cen_y = b//2

    if maximum == True:
        cogMaximum = centroid(img,threshold = 0.1*img.max())
        cen_x = cogMaximum[0] + a/2
        cen_y = cogMaximum[1] + b/2
        print('Maximum position = [px]')
        print(cen_x, cen_y)
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

def set_binning( array, binning_factor,mode='sum'):
    if array.shape[0]%binning_factor == 0:
        if array.ndim == 2:
            new_shape = [int(np.round(array.shape[0]/binning_factor)), int(np.round(array.shape[1]/binning_factor))]
            shape = (new_shape[0], array.shape[0] // new_shape[0], 
                     new_shape[1], array.shape[1] // new_shape[1])
            if mode == 'sum':
                return array.reshape(shape).sum(-1).sum(1)
            else:
                return array.reshape(shape).mean(-1).mean(1)
        else:
            new_shape = [int(np.round(array.shape[0]/binning_factor)), int(np.round(array.shape[1]/binning_factor)), array.shape[2]]
            shape = (new_shape[0], array.shape[0] // new_shape[0], 
                     new_shape[1], array.shape[1] // new_shape[1], new_shape[2])
            if mode == 'sum':
                return array.reshape(shape).sum(-2).sum(1)
            else:
                return array.reshape(shape).mean(-2).mean(1)
    else:
        raise ValueError('Binning factor %d not compatible with the array size'%(binning_factor))