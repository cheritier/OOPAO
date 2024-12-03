# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:31:46 2024

@author: astriffl
"""

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, fft2, ifft2
from tqdm import tqdm

class GainSensingCamera:
    
    def __init__(self, mask: np.array, basis: np.array, n_jobs: int = 10) -> None:
        """
        Class gain sensing camera. Allows to compute optical gains using focal
        plane images of the modulated PSF combined with a convolutional based
        analytical model. This class is made to work with the Pyramid class.
        Reference: Chambouleyron et al. 2021, A&A
        
        Parameters
        ----------
        mask : np.array
            Complex mask of the pyramid.
        basis : np.array
            Modal basis, should be [n_pix x n_pix x n_modes].
        n_jobs : int, optional
            Number of ffts made in parallel. The default is 10.
            
        Exemple
        -------
        1. Setting a focal plane camera to the right dimensions in the OOPAO 
        Pyramid object:
        >>> wfs.focal_plane_camera.resolution = wfs.nRes
        
        2. Creation of the GSC object:
        >>> gsc = GainSensingCamera(wfs.mask, modal_basis, n_jobs = 10)
        
        3. Calibration of the GSC:
            - Reset telescope OPD to calibrate around diffraction
        >>> tel.resetOPD()
            - Propagate light to the Pyramid
        >>> tel * wfs
            - Propagate light to the focal plane camera
        >>> wfs * wfs.focal_plane_camera
            - Calibrate the GSC (first mul of focal plane camera and GSC is
                                 for calibration)
        >>> wfs.focal_plane_camera * gsc
        
        4. Compute optical gains around a random phase (ex on turbulence):
        >>> tel.OPD = atm.OPD.copy()
        >>> tel * wfs
        >>> wfs * wfs.focal_plane_camera
        >>> wfs.focal_plane_camera * gsc
        
        5. Results of optical gains are in variable gsc.og [n_modes]
        
        6. To reset the calibration (to calibrate around another working point
                                     for exemple):
        >>> gsc.reset_calibration()
            - Then start again to step 3.
        
        Display properties
        ------------------
        The properties of the GSC are derived from the properties of the focal 
        plane camera
            
        """
        self.mask = mask
        self.basis = basis
        self.n_modes = self.basis.shape[-1]
        self.n_jobs = n_jobs
        self.calibration_ready = False
        self.detector_properties = None
        self.tag = 'GSC'
    
    def calibration(self, frame: np.array) -> None:
        """
        Calibration process of the GSC

        Parameters
        ----------
        frame : np.array
            frame from the focal plane camera.
        """
        frame = frame/frame.sum()
        print('\nCalibration started')
        self.basis_product = split_basis_product(padder(self.basis, self.mask.shape[0]), self.n_jobs)
        self.IR_calib = impulse_response(self.mask, frame)
        self.sensi_calib = sensitivity(self.IR_calib, self.IR_calib, self.basis_product)
        print('GSC is calibrated \n')
        self.calibration_ready = True
        self.__repr__()
        
    def reset_calibration(self) -> None:
        """
        Reset the calibration and set vector of optical gains to 1
        """
        self.calibration_ready = False
        self.og = np.ones(self.n_modes)
    
    def compute_optical_gains(self, frame: np.array) -> None:
        """
        Compute the optical gains value 

        Parameters
        ----------
        frame : np.array
            frame from the focal plane camera.

        """
        if self.calibration_ready is False:
            raise AttributeError('Optical gains must be initialized first!')
        else:
            frame = frame/frame.sum()
            self.IR_sky = impulse_response(self.mask, frame)
            self.og = optical_gains(self.IR_sky, self.IR_calib, self.basis_product, self.sensi_calib)
    
    def __repr__(self) -> str:
        if self.detector_properties is None:
            prop = f'{"Calibrated":<16}|{str(self.calibration_ready):^7}\n'
            prop += f'{"Number of modes":<16}|{self.n_modes:^7}\n'
            n_char = len(max(prop.split('\n'), key=len))
        else:
            prop = f'{"Calibrated":<25}|{str(self.calibration_ready):^9}\n'
            prop += f'{"Number of modes":<25}|{self.n_modes:^9}\n'
            n_char = max(len(max(self.detector_properties.values())), len(max(prop.split('\n'), key=len)))
            for i in range(len(self.detector_properties.values())):
                prop += list(self.detector_properties.values())[i] + '\n'
        title = f'{"GSC":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n' 
        table = title + prop + end_line
        return table
    

def padder(array: np.array, size_out: int) -> np.array:
    if size_out > array.shape[0]:
        pad = (size_out - array.shape[0]) // 2
        array_padded = np.pad(array, ((pad,pad),(pad,pad),(0,0)))
    else:
        raise ValueError(f'Output size {size_out} should be higher than input size {array.shape[0]}')
    return array_padded

def fft3(array: np.array) -> np.array:
    return fftshift(fftn(fftshift(array, axes=(0,1)), axes=(0,1)), axes=(0,1)) / array.shape[0]

def ifft3(array: np.array) -> np.array:
    return fftshift(ifftn(fftshift(array, axes=(0,1)), axes=(0,1)), axes=(0,1)) * array.shape[0]

def split_basis_product(basis: np.array, n_jobs: int) -> np.array:
    n_chunks = int(np.ceil(basis.shape[-1] / n_jobs))
    basis_product = np.zeros(basis.shape, dtype='complex128')
    for i in tqdm(range(n_chunks)):
        split = np.array_split(basis, n_chunks, axis=-1)[i]
        basis_product[:,:,i*n_jobs:(i+1)*n_jobs] = fft3(split) * ifft3(split)
    return basis_product

def convolution(array_1: np.array, array_2: np.array) -> np.array:
    fft_array_1 = fftshift(fft2(fftshift(array_1))) / array_1.shape[0]
    fft_array_2 = fftshift(fft2(fftshift(array_2))) / array_1.shape[0]
    convo_array = fftshift(ifft2(fftshift(fft_array_1 * fft_array_2))) *np.sqrt(array_1.shape[0]*array_1.shape[1])
    return convo_array

def impulse_response(mask: np.array, frame: np.array) -> np.array:
    return 2 * np.imag(np.conj(fftshift(fft2(fftshift(mask)))/mask.shape[0]) * fftshift(fft2(fftshift(mask * frame)))/mask.shape[0])

def sensitivity(IR_1: np.array, IR_2: np.array, basis_product: np.array) -> np.array:
    fft_IR_1 = fftshift(fft2(fftshift(IR_1))).flatten() / IR_1.shape[0]
    ifft_IR_2 = fftshift(ifft2(fftshift(IR_2))).flatten() * IR_2.shape[0]
    S = (fft_IR_1 * ifft_IR_2) @ basis_product.reshape(basis_product.shape[0]**2,basis_product.shape[-1])
    return S
    
def optical_gains(IR_sky: np.array, IR_calib: np.array, basis_product: np.array, sensi_calib: np.array) -> np.array:
    sensi_sky = sensitivity(IR_sky, IR_calib, basis_product)
    og = np.real(sensi_sky / sensi_calib)
    return og
