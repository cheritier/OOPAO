# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:25:47 2024

@author: astriffl
"""

import numpy as np
from joblib import Parallel, delayed
import time
from .DetectorV2 import Detector

class GainSensingCamera():
    
    __wfs      = None
    __tel      = None
    __num      = None
    __den      = None
    __mask     = None
    __fft_mask = None
    # ex: print with _GainSensingCamera__tel

    def __init__(self, 
                 tel,
                 dm,
                 wfs,
                 M2C,
                 nRes:int=None,
                 ncpa=None,
                 maximum:float=0.8,
                 integrationTime:float=None,
                 bits:int=None,
                 FWC:int=None,
                 gain:int=1,
                 sensor:str='CCD',
                 QE:float=1,
                 binning:int=1,
                 darkCurrent:float=0,
                 readoutNoise:float=0,
                 photonNoise:bool=False,
                 backgroundNoise:bool=False,
                 backgroundNoiseMap:float = None):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        _ tel : OOPAO telescope object
        _ dm  : OOPAO deformable miror object
        _ wfs : OOPAO wavefront sensor object
        _ M2C : Matrix modes-to-commands [KL, Zernike, ...] for optical gains estimation
        
        ************************** OPTIONAL PARAMETERS **************************
        _ ncpa             : To 
        _ FoV              : Field of view in lambda/D
        _ bits             : Quantization (bits)
        _ binning          : Binning 
        _ QE               : Quantum efficiency of the GSC float in [0-1]
        _ photon_noise     : True / False
        _ background_noise : True / False
        _ readout_noise    : in e-
            
        """
        
        self.tag  = 'GSC_dev'
        self.__tel  = tel
        self.__wfs  = wfs
        self.binning = binning
        self.ncpa = ncpa
        self.n_modes = M2C.shape[1]
        self.is_og_calibrated = False

        self.cam = Detector(nRes               = wfs.nRes,
                            maximum            = maximum,
                            integrationTime    = integrationTime,
                            bits               = bits,
                            FWC                = FWC,
                            gain               = gain,
                            sensor             = sensor,
                            QE                 = QE,
                            binning            = binning,
                            darkCurrent        = darkCurrent,
                            readoutNoise       = readoutNoise,
                            photonNoise        = photonNoise,
                            backgroundNoise    = backgroundNoise,
                            backgroundNoiseMap = backgroundNoiseMap)
        
        dm.coefs = M2C
        self.__tel*dm
        self.basis = self.__tel.OPD.copy()
        m = self.basis.shape[0]
        n = self.__wfs.nRes
        basis_pad = np.zeros((n, n, self.basis.shape[2]))
        basis_pad[n//2-m//2:n//2+m//2,n//2-m//2:n//2+m//2] = self.basis
        self.basis = self.cam.set_binning(basis_pad, self.binning, mode='mean')
        
        self.optical_gains()
        
    def optical_gains(self):
        if self.is_og_calibrated is False:
            t0 = time.time()
            if self.ncpa is None:
                self.__tel.resetOPD()
                self.__tel*self.__wfs
                self.__wfs*self
                self.calib_frame = np.flip(self.cam.frame.copy())
            else:
                self.__tel.resetOPD()
                self.__tel*self.ncpa*self.__wfs
                self.__wfs*self
                self.calib_frame = np.flip(self.cam.frame.copy())
            
            self.__mask = self.cam.set_binning(self.__wfs.mask,self.binning,mode='mean')
            self.__fft_mask = fft_centre(self.__mask)
            self.IR_calib  = 2 * np.imag(np.conj(self.__fft_mask) * fft_centre(self.__mask * self.calib_frame/np.sum(self.calib_frame) * np.sqrt(self.calib_frame.shape[0]*self.calib_frame.shape[1])))
            fft_IR_calib   = np.tile(fft_centre(self.IR_calib)[:,:,None], self.n_modes)
            ifft_IR_calib  = np.tile(ifft_centre(self.IR_calib)[:,:,None], self.n_modes)
            fft_basis = np.asarray(fftn(self.basis)).T
            ifft_basis   = np.asarray(ifftn(self.basis)).T
            prod_basis   = fft_basis * ifft_basis
            self.__num = ifft_IR_calib * prod_basis
            self.__den = np.sum(fft_IR_calib * ifft_IR_calib * prod_basis, axis=(0,1))
            self.is_og_calibrated = True
            self.t_init = time.time() - t0
            self.print_properties()
        else:
            self.IR_sky = 2 * np.imag(np.conj(self.__fft_mask) * fft_centre(self.__mask * self.cam.frame/np.sum(self.cam.frame) * np.sqrt(self.cam.frame.shape[0]*self.cam.frame.shape[1])))
            fft_IR_sky = np.tile(fft_centre(self.IR_sky)[:,:,None], self.n_modes)
            self.OG = np.real(np.sum(fft_IR_sky * self.__num, axis=(0,1)) / self.__den)
            return self.OG
        
    def reset_OG_calibration(self):
        self.is_og_calibrated = False
        
        
    def print_properties(self):
        print()
        print('-------------- GSC ---------------')
        self.cam.print_properties()
        if self.is_og_calibrated is True:
            print()
            print('-------------- OG ----------------')
            print('{:^20s} {:^9s}'.format('Calibration around','flat' if self.ncpa is None else 'NCPA'))
            if self.ncpa is not None:
                print('{:^20s}|{:^9.2f}'.format('Amplitude [nm RMS]',np.std(self.ncpa.OPD[np.where(self.__tel.pupil>0)])*1e9))
            print('{:^20s}|{:^9.3f}'.format('T init [sec]',self.t_init))
            print('{:^20s}|{:^9d}'.format('N modes', self.n_modes))
        print('----------------------------------')
        
    def __repr__(self):
        self.print_properties()
        return ' '
    
    
def fft_centre(X):
    # Compute a phasor in direct space + Fourier space to be centered on both.
    # Phasor computation
    nPx = X.shape[0]
    [xx,yy] = np.meshgrid(np.linspace(0,nPx-1,nPx),np.linspace(0,nPx-1,nPx))
    phasor = np.exp(-(1j*np.pi*(nPx+1)/nPx)*(xx+yy))
    # Phasor in Direct space
    Y = np.fft.fft2(X*phasor)
    # Phasor in Fourier space
    Y = phasor*Y
    # Normalisation
    # Y = np.exp(-(1j*np.pi*(nPx+1)**2/nPx))*Y
    # Normalisation DFT - Divide by nPx because fft2
    Y = Y/nPx
    return Y

def ifft_centre(X):
    # Compute a phasor in direct space + Fourier space to be centered on both.
    # Phasor computation
    nPx = X.shape[0]
    [xx,yy] = np.meshgrid(np.linspace(0,nPx-1,nPx),np.linspace(0,nPx-1,nPx))
    phasor = np.exp((1j*np.pi*(nPx+1)/nPx)*(xx+yy))
    # Phasor in Direct space
    Y = np.fft.ifft2(X*phasor)
    # Phasor in Fourier space
    Y = phasor*Y
    # Normalisation
    # Y = np.exp((1j*np.pi*(nPx+1)**2/nPx))*Y
    # Normalisation DFT - Multiply by nPx because ifft2
    Y = Y*nPx
    return Y

def convo(X,Y):
    a=fft_centre(X)
    b=fft_centre(Y)
    c=ifft_centre(a*b)*np.sqrt(X.shape[0]*X.shape[1])  #normalisé
    return c

def fftn(M):
    Q=Parallel(n_jobs=1,prefer='threads')(delayed(fft_centre)(i) for i in (M.T))
    return Q

def ifftn(M):
    Q=Parallel(n_jobs=1,prefer='threads')(delayed(ifft_centre)(i) for i in (M.T))
    return Q

def convon(X, Y):
    a = np.asarray(fftn(X))
    b = np.asarray(fftn(Y))
    res = np.asarray(ifftn((a*b).T)) * np.sqrt(X.shape[0]*X.shape[1])
    return res