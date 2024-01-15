# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:30:02 2023

@author: astriffl
"""
import numpy as np
from numpy.fft import fft2, fftshift, ifft2
from joblib import Parallel, delayed
import time
from tqdm import tqdm

class ConvolutionalModel:
    def __init__(self, tel, dm, wfs, M2C, GSC_sampling=2, NCPA=None, custom_mask=None):
        
        self.tel      = tel
        self.dm       = dm
        self.wfs      = wfs
        self.M2C      = M2C
        self.sampling = GSC_sampling
        
        self.dm.coefs = self.M2C
        self.tel*self.dm
        self.basis = self.tel.OPD.copy()
        m = self.basis.shape[0]
        n = self.wfs.nRes
        basis_pad = np.zeros((n, n, self.basis.shape[2]))
        basis_pad[n//2-m//2:n//2+m//2,n//2-m//2:n//2+m//2] = self.basis
        self.basis = basis_pad
        
        if NCPA is not None:
            self.NCPA_OPD = np.zeros((n,n))
            self.NCPA_OPD[n//2-m//2:n//2+m//2,n//2-m//2:n//2+m//2] = self.tel.pupil * NCPA
        else:
            self.NCPA_OPD = np.zeros((n,n))
        
        if custom_mask is None:
            self.m = self.wfs.mask
            self.mchap = fft_centre(self.m)
        else:
            self.m = custom_mask
            self.mchap = fft_centre(custom_mask)
            
        self.isOGconstants = False
        
    def impulse_response_dirac(self, stroke):
        dirac = np.zeros((self.tel.resolution, self.tel.resolution))
        if self.tel.resolution % 2 == 0:
            dirac[self.tel.resolution//2-1:self.tel.resolution//2+1,self.tel.resolution//2-1:self.tel.resolution//2+1] = np.ones((2,2)) * stroke
            norma = 4 * stroke
        else:
            dirac[self.tel.resolution//2,self.tel.resolution//2] = stroke
            norma = stroke
        self.tel.OPD = dirac
        self.tel*self.wfs
        self.IR_dirac = (self.wfs.pyramidFrame - self.wfs.referencePyramidFrame) / norma 
        return self.IR_dirac
        
    def compute_impulse_response(self, GSC=None):
        if GSC is None:
            self.tel.resetOPD()
            self.dm.coefs = 0
            self.dm.coefs = self.dm.coefs
            self.tel*self.dm*self.wfs
            self.wfs.get_modulation_frame(norma=False)
            self.GSC_calib = self.wfs.modulation_camera_frame
            self.IR_calib = 2 * np.imag(np.conj(self.mchap) * fft_centre(self.m * self.GSC_calib/np.sum(self.GSC_calib) * np.sqrt(self.GSC_calib.shape[0]*self.GSC_calib.shape[1])))
            
        else:
            self.IR_sky = 2 * np.imag(np.conj(self.mchap) * fft_centre(self.m * GSC/np.sum(GSC) * np.sqrt(GSC.shape[0]*GSC.shape[1])))
            
    def compute_OG_constants(self, GSC_calib=None):
        t0 = time.time()
        if GSC_calib is None:
            self.compute_impulse_response()
        else:
            self.IR_calib = 2 * np.imag(np.conj(self.mchap) * fft_centre(self.m * GSC_calib/np.sum(GSC_calib) * np.sqrt(GSC_calib.shape[0]*GSC_calib.shape[1])))
        fftIRc   = np.tile(fft_centre(self.IR_calib)[:,:,None], self.M2C.shape[1])
        ifftIRc  = np.tile(ifft_centre(self.IR_calib)[:,:,None], self.M2C.shape[1])
        fftBasis = np.asarray(fftn(self.basis)).T
        ifftBasis   = np.asarray(ifftn(self.basis)).T
        basisProd   = fftBasis * ifftBasis
        self.numerator = ifftIRc * basisProd
        self.denominator = np.sum(fftIRc * ifftIRc * basisProd, axis=(0,1))
        print('\nConstants computed in '+str(np.round(time.time()-t0,3))+' s')
        self.isOGconstants = True
        
    def optical_gains(self, GSC):
        if self.isOGconstants:
            t0 = time.time()
            self.compute_impulse_response(GSC=GSC)
            fft_IR_sky = np.tile(fft_centre(self.IR_sky)[:,:,None], self.M2C.shape[1])
            self.OG = np.real(np.sum(fft_IR_sky * self.numerator, axis=(0,1)) / self.denominator)
            self.duration = np.round(time.time()-t0,5)
            return self.OG
        else:
            raise TypeError("Optical gains constant not computed ! Use class.compute_OG_constants()")
    
    def interaction_matrix(self, IR):
        IR = np.tile(IR[:,:,None], self.M2C.shape[1])
        IM = np.real(convon(IR, self.basis)).T
        IM = IM.reshape(IM.shape[0]*IM.shape[1],self.M2C.shape[1])
        return IM
    
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

def shannon_padding(M):
    l = M.shape[1]
    h = M.shape[0]
    dtype = ['complex64', 'complex128']
    if M.dtype in dtype:
        d = complex
    else:
        d = float
    if M.ndim == 2:
        MatPadded = np.zeros((h*2, l*2), dtype=d)
        MatPadded[h-h//2:h+h//2, l-l//2:l+l//2] = M
    if M.ndim == 3:
        MatPadded = np.zeros((h*2, l*2, M.shape[2]), dtype=d)
        MatPadded[h-h//2:h+h//2, l-l//2:l+l//2, :] = M
    return MatPadded

def shannon_binning(M):
    new_shape = [int(M.shape[0]//2) ,int(M.shape[1]//2)]
    shape = (new_shape[0], M.shape[0] // new_shape[0], 
             new_shape[1], M.shape[1] // new_shape[1])
    return M.reshape(shape).mean(-1).mean(1)