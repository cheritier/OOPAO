# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:04:31 2024

@author: astriffl
"""

import numpy as np
from joblib import Parallel, delayed

class OpticalGains:
    
    def __init__(self,
                 tel,
                 dm,
                 M2C,
                 GSC):
        
        self.resolution  = GSC.resolution
        self.pupil_resolution = self.resolution // 2 - 2
        self.D           = tel.D
        self.obstruction = tel.centralObstruction
        self.nAct        = int(dm.nActAlongDiameter+1)
        self.coupling    = dm.mechCoupling 
        self.M2C         = M2C
        self.n_modes     = M2C.shape[1]
        self.calibration_ready = False
        self.tag = 'OG'
        
        
    def optical_gains_calibration(self,frame):    
        self.calib_frame = frame.copy()
        frame = frame/frame.sum()
        self.compute_basis()
        self.compute_mask()
        
        self.IR_calib, self.fft_IR_calib = self.compute_IR(frame)
        self.ifft_IR_calib = ifft_centre(self.IR_calib)
        
        prod_basis   = self.fft_basis * self.ifft_basis
        self.num = np.tile(self.ifft_IR_calib[:,:,None], self.n_modes) * prod_basis
        self.den = np.sum(np.tile(self.fft_IR_calib[:,:,None], self.n_modes) * np.tile(self.ifft_IR_calib[:,:,None], self.n_modes) * prod_basis, axis=(0,1))
        self.calibration_ready = True    
        print('Optical gain calibration done')

    
    def reset_calibration(self):
        self.calibration_ready = False
        self.OG = np.ones(self.n_modes)

    
    def optical_gains_computation(self, frame):
        if self.calibration_ready is False:
            raise AttributeError('Optical gains must be initialized first!')
        else:
            frame = frame/frame.sum()
            _, self.fft_IR_sky = self.compute_IR(frame)
            self.OG = np.real(np.sum(np.tile(self.fft_IR_sky[:,:,None], self.n_modes)  * self.num, axis=(0,1)) / self.den)
    
    
    def compute_basis(self):
        Dpup    = self.pupil_resolution+1
        x       = np.linspace(-self.pupil_resolution/2,self.pupil_resolution/2,self.pupil_resolution)
        xx,yy   = np.meshgrid(x,x)
        circle  = xx**2+yy**2
        obs     = circle>=(self.obstruction*Dpup/2)**2
        pupil   = circle<(Dpup/2)**2 
        pupil   = pupil*obs
        pupil   = pupil.reshape(self.pupil_resolution**2)
        
        IF = self.influence_function_grid()
        modes = IF @ self.M2C
        modes = modes * np.tile(pupil[:,None], (1,self.n_modes))
        modes = modes.reshape(self.pupil_resolution,self.pupil_resolution,self.n_modes)
        pad = (self.resolution - self.pupil_resolution) // 2
        self.basis = np.pad(modes, ((pad,pad),(pad,pad),(0,0)))
        self.fft_basis = np.asarray(fftn(self.basis)).T
        self.ifft_basis   = np.asarray(ifftn(self.basis)).T
        
    def influence_function_grid(self):
        pitch = self.D / (self.nAct-1)#)+1)
        x = np.linspace(-self.D/2,self.D/2,self.nAct)
        X,Y = np.meshgrid(x,x)            
        xIF0 = np.reshape(X,[self.nAct**2])
        yIF0 = np.reshape(Y,[self.nAct**2])
        r = np.sqrt(xIF0**2 + yIF0**2)
        validActInner = r > (self.obstruction*self.D/2 - 0.5*pitch)
        validActOuter = r <= (self.D/2 + 0.7533*pitch)
        validAct = validActInner*validActOuter
        xIF = xIF0[validAct]
        yIF = yIF0[validAct]
        u0x      = self.pupil_resolution/2+xIF*self.pupil_resolution/self.D
        u0y      = self.pupil_resolution/2+yIF*self.pupil_resolution/self.D      
        nIF = len(xIF)
        IF_grid = np.zeros((self.pupil_resolution**2,nIF))
        for i in range(nIF):
            IF_grid[:,i] = self.influence_function(self.pupil_resolution,u0x[i],u0y[i],self.coupling,self.nAct)
        return IF_grid
    
    
    def influence_function(self, res, i, j, coupling, nAct):     
        cx = (res/nAct)/np.sqrt(2*np.log(1/coupling))
        cy = (res/nAct)/np.sqrt(2*np.log(1/coupling))
        x = np.linspace(0, 1, res)*res
        X, Y = np.meshgrid(x, x)
        G = np.exp(-(1/(2*cx**2)*(X-i)**2 + 1/(2*cy**2)*(Y-j)**2))
        return G.reshape(res**2)
    
    
    def compute_mask(self):
        norma = self.resolution / 2
        lim = np.pi / 4 * (1 - 1 / (self.resolution // 2))
        mask = np.zeros((self.resolution, self.resolution))
        Tip, Tilt = np.meshgrid(np.linspace(-lim, lim, self.resolution // 2), np.linspace(-lim, lim, self.resolution // 2))
        mask[0:self.resolution // 2, 0:self.resolution // 2] = Tip * norma + Tilt * norma
        mask[0:self.resolution // 2, self.resolution // 2:self.resolution] = -Tip * norma + Tilt * norma
        mask[self.resolution // 2:self.resolution, self.resolution // 2:self.resolution] = -Tip * norma - Tilt * norma
        mask[self.resolution // 2:self.resolution, 0:self.resolution // 2] = Tip * norma - Tilt * norma
        mask = -mask
        self.mask = np.exp(1j*mask)
        self.fft_mask = fft_centre(self.mask)
    
    
    def compute_IR(self, frame):
        IR = 2 * np.imag(np.conj(self.fft_mask) * fft_centre(self.mask * frame))
        fft_IR = fft_centre(IR)
        return IR, fft_IR
        
        
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