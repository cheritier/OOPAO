# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:18 2021

@author: cheritie
"""

import numpy as np
import sys
try:
    import cupy as xp
    global_gpu_flag = True
    xp = np # for now
except ImportError or ModuleNotFoundError:
    xp = np


class SpatialFilter():
    def __init__(self,
                 telescope,
                 shape,
                 diameter,
                 zeroPaddingFactor=2):
        OOPAO_path = [s for s in sys.path if "OOPAO" in s]
        l = []
        for i in OOPAO_path:
            l.append(len(i))
        path = OOPAO_path[np.argmin(l)]
        precision = np.load(path+'/precision_oopao.npy')
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        self.tag = 'spatialFilter'
        self.telescope_resolution = telescope.resolution
        self.diameter = diameter
        self.shape = shape
        # set up an initial filter
        self.set_spatial_filter(zeroPaddingFactor=zeroPaddingFactor)

    def set_spatial_filter(self, zeroPaddingFactor):
        self.diameter_padded = self.diameter*zeroPaddingFactor
        self.resolution = int(self.telescope_resolution*zeroPaddingFactor)
        self.center = self.resolution//2
        valid_input = False
        if self.shape == 'circular':
            D = self.resolution
            x = np.linspace(-D//2, (D-1)//2, D, endpoint=False)
            xx, yy = np.meshgrid(x, x)
            R = xx**2+yy**2
            SF = R <= (self.diameter_padded)**2
            self.mask = (SF + 1j*SF) / np.sqrt(2)
            valid_input = True
        if self.shape == 'square':
            SF = np.zeros([self.resolution, self.resolution], dtype=float)
            SF[self.center-self.diameter_padded//2:self.center+self.diameter_padded//2, self.center-self.diameter_padded//2:self.center+self.diameter_padded//2] = 1
            self.mask = (SF + 1j*SF) / np.sqrt(2)
            valid_input = True
        if self.shape == 'foucault':
            SF = np.zeros([self.resolution, self.resolution], dtype=float)
            SF[:self.center] = 1
            self.mask = (SF + 1j*SF) / np.sqrt(2)
            valid_input = True
        if valid_input is False:
            raise ValueError("The input shape: '"+str(self.shape)+"' is not valid. The valid inputs are: 'circular', 'square', 'foucault'.")
        self.mask[:self.resolution-1, :self.resolution-1] = self.mask[1:, 1:]
        return

    def relay(self, src):
        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src
        for src in src_list:
            em_field_in = xp.zeros([self.resolution, self.resolution], dtype=self.precision_complex())
            n_extra_pix = (self.resolution - src.OPD.shape[0])//2
            em_field_in[n_extra_pix:-n_extra_pix, n_extra_pix:-n_extra_pix] = np.sqrt(src.fluxMap)*xp.exp(1j*(src.OPD_no_pupil*2*xp.pi/src.wavelength))
            em_field_focal_plane_filtered = xp.fft.fft2(em_field_in)*xp.fft.fftshift(self.mask)
            em_field_out = xp.fft.ifft2(em_field_focal_plane_filtered)
            src.em_field_filtered = em_field_out[n_extra_pix:-n_extra_pix, n_extra_pix:-n_extra_pix]
            src.phase_filtered = ((xp.angle(src.em_field_filtered)))*src.mask
            src.amplitude_filtered = xp.abs(src.em_field_filtered)
            src.spatialFilter = self
        return
