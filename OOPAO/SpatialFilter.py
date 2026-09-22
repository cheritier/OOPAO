# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:18 2021

@author: cheritie
"""

import numpy as np
from .runtime import precision_bits, backend_of, to_backend


class SpatialFilter():
    def __init__(self,
                 telescope,
                 shape,
                 diameter,
                 zeroPaddingFactor=2):
        precision = precision_bits()
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is np.float32:
            self.precision_complex = np.complex64
        else:
            self.precision_complex = np.complex128
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
        # fftshift-ed mask on the backend in use, rebuilt when the filter changes
        self._mask_cache = None
        return

    def _shifted_mask(self, backend):
        """fftshift(self.mask) on `backend`, at the working precision."""
        cache = self._mask_cache
        if cache is None or cache[0] is not self.mask or cache[1] is not backend:
            shifted = backend.asarray(np.fft.fftshift(self.mask).astype(self.precision_complex))
            cache = (self.mask, backend, shifted)
            self._mask_cache = cache
        return cache[2]

    def relay(self, src):
        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src
        for src in src_list:
            # computed on the backend of the source (NumPy, or CuPy with GPU residency)
            backend = backend_of(src.OPD_no_pupil)
            n_pix = src.OPD.shape[0]
            n_extra_pix = (self.resolution - n_pix)//2
            pupil = slice(n_extra_pix, n_extra_pix + n_pix)
            em_field_in = backend.zeros([self.resolution, self.resolution], dtype=self.precision_complex)
            # src.intensity = fluxMap * scintillation (the amplitude used by the wave-front sensors)
            amplitude = backend.sqrt(to_backend(src.intensity, backend))
            em_field_in[pupil, pupil] = amplitude*backend.exp(1j*(to_backend(src.OPD_no_pupil, backend)*2*np.pi/src.wavelength))
            em_field_focal_plane_filtered = backend.fft.fft2(em_field_in)*self._shifted_mask(backend)
            em_field_out = backend.fft.ifft2(em_field_focal_plane_filtered)
            src.em_field_filtered = em_field_out[pupil, pupil]
            # filtered field: its amplitude carries the flux (amplitude_filtered**2 is the filtered intensity)
            src.phase_filtered = ((backend.angle(src.em_field_filtered)))*to_backend(src.mask, backend)
            src.amplitude_filtered = backend.abs(src.em_field_filtered)
            src.spatialFilter = self
        return