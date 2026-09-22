# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:56:25 2023

@author: cheritier
"""
from .runtime import backend_of, to_backend


class OPD_map:
    def __init__(self, OPD):
        """
        ************************** REQUIRED PARAMETERS **************************
        An OPD_map object consists in defining the 2D map that acts as a static OPD offset. It requires the following parameters
        ************************** MAIN PROPERTIES **************************
        The main properties of an OPD_map object are listed here:
        _ static_phase_screen.OPD       : the optical path difference
        ************************** EXEMPLE **************************
        1) Create a blank OPD_map object corresponding to a Telescope object
        opd = OPD_map(OPD)
        2) Update the OPD of the OPD_map object using a given OPD_map
        opd.OPD = OPD_map
        3) propagate through the telescope
        src*tel*opd
        """
        self.OPD = OPD
        self.tag = 'OPD_map'
        self._opd_cache = None

    def _opd_on(self, backend):
        """self.OPD on `backend` (uploaded once, and again only when self.OPD is replaced)."""
        cache = self._opd_cache
        if cache is None or cache[0] is not self.OPD or cache[1] is not backend:
            cache = (self.OPD, backend, to_backend(self.OPD, backend))
            self._opd_cache = cache
        return cache[2]

    def relay(self, src):
        self.src = src
        if src.tag == 'source':
            self.src_list = [src]
        elif src.tag == 'asterism':
            self.src_list = src.src
        for src in self.src_list:
            src.optical_path.append([self.tag, self])
            # added on the backend of the source (NumPy, or CuPy with GPU residency); for a cube of OPDs
            # (e.g. interaction matrix) the static map is added to every frame
            opd = self._opd_on(backend_of(src.OPD_no_pupil))
            if src.OPD_no_pupil.ndim == 3 and opd.ndim == 2:
                opd = opd[:, :, None]
            src.OPD_no_pupil = src.OPD_no_pupil + opd