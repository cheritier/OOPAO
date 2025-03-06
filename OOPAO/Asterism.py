# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:35:32 2022

@author: cheritie
"""

import numpy as np


class Asterism:
    def __init__(self,
                 list_src: list):
        """
        ************************** REQUIRED PARAMETERS **************************

        An Asterism object is an asterism of Source objects. It requires the following parameters:

        _ list_src              : a list of Source objects that can combine NGS and LGS types

        ************************** COUPLING AN ASTERISM OBJECT **************************

        Once generated, an asterism  object "ast" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : ast*tel
        _ It can be accessed using                  : tel.src


        ************************** MAIN PROPERTIES **************************

        The main properties of a Telescope object are listed here:
        _ ast.coordinates   : coordinates of the source objects
        _ ast.altitude      : altitude of the source objects
        _ ast.nPhoton       : nPhoton property of the source objects

        ************************** EXEMPLE **************************

        Create a list of source object in H band with a magnitude 8 and combine it to the telescope

        src_1 = Source(opticalBand = 'H', magnitude = 8, coordinates=[0,0])
        src_2 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,0])
        src_3 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,120])
        src_4 = Source(opticalBand = 'H', magnitude = 8, coordinates=[60,240])

        ast = Asterism([src_1,src_2,src_3, src_4])


        """
        self.n_source = len(list_src)
        self.src = list_src
        self.coordinates = []
        self.altitude = []
        self.nPhoton = 0
        self.chromatic_shift = None
        print(self)
        
    def __mul__(self, telescope):
        if type(telescope.OPD) is not list:
            tmp_OPD = telescope.OPD.copy()
            telescope.OPD = [tmp_OPD for i in range(self.n_source)]
            tmp_OPD = telescope.OPD_no_pupil.copy()
            telescope.OPD_no_pupil = [tmp_OPD for i in range(self.n_source)]
        for i in range(self.n_source):
            # update the phase of the source
            self.src[i].phase = telescope.OPD[i]*2*np.pi/self.src[i].wavelength
            self.src[i].phase_no_pupil = telescope.OPD_no_pupil[i] * \
                2*np.pi/self.src[i].wavelength
        # assign the source object to the telescope object
        telescope.src = self
        return telescope
    
    def properties(self) -> dict:
        self.prop = dict()
        self.prop['type']       = f"{'Source':<20s}|"
        self.prop['wavelength'] = f"{'Wavelength [m]':<20s}|"
        self.prop['zenith']     = f"{'Zenith [arcsec]':<20s}|"
        self.prop['azimuth']    = f"{'Azimuth [°]':<20s}|"
        self.prop['altitude']   = f"{'Altitude [m]':<20s}|"
        self.prop['magnitude']  = f"{'Magnitude':<20s}|"
        self.prop['flux']       = f"{'Flux [photon/m²/s]':<20s}|"
        for i in range(min(self.n_source,6)):
            self.prop['type']       += f"{'%d - %s'%(i+1,self.src[i].type):^9s}|"
            self.prop['wavelength'] += f"{self.src[i].wavelength:^9.1e}|"
            self.prop['zenith']     += f"{self.src[i].coordinates[0]:^9.2f}|"
            self.prop['azimuth']    += f"{self.src[i].coordinates[1]:^9.2f}|"
            self.prop['altitude']   += f"{self.src[i].altitude:^9.2f}|"
            self.prop['magnitude']  += f"{self.src[i]._magnitude:^9.2f}|"
            self.prop['flux']       += f"{self.src[i]._nPhoton:^9.1e}|"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        if self.n_source > 6:
            title = f'\n{" Asterism - printing 6 first sources out of %d "%self.n_source:-^{n_char}}\n'
        else:
            title = f'\n{" Asterism ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table