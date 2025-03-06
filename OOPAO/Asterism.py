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
        self.prop['parameters'] = f"{'Source':^8s}|{'Wavelength':^12s}|{'Zenith':^8s}|{'Azimuth':^9s}|{'Altitude':^10s}|{'Magnitude':^11s}|{'Flux':^11s}|"
        self.prop['units'] = f"{'':^8s}|{'[m]':^12s}|{'[arcsec]':^8s}|{'[°]':^9s}|{'[m]':^10s}|{'':^11s}|{'[ph/m²/s]':^11s}|"
        for i in range(self.n_source):
            if i%2==0:
                self.prop['layer_%02d'%i] = f"\033[00m{'%3d-%s'%(i+1,self.src[i].type):^8s}|{self.src[i].wavelength:^12.1e}|{self.src[i].coordinates[0]:^8.2f}|{self.src[i].coordinates[1]:^9.2f}|{self.src[i].altitude:^10.2f}|{self.src[i]._magnitude:^11.2f}|{self.src[i]._nPhoton:^11.1e}|"
            else:
                self.prop['layer_%02d'%i] = f"\033[47m{'%3d-%s'%(i+1,self.src[i].type):^8s}|{self.src[i].wavelength:^12.1e}|{self.src[i].coordinates[0]:^8.2f}|{self.src[i].coordinates[1]:^9.2f}|{self.src[i].altitude:^10.2f}|{self.src[i]._magnitude:^11.2f}|{self.src[i]._nPhoton:^11.1e}|"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len)) - len('\033[00m')
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Asterism ":-^{n_char}}\n'
        end_line = f'\033[00m{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table