# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:35:32 2022

@author: cheritie
"""

import numpy as np
import matplotlib.pyplot as plt
from OOPAO.tools.displayTools import makeSquareAxes
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

        for i, src in enumerate(self.src):
            src.inAsterism = True
            src.ast_idx = i


        self.chromatic_shift = None
        print(self)
        self.tag = 'asterism'
        self.type = 'asterism'

        # for i in range(self.n_source):
        #     self.coordinates.append(self.src[i].coordinates)
        #     self.altitude.append(self.src[i].altitude)
        #     self.nPhoton += self.src[i].nPhoton/self.n_source
            # self.phase.append(self.src[i].phase)
            # self.fluxMap.append(self.src[i].phase)

        # self.phase = np.asarray(self.phase)
        # self.fluxMap = np.asarray(self.fluxMap)

        self.wavelength = self.src[0].wavelength

    @property
    def fluxMap(self):
        _fluxMap = []
        for src in self.src:
            _fluxMap.append(src.fluxMap)
        return _fluxMap

    @property
    def phase(self):
        _phase = []
        for src in self.src:
            _phase.append(src.phase)
        return _phase

    @phase.setter
    def phase(self, val):
        for src in self.src:
            src.phase = val[src.ast_idx]
            # src.OPD = (val[src.ast_idx]*self.wavelength) / (2*np.pi)


    @property
    def phase_no_pupil(self):
        _phase_no_pupil = []
        for src in self.src:
            _phase_no_pupil.append(src.phase_no_pupil)
        return _phase_no_pupil

    @property
    def coordinates(self):
        _coordinates = []
        for src in self.src:
            _coordinates.append(src.coordinates)
        return _coordinates

    @property
    def altitude(self):
        _altitude = []
        for src in self.src:
            _altitude.append(src.altitude)
        return _altitude

    @property
    def nPhoton(self):
        _nPhoton = []
        for src in self.src:
            _nPhoton.append(src.nPhoton)
        return _nPhoton

    @property
    def OPD(self):
        _OPD = []
        for src in self.src:
            _OPD.append(src.OPD)
        return np.array(_OPD)

    @property
    def OPD_no_pupil(self):
        _OPD_no_pupil = []
        for src in self.src:
            _OPD_no_pupil.append(src.OPD_no_pupil)
        return np.array(_OPD_no_pupil)





    def __pow__(self, obj):
        obj.src = self
        # obj.resetOPD()

        for src in self.src:
            src.optical_path = [[src.type + '('+src.optBand+')', src]]

        self.resetOPD()

        self*obj

        # if obj.isPaired:
        #     atm = obj.atm
        #     obj-atm
        #     self*obj
        #     atm.asterism=self
        #     obj+atm

        # else:
        #     self * obj

            # for src in self.src:
            #     src.optical_path = [[src.type + '('+src.optBand+')', src]]
            #     src.resetOPD()
            #     src*obj

        return self



    def __mul__(self, obj):

        obj.relay(self)
        return self



    def resetOPD(self):
        for src in self.src:
            src.resetOPD()


    def print_optical_path(self):
        for src in self.src:
            src.print_optical_path()


    # for backward compatibility
    def print_properties(self):
        print(self)

    def display_asterism(self):
        
        plt.figure()
        x = np.linspace(0,2*np.pi,100,endpoint=True)
        r = np.linspace(0,max(self.coordinates)[0],6,endpoint=True)
        for i_circle in range(1,6):
            plt.plot(r[i_circle]*np.cos(x),
                     r[i_circle]*np.sin(x),'--k')    
        cm = plt.get_cmap('gist_rainbow')
        for i_src in range(self.n_source):
            if self.src[i_src].type == "NGS":
                marker = '*'
                size_marker=20
            else:
                marker = '*'
                size_marker=10
                
            plt.plot(self.src[i_src].coordinates[0]*np.cos(np.deg2rad(self.src[i_src].coordinates[1])),
                     self.src[i_src].coordinates[0]*np.sin(np.deg2rad(self.src[i_src].coordinates[1])),
                     marker, markersize=size_marker, color = cm(1.*i_src/self.n_source),markeredgecolor = 'k',
                     label = self.src[i_src].type+'_'+str(i_src))
        makeSquareAxes()
        plt.xlabel('[Arcsec]')
        plt.ylabel('[Arcsec]')
        plt.legend()
        
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
