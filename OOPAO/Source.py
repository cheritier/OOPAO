# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:32:15 2020

@author: cheritie
"""
import numpy as np
from .tools.tools import OopaoError
import sys


class Source:
    def __init__(self,
                 optBand: str,
                 magnitude: float,
                 coordinates: list = [0, 0],
                 altitude: float = np.inf,
                 laser_coordinates: list = [0, 0],
                 Na_profile: float = None,
                 FWHM_spot_up: float = None,
                 display_properties: bool = True,
                 chromatic_shift: list = None):
        """SOURCE
        A source object can be defined as a point source at infinite distance (NGS) or as a extended object

        Parameters
        ----------
        optBand : str
            The optical band of the source (see the method photometry)
            ex, 'V' corresponds to a wavelength of 500 nm
.
        magnitude : float
            The magnitude of the star.
        coordinates : list, optional
            DESCRIPTION. The default is [0,0].
        altitude : float, optional
            DESCRIPTION. The default is np.inf.
        laser_coordinates : list, optional
            DESCRIPTION. The default is [0,0].
        Na_profile : float, optional
            DESCRIPTION. The default is None.
        FWHM_spot_up : float, optional
            DESCRIPTION. The default is None.
        display_properties : bool, optional
            DESCRIPTION. The default is True.
        chromatic_shift : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        """
        ************************** REQUIRED PARAMETERS **************************

        A Source object is characterised by two parameter:
        _ optBand               : the optical band of the source (see the method photometry)
        _ magnitude             : The magnitude of the star

        ************************** COUPLING A SOURCE OBJECT **************************

        Once generated, a Source object "src" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src


        ************************** MAIN PROPERTIES **************************

        The main properties of a Source object are listed here:
        _ src.phase     : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ src.type      : Ngs or LGS

        _ src.nPhoton   : number of photons per m2 per s. if this property is changed after the initialization, the magnitude is automatically updated to the right value.
        _ src.fluxMap   : 2D map of the number of photons per pixel per frame (depends on the loop frequency defined by tel.samplingTime)
        _ src.display_properties : display the properties of the src object
        _ src.chromatic_shift : list of shift in arcesc to be applied to the pupil footprint at each layer of the atmosphere object.

        The main properties of the object can be displayed using :
            src.print_properties()

        ************************** OPTIONAL PROPERTIES **************************
        _ altitude              : altitude of the source. Default is inf (NGS)
        _ laser_coordinates     : The coordinates in [m] of the laser launch telescope
        _ Na_profile            : An array of 2 dimensions and n sampling points for the Sodium profile. The first dimension corresponds to the altitude and the second dimention to the sodium profile value.
        _ FWHM_spot_up          : FWHM of the LGS spot in [arcsec]
        ************************** EXEMPLE **************************

        Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(opticalBand = 'H', magnitude = 8)
        src*tel


        """
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

        if self.precision is np.float32:
            self.precision_complex = np.complex64
        else:
            self.precision_complex = np.complex128
        self.is_initialized = False
        self.display_properties = display_properties
        # get the photometry properties
        self.__updating_flux = False
        tmp = self.photometry(optBand)
        self.optBand = optBand                               # optical band
        # wavelength in m
        self.wavelength = tmp[0]
        # optical bandwidth
        self.bandwidth = tmp[1]
        self.zeroPoint = tmp[2]/368                            # zero point
        self._magnitude = magnitude                            # magnitude
        # self.phase = []                                    # phase of the source
        # phase of the source (no pupil)
        # self.phase_no_pupil = []
        self.fluxMap = []                                    # 2D flux map of the source
        # number of photon per m2 per s
        self._nPhoton = self.zeroPoint*10**(-0.4*magnitude)
        self.tag = 'source'                              # tag of the object
        # altitude of the source object in m
        self.altitude = altitude
        # polar coordinates [r,theta]
        self.coordinates = coordinates
        # Laser Launch Telescope coordinates in [m]
        self.laser_coordinates = laser_coordinates
        # shift in arcsec to be applied to the atmospheric phase screens (one value for each layer) to simulate a chromatic effect
        self.chromatic_shift = chromatic_shift
        if Na_profile is not None and FWHM_spot_up is not None:
            self.Na_profile = Na_profile
            self.FWHM_spot_up = FWHM_spot_up

            # consider the altitude weigthed by Na profile
            self.altitude = np.sum(Na_profile[0, :]*Na_profile[1, :])
            self.type = 'LGS'
        else:

            self.type = 'NGS'

        if self.display_properties:
            print(self)

        self.is_initialized = True

        # <JM @ SpaceODT>

        self._OPD = None    # set the initial OPD
        self._OPD_no_pupil = None  # set the initial OPD

        self.mask = 1

        self.optical_path = [[self.type + '('+self.optBand+')', self]]

        self.inAsterism = False
        self.ast_idx = -1

        # <\JM @ SpaceODT>

    def __pow__(self, obj):
        if obj.isPaired:
            atm = obj.atm
            obj-atm
            self.optical_path = [[self.type + '(' + self.optBand + ')', self]]
            self.resetOPD()
            self*obj
            obj+atm

        return self

    def __mul__(self, obj):

        obj.relay(self)
        return self

        # self.optical_path.append([obj.tag, obj])

        # if obj.tag == 'telescope':
        #     if obj.src == None:
        #         obj.src = self





        # if obj.tag == 'telescope':
        #     self.tel = obj
        #     if obj.src is None:
        #         obj.src = self
        #     obj.relay(self)
        #     return self
        #
        # elif obj.tag == 'deformableMirror':
        #     self.dm = obj
        #     obj.relay(self)
        #     return self
        #
        # elif obj.tag == 'shackHartmann':
        #     self.wfs = obj
        #     obj.relay(self)
        #     return self


    def resetOPD(self):
        # self.OPD = None
        # self.OPD_no_pupil = None
        # self.OPD = 0*self.OPD
        # TODO: Does not work the first time someone resets the OPD
        self.OPD_no_pupil = 0*self.OPD_no_pupil

    def print_optical_path(self):
        if self.optical_path is not None:
            tmp_path = ''
            for i in range(len(self.optical_path)):
                tmp_path += self.optical_path[i][0]
                if i < len(self.optical_path)-1:
                    tmp_path += ' ~~> '
            print(tmp_path)

    @property
    def OPD(self):
        return self._OPD
        # if self._OPD is not None:
        #     return self._OPD
        #
        # if len(self._OPD_no_pupil.shape) > 2:
        #     for i in range(self._OPD_no_pupil.shape[-1]):
        #         self._OPD_no_pupil[:, :, i] = self._OPD_no_pupil[:, :, i]*self.mask
        #     return self._OPD_no_pupil
        #
        # return self.OPD_no_pupil*self.mask


    @OPD.setter
    def OPD(self, val):
        self._OPD = val

    @property
    def OPD_no_pupil(self):
        return self._OPD_no_pupil

    @OPD_no_pupil.setter
    def OPD_no_pupil(self, val):
        self._OPD_no_pupil = val
        self._OPD = val
        if len(self._OPD.shape) > 2:
            for i in range(self._OPD.shape[-1]):
                self._OPD[:, :, i] = self._OPD[:, :, i] * self.mask
        else:
            self._OPD = val*self.mask

    @property
    def phase(self):
        return self.OPD*2*np.pi/self.wavelength

    @phase.setter
    def phase(self, val):
        self._OPD = (val * self.wavelength) / (2 * np.pi)


    @property
    def phase_no_pupil(self):
        return self.OPD_no_pupil*2*np.pi/self.wavelength


    def photometry(self, arg):
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass

        # New entries with corrected zero point flux values
        phot.U = [0.360e-6, 0.070e-6, 1.96e12]
        phot.B = [0.440e-6, 0.100e-6, 5.38e12]
        phot.V0 = [0.500e-6, 0.090e-6, 3.64e12]
        phot.V = [0.550e-6, 0.090e-6, 3.31e12]
        phot.R = [0.640e-6, 0.150e-6, 4.01e12]
        phot.R2 = [0.650e-6, 0.300e-6, 7.9e12]
        # Fixed (entries with big difference)
        phot.R3 = [0.600e-6, 0.300e-6, 8.56e12]
        # Fixed (entries with big difference)
        phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
        phot.I = [0.790e-6, 0.150e-6, 2.69e12]
        # Fixed (entries with big difference)
        phot.I1 = [0.700e-6, 0.033e-6, 0.67e12]
        # Fixed (entries with big difference)
        phot.I2 = [0.750e-6, 0.033e-6, 0.62e12]
        # Fixed (entries with big difference)
        phot.I3 = [0.800e-6, 0.033e-6, 0.58e12]
        # Fixed (entries with big difference)
        phot.I4 = [0.700e-6, 0.100e-6, 2.02e12]
        # Fixed (entries with big difference)
        phot.I5 = [0.850e-6, 0.100e-6, 1.67e12]
        # Fixed (entries with big difference)
        phot.I6 = [1.000e-6, 0.100e-6, 1.42e12]
        # Fixed (entries with big difference)
        phot.I7 = [0.850e-6, 0.300e-6, 5.00e12]
        # Fixed (entries with big difference)
        phot.I8 = [0.750e-6, 0.100e-6, 1.89e12]
        # Fixed (entries with big difference)
        phot.I9 = [0.850e-6, 0.300e-6, 5.00e12]
        # Fixed (entries with big difference)
        phot.I10 = [0.900e-6, 0.300e-6, 4.72e12]
        phot.J = [1.215e-6, 0.260e-6, 1.90e12]
        # Fixed (entries with big difference)
        phot.J2 = [1.550e-6, 0.260e-6, 1.49e12]
        phot.H = [1.654e-6, 0.290e-6, 1.05e12]
        phot.Kp = [2.1245e-6, 0.351e-6, 0.62e12]
        phot.Ks = [2.157e-6, 0.320e-6, 0.55e12]
        phot.K = [2.179e-6, 0.410e-6, 0.70e12]
        phot.K0 = [2.000e-6, 0.410e-6, 0.76e12]
        phot.K1 = [2.400e-6, 0.410e-6, 0.64e12]
        '''
        #  Old entries
        phot.U      = [ 0.360e-6 , 0.070e-6 , 2.0e12 ]
        phot.B      = [ 0.440e-6 , 0.100e-6 , 5.4e12 ]
        phot.V0     = [ 0.500e-6 , 0.090e-6 , 3.3e12 ]
        phot.V      = [ 0.550e-6 , 0.090e-6 , 3.3e12 ]
        phot.R      = [ 0.640e-6 , 0.150e-6 , 4.0e12 ]
        phot.I      = [ 0.790e-6 , 0.150e-6 , 2.7e12 ]
        phot.I1     = [ 0.700e-6 , 0.033e-6 , 2.7e12 ]
        phot.I2     = [ 0.750e-6 , 0.033e-6 , 2.7e12 ]
        phot.I3     = [ 0.800e-6 , 0.033e-6 , 2.7e12 ]
        phot.I4     = [ 0.700e-6 , 0.100e-6 , 2.7e12 ]
        phot.I5     = [ 0.850e-6 , 0.100e-6 , 2.7e12 ]
        phot.I6     = [ 1.000e-6 , 0.100e-6 , 2.7e12 ]
        phot.I7     = [ 0.850e-6 , 0.300e-6 , 2.7e12 ]
        phot.R2     = [ 0.650e-6 , 0.300e-6 , 7.92e12 ]
        phot.R3     = [ 0.600e-6 , 0.300e-6 , 7.92e12 ]
        phot.R4     = [ 0.670e-6 , 0.300e-6 , 7.92e12 ]
        phot.I8     = [ 0.750e-6 , 0.100e-6 , 2.7e12 ]
        phot.I9     = [ 0.850e-6 , 0.300e-6 , 7.36e12 ]
        phot.I10    = [ 0.900e-6 , 0.300e-6 , 2.7e12 ]
        phot.J      = [ 1.215e-6 , 0.260e-6 , 1.9e12 ]
        phot.J2     = [ 1.550e-6 , 0.260e-6 , 1.9e12 ]
        phot.H      = [ 1.654e-6 , 0.290e-6 , 1.1e12 ]
        phot.Kp     = [ 2.1245e-6 , 0.351e-6 , 6e11 ]
        phot.Ks     = [ 2.157e-6 , 0.320e-6 , 5.5e11 ]
        phot.K      = [ 2.179e-6 , 0.410e-6 , 7.0e11 ]
        phot.K0     = [ 2.000e-6  , 0.410e-6 , 7.0e11 ]
        phot.K1     = [ 2.400e-6 , 0.410e-6 , 7.0e11 ]
        phot.IR1310 = [ 1.310e-6 , 0        , 2e12 ]   # bandwidth is zero?
        '''
        phot.L = [3.547e-6, 0.570e-6, 2.5e11]
        phot.M = [4.769e-6, 0.450e-6, 8.4e10]
        phot.Na = [0.589e-6, 0, 3.3e12]  # bandwidth is zero?
        phot.EOS = [1.064e-6, 0, 3.3e12]  # bandwidth is zero?
        phot.IR1310 = [1.310e-6, 0, 2e12]   # bandwidth is zero?

        if isinstance(arg, str):
            if hasattr(phot, arg):
                return getattr(phot, arg)
            else:
                raise OopaoError('Wrong name for the photometry object')
        else:
            raise OopaoError('The photometry object takes a scalar as an input')

    @property
    def nPhoton(self):
        return self._nPhoton

    @nPhoton.setter
    def nPhoton(self, val):
        if self.__updating_flux:
            return
        self.__updating_flux = True
        self._nPhoton = val
        self._magnitude = -2.5*np.log10(val/self.zeroPoint)
        print('Flux updated, magnitude is %2i and flux is %.2e'%(self._magnitude, self._nPhoton))
        self.__updating_flux = False

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, val):
        if self.__updating_flux:
            return
        self.__updating_flux = True
        self._magnitude = val
        self._nPhoton = self.zeroPoint*10**(-0.4*self._magnitude)
        print('Flux updated, magnitude is %2i and flux is %.2e'%(self._magnitude, self._nPhoton))
        self.__updating_flux = False

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['type'] = f"{'Source':<25s}|{self.type:^9s}"
        self.prop['wavelength'] = f"{'Wavelength [m]':<25s}|{self.wavelength:^9.1e}"
        self.prop['zenith'] = f"{'Zenith [arcsec]':<25s}|{self.coordinates[0]:^9.2f}"
        self.prop['azimuth'] = f"{'Azimuth [°]':<25s}|{self.coordinates[1]:^9.2f}"
        self.prop['altitude'] = f"{'Altitude [m]':<25s}|{self.altitude:^9.2f}"
        self.prop['magnitude'] = f"{'Magnitude':<25s}|{self._magnitude:^9.2f}"
        self.prop['flux'] = f"{'Flux [photon/m²/s]':<25s}|{self._nPhoton:^9.1e}"
        self.prop['coordinates'] = f"{'Coordinates [arcsec,deg]':<25s}|{' ['+str(self.coordinates[0])+','+str(self.coordinates[1])+']':s}"
        # self.prop['binning'] = f"{'Binning':<25s}|{str(self.binning)+'x'+str(self.binning):^9s}"

        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Source ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table