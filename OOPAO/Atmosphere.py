# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:59:02 2020

@author: cheritie
"""
import sys
import json
import time
import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import matplotlib.gridspec as gridspec
from .phaseStats import ft_phase_screen, ft_sh_phase_screen, makeCovarianceMatrix
from .tools.displayTools import makeSquareAxes
from .tools.interpolateGeometricalTransformation import interpolate_cube, interpolate_image
from .tools.tools import createFolder, emptyClass, globalTransformation, pol2cart, translationImageMatrix, OopaoError, warning
try:
    import cupy as xp
    global_gpu_flag = True
    xp = np  # for now
except ImportError or ModuleNotFoundError:
    xp = np


class Atmosphere:
    def __init__(self,
                 telescope,
                 r0: float,
                 L0: float,
                 windSpeed: list,
                 fractionalR0: list,
                 windDirection: list,
                 altitude: list,
                 src=None,
                 param=None,
                 mode: float = 2):
        """ ATMOSPHERE.
        An Atmosphere is made of one or several layer of turbulence that follow the Van Karmann statistics.
        Each layer is considered to be independant to the other ones and has its own properties (direction, speed, etc.)
        The Atmosphere object can be defined for a single Source object (default) or multi Source Object.
        The Source coordinates allow to span different areas in the field (defined as well by the tel.fov).
        If the source type is an LGS the cone effect is considered using an interpolation.
        NGS and LGS can be combined together in the Asterism object.
        The convention chosen is that all the wavelength-dependant atmosphere parameters are expressed at 500 nm.

        Parameters
        ----------
        telescope : Telescope
            The telescope object to which the Atmosphere is associated.
            This object carries the phase, flux, pupil information and sampling time as well as the type of source (NGS/LGS, source/asterism).
        r0 : float
            the Fried Parameter in m, at 500 nm.
        L0 : float
            Outer scale parameter.
        windSpeed : list
            List of wind-speed for each layer in [m/s].
        fractionalR0 : list
            Cn2 profile of the turbulence. This should be a list of values for each layer.
        windDirection : list
            List of wind-direction for each layer in [deg].
        altitude : list
            List of altitude for each layer in [m].
        mode : float, optional
            Method to compute the atmospheric spectrum from which are computed the atmospheric phase screens.
            1 : using aotools dependency
            2 : using OOPAO dependancy
            The default is 2.
        param : Parameter File Object, optional
            Parameter file of the system. Once computed, the covariance matrices are saved in the calibration data folder and loaded instead of re-computed evry time.
            The default is None.
        asterism : Asterism, optional
            If the system contains multiple source, an astrism should be input to the atmosphere object.
            The default is None.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** COUPLING A TELESCOPE AND AN ATMOSPHERE OBJECT **************************
        A Telescope object "tel" can be coupled to an Atmosphere object "atm" using:
            _ tel + atm
        This means that a bridge is created between atm and tel: everytime that atm.OPD is updated, the tel.OPD property is automatically set to atm.OPD to reproduce the effect of the turbulence.

        A Telescope object "tel" can be separated of an Atmosphere object "atm" using:
            _ tel - atm
        This corresponds to a diffraction limited case (no turbulence)


        ************************** PROPERTIES **************************

        The main properties of the Atmosphere object are listed here:

        _ atm.OPD : Optical Path Difference in [m] truncated by the telescope pupil. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.OPD_no_pupil : Optical Path Difference in [m]. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.r0
        _ atm.L0
        _ atm.nLayer                                : number of turbulence layers
        _ atm.seeingArcsec                          : seeing in arcsec at 500 nm
        _ atm.layer_X                               : access the child object corresponding to the layer X where X starts at 0

        The main properties of the object can be displayed using :
            atm.print_properties()

        the following properties can be updated on the fly:
            _ atm.r0
            _ atm.windSpeed
            _ atm.windDirection
        ************************** FUNCTIONS **************************

        _ atm.update()                              : update the OPD of the atmosphere for each layer according to the time step defined by tel.samplingTime
        _ atm.update(OPD)                           : update the OPD of the atmosphere using a user defined OPD
        _ atm.generateNewPhaseScreen(seed)          : generate a new phase screen for the atmosphere OPD
        _ atm.print_atm_at_wavelength(wavelength)   : prompt seeing and r0 at specified wavelength
        _ atm.print_atm()                           : prompt the main properties of the atm object
        _ display_atm_layers(layer_index)           : imshow the OPD of each layer with the intersection beam for each source

        """
        self.tag = 'atmosphere'      # Tag of the object
        # detect the simulation precision requested
        OOPAO_path = [s for s in sys.path if "OOPAO" in s]
        l_ = []
        for i in OOPAO_path:
            l_.append(len(i))
        path = OOPAO_path[np.argmin(l_)]
        precision = np.load(path+'/precision_oopao.npy')
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        self.hasNotBeenInitialized = True
        # Wavelengt used to define the properties of the atmosphere
        self.wavelength = 500*1e-9
        self.r0_def = 0.15  # Default Fried Parameter in m at 500 nm to build covariance matrices once and scale them later
        self.r0 = r0  # User input Fried Parameter in m at 500 nm
        self.rad2arcsec = (180. / np.pi) * 3600
        self.fractionalR0 = fractionalR0      # Fractional Cn2 profile in percentage
        self.altitude = altitude  # altitude of the atmospheric layers
        self.cn2 = (self.r0**(-5. / 3) / (0.423 * (2*np.pi/self.wavelength)**2))/np.max([1, np.max(self.altitude)])  # Cn2 m^(-2/3)
        self.L0 = L0                # Outer Scale in m
        self.nLayer = len(fractionalR0)     # number of layer
        self.windSpeed = windSpeed         # wind speed of the layers in m/s
        self.windDirection = windDirection     # wind direction in degrees
        self.n_extra_pixel = 2                 # number of extra pixel to generate the phase screens
        self.telescope = telescope         # associated telescope object
        self.V0 = (np.sum(np.asarray(self.fractionalR0) * np.asarray(self.windSpeed)**(5/3)))**(3/5)  # computation of equivalent wind speed, Roddier 1982
        self.tau0 = 0.31 * self.r0 / self.V0  # Coherence time of atmosphere, Roddier 1981
        # default value to update phase screens at each iteration
        self.is_user_defined_opd = False
        self.mode = mode              # DEBUG -> first phase screen generation mode
        self.seeingArcsec = self.rad2arcsec*(self.wavelength/self.r0)
        if src is None and self.telescope.src is None:
            raise OopaoError(
                "The Atmosphere object requires a Source. "
                "Either provide a Source directly as an attribute, or propagate the Source through the Telescope before creating the Atmosphere.")
        if src:
            self.src = src
        else:
            self.src = self.telescope.src
        if self.src.tag == 'source':
            self.src_list = [self.src]
            self.asterism = None
        elif self.src.tag == 'asterism':
            self.src_list = self.src.src
            self.asterism = self.src
        self.param = param

    def initializeAtmosphere(self, telescope=None, compute_covariance=True):
        if telescope is not None:
            self.telescope = telescope
        self.compute_covariance = compute_covariance  # flag to compute the covariance matrices
        # initialization of the support on which the turbulent wave-front is computed.
        OPD_support = self.initialize_OPD_support()
        # save the field of view requested from the telescope class
        self.fov = telescope.fov
        self.fov_rad = telescope.fov_rad
        if self.hasNotBeenInitialized:
            self.initial_r0 = self.r0
            for i_layer in range(self.nLayer):
                print('Creation of layer' + str(i_layer+1) + '/' + str(self.nLayer) + ' ...')
                # atmsopheric layer computation
                tmp_layer = self.buildLayer(telescope, self.r0_def, self.L0, i_layer=i_layer, compute_covariance=self.compute_covariance)
                setattr(self, 'layer_'+str(i_layer+1), tmp_layer)
                # grab the volume of atmosphere requested and fill OPD_support accordingly
                OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
                tmp_layer.OPD_support = OPD_support
                # wavelength scaling to compute the wavefront in [m]
                tmp_layer.OPD *= self.wavelength/2/xp.pi
        else:
            print('Re-setting the atmosphere to its initial state...')
            self.r0 = self.initial_r0
            for i_layer in range(self.nLayer):
                print('Updating layer' + str(i_layer+1) + '/' + str(self.nLayer) + ' ...')
                # access each layer to modify its properties
                tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                # re-load the saved initial OPD
                tmp_layer.OPD = tmp_layer.initial_OPD/self.wavelength*2*xp.pi
                # reset the random state to its initial value
                tmp_layer.randomState = RandomState(42+i_layer*1000)
                # reset the covariance matrices value
                Z = tmp_layer.OPD[tmp_layer.innerMask[1:-1, 1:-1] != 0]
                X = xp.matmul(tmp_layer.A, Z) + xp.matmul(tmp_layer.B, tmp_layer.randomState.normal(size=tmp_layer.B.shape[1]))
                tmp_layer.mapShift[tmp_layer.outerMask != 0] = X
                tmp_layer.mapShift[tmp_layer.outerMask == 0] = xp.reshape(tmp_layer.OPD, tmp_layer.resolution*tmp_layer.resolution)
                # set back the flag to its default value
                tmp_layer.notDoneOnce = True
                # attribute to each layer the modified layer
                setattr(self, 'layer_'+str(i_layer+1), tmp_layer)
                # re-intialise the atmosphere phase-support
                OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
                # wavelength scaling
                tmp_layer.OPD *= self.wavelength/2/xp.pi
        self.hasNotBeenInitialized = False
        self.src_list = []
        # reset the r0 and generate a new phase screen to override the ro_def computation
        self.r0 = self.r0
        self.generateNewPhaseScreen(0)
        # relay to the src to initialize the variables
        self.relay(self.src)
        print(self)

    def buildLayer(self, telescope, r0, L0, i_layer, compute_covariance=True):
        """
            Generation of phase screens using the method introduced in Assemat et al (2006)
        """
        # initialize layer object
        layer = emptyClass()
        # create a random state to allow reproductible sequences of phase screens
        layer.randomState = RandomState(42+i_layer*1000)
        # gather properties of the atmosphere
        layer.altitude = self.altitude[i_layer]
        layer.windSpeed = self.windSpeed[i_layer]
        layer.direction = self.windDirection[i_layer]
        # compute the X and Y wind speed
        layer.vY = layer.windSpeed*xp.cos(xp.deg2rad(layer.direction))
        layer.vX = layer.windSpeed*xp.sin(xp.deg2rad(layer.direction))
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov = self.telescope.D+2*xp.tan(self.fov_rad/2)*layer.altitude
        layer.resolution_fov = int(xp.ceil((self.telescope.resolution/self.telescope.D)*layer.D_fov))
        # 4 pixels are added as a margin for the edges
        layer.resolution = layer.resolution_fov + 4
        layer.center = layer.resolution//2
        # diameter of the layer in [m]
        layer.D = layer.resolution * self.telescope.D / self.telescope.resolution
        layer.pupil_footprint = []
        layer.extra_sx = []
        layer.extra_sy = []
        for i_src in range(len(self.src_list)):
            [x_z, y_z] = pol2cart(layer.altitude*xp.tan(self.src_list[i_src].coordinates[0]/self.rad2arcsec) * layer.resolution / layer.D, xp.deg2rad(self.src_list[i_src].coordinates[1]))
            layer.extra_sx.append(int(x_z)-x_z)
            layer.extra_sy.append(int(y_z)-y_z)
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
            pupil_footprint_support = xp.zeros([layer.resolution, layer.resolution], dtype=self.precision())
            pupil_footprint_support[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2, center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2] = 1
            layer.pupil_footprint.append(pupil_footprint_support)
        # layer pixel size in [m]
        layer.pixel_size = layer.D/layer.resolution
        # number of extra pixel for the phase screens computation
        layer.n_extra_pixel = self.n_extra_pixel
        layer.nPixel = int(1+xp.round(layer.D/layer.pixel_size))
        print('-> Computing the initial phase screen...')
        a = time.time()
        layer.OPD = ft_sh_phase_screen(atm=self,
                                       resolution=layer.resolution,
                                       pixel_size=layer.D/layer.resolution,
                                       seed=i_layer)
        layer.initial_OPD = layer.OPD.copy()
        layer.seed = i_layer
        b = time.time()
        print('initial phase screen : ' + str(b-a) + ' s')

        # Outer ring of pixel for the phase screens update
        layer.outerMask = xp.ones([layer.resolution+layer.n_extra_pixel, layer.resolution+layer.n_extra_pixel], dtype=self.precision())
        layer.outerMask[1:-1, 1:-1] = 0

        # inner pixels that contains the phase screens
        layer.innerMask = xp.ones([layer.resolution+layer.n_extra_pixel, layer.resolution+layer.n_extra_pixel], dtype=self.precision())
        layer.innerMask -= layer.outerMask
        layer.innerMask[1+layer.n_extra_pixel:-1-layer.n_extra_pixel,
                        1+layer.n_extra_pixel:-1-layer.n_extra_pixel] = 0

        x = xp.linspace(0, layer.resolution+1, layer.resolution + 2, dtype=self.precision()) * layer.D/(layer.resolution-1)
        u, v = xp.meshgrid(x, x)

        layer.innerZ = u[layer.innerMask != 0] + 1j*v[layer.innerMask != 0]
        layer.outerZ = u[layer.outerMask != 0] + 1j*v[layer.outerMask != 0]
        if self.compute_covariance:

            layer.ZZt, layer.ZXt, layer.XXt, layer.ZZt_inv = self.get_covariance_matrices(layer)

            layer.ZZt_r0 = self.ZZt_r0.copy()
            layer.ZXt_r0 = self.ZXt_r0.copy()
            layer.XXt_r0 = self.XXt_r0.copy()
            layer.ZZt_inv_r0 = self.ZZt_inv_r0.copy()

            layer.A = xp.matmul(layer.ZXt_r0.T, layer.ZZt_inv_r0)
            layer.BBt = layer.XXt_r0 - xp.matmul(layer.A, layer.ZXt_r0)
            layer.B = xp.linalg.cholesky(layer.BBt)
            layer.mapShift = xp.zeros([layer.resolution+self.n_extra_pixel, layer.resolution+self.n_extra_pixel], dtype=self.precision())
            Z = layer.OPD[layer.innerMask[1:-1, 1:-1] != 0]
            X = xp.matmul(layer.A, Z) + xp.matmul(layer.B, layer.randomState.normal(size=layer.B.shape[1]))

            layer.mapShift[layer.outerMask != 0] = X
            layer.mapShift[layer.outerMask == 0] = xp.reshape(layer.OPD, layer.resolution*layer.resolution)
            layer.notDoneOnce = True
            layer.A = layer.A.astype(self.precision())
            layer.B = layer.A.astype(self.precision())
            print('Done!')
        return layer

    def add_row(self, layer, stepInPixel, map_full=None):
        if map_full is None:
            map_full = layer.mapShift
        shiftMatrix = translationImageMatrix(map_full, [stepInPixel[0], stepInPixel[1]])  # units are in pixel of the M1
        tmp = globalTransformation(map_full, shiftMatrix)
        onePixelShiftedPhaseScreen = tmp[1:-1, 1:-1]
        Z = onePixelShiftedPhaseScreen[layer.innerMask[1:-1, 1:-1] != 0]
        X = layer.A@Z + layer.B@layer.randomState.normal(size=layer.B.shape[1]).astype(self.precision())
        map_full[layer.outerMask != 0] = X
        map_full[layer.outerMask == 0] = xp.reshape(
            onePixelShiftedPhaseScreen, layer.resolution*layer.resolution)
        return onePixelShiftedPhaseScreen

    def set_pupil_footprint(self):
        for i_layer in range(self.nLayer):
            layer = getattr(self, 'layer_'+str(i_layer+1))
            layer.pupil_footprint = []
            layer.extra_sx = []
            layer.extra_sy = []
            for i_src in range(len(self.src_list)):
                src = self.src_list[i_src]
                if src.chromatic_shift is not None:
                    if len(src.chromatic_shift) == self.nLayer:
                        chromatic_shift = src.chromatic_shift[i_layer]
                    else:
                        raise OopaoError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')
                else:
                    chromatic_shift = 0
                [x_z, y_z] = pol2cart(layer.altitude*xp.tan((src.coordinates[0]+chromatic_shift)/self.rad2arcsec) * layer.resolution / layer.D, xp.deg2rad(src.coordinates[1]))
                layer.extra_sx.append(int(x_z)-x_z)
                layer.extra_sy.append(int(y_z)-y_z)
                center_x = int(y_z)+layer.resolution//2
                center_y = int(x_z)+layer.resolution//2
                pupil_footprint_support = xp.zeros([layer.resolution, layer.resolution], dtype=self.precision())
                pupil_footprint_support[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2, center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2] = 1
                layer.pupil_footprint.append(pupil_footprint_support)

    def updateLayer(self, layer, shift=None):
        if self.compute_covariance is False:
            raise OopaoError('The computation of the covariance matrices was set to False in the atmosphere initialisation. Set it to True to provide moving layers.')
        layer.pixel_scale = layer.D / (layer.resolution)
        ps_turb_x = layer.vX*self.telescope.samplingTime
        ps_turb_y = layer.vY*self.telescope.samplingTime

        if layer.vX == 0 and layer.vY == 0 and shift is None:
            layer.OPD = layer.OPD

        else:
            if layer.notDoneOnce:
                layer.notDoneOnce = False
                layer.ratio = xp.zeros(2)
                layer.ratio[0] = ps_turb_x/layer.pixel_size
                layer.ratio[1] = ps_turb_y/layer.pixel_size
                layer.buff = xp.zeros(2)

            if shift is None:
                ratio = layer.ratio
            else:
                ratio = shift    # shift in pixels
            tmpRatio = xp.abs(ratio)
            tmpRatio[xp.isinf(tmpRatio)] = 0
            nScreens = (tmpRatio)
            nScreens = nScreens.astype('int')

            stepInPixel = xp.zeros(2)
            stepInSubPixel = xp.zeros(2)

            for i in range(nScreens.min()):
                stepInPixel[0] = 1
                stepInPixel[1] = 1
                stepInPixel = stepInPixel*xp.sign(ratio)
                layer.OPD = self.add_row(layer, stepInPixel)

            for j in range(nScreens.max()-nScreens.min()):
                stepInPixel[0] = 1
                stepInPixel[1] = 1
                stepInPixel = stepInPixel*xp.sign(ratio)
                stepInPixel[xp.where(nScreens == nScreens.min())] = 0
                layer.OPD = self.add_row(layer, stepInPixel)

            stepInSubPixel[0] = (xp.abs(ratio[0]) % 1)*xp.sign(ratio[0])
            stepInSubPixel[1] = (xp.abs(ratio[1]) % 1)*xp.sign(ratio[1])

            layer.buff += stepInSubPixel
            if xp.abs(layer.buff[0]) >= 1 or xp.abs(layer.buff[1]) >= 1:
                stepInPixel[0] = 1*xp.sign(layer.buff[0])
                stepInPixel[1] = 1*xp.sign(layer.buff[1])
                stepInPixel[xp.where(xp.abs(layer.buff) < 1)] = 0
                layer.OPD = self.add_row(layer, stepInPixel)

            layer.buff[0] = (xp.abs(layer.buff[0]) % 1)*xp.sign(layer.buff[0])
            layer.buff[1] = (xp.abs(layer.buff[1]) % 1)*xp.sign(layer.buff[1])

            shiftMatrix = translationImageMatrix(
                layer.mapShift, [layer.buff[0], layer.buff[1]])  # units are in pixel of the M1
            layer.OPD = globalTransformation(
                layer.mapShift, shiftMatrix)[1:-1, 1:-1]

    def update(self, OPD=None):
        if self.hasNotBeenInitialized:
            raise OopaoError('The Atmosphere object needs to be initialised using the initialiseAtmosphere()')
        if OPD is None:
            self.is_user_defined_opd = False
            for i_layer in range(self.nLayer):
                tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                self.updateLayer(tmp_layer)
        else:
            self.is_user_defined_opd = True
            self.user_defined_opd = OPD
        if self.telescope.isPaired:
            self.telescope.src**self*self.telescope
        return

    def relay(self, src):
        # update the src attached to the atmosphere
        self.src = src
        # different cases between single and multiple sources
        if src.tag == 'source':
            self.src_list = [src]
            self.asterism = None
        elif src.tag == 'asterism':
            self.src_list = src.src
            self.asterism = src
        if self.is_user_defined_opd:
            OPD_support = [self.user_defined_opd]*len(self.src_list)
            warning('User-Defined OPD are only propagated once in the Atmosphere class. Consider using the OPD_map class.')
        else:
            # compute the pupil footprint for each layers and each source
            self.set_pupil_footprint()
            for src in self.src_list:
                if src.coordinates[0] > self.fov/2:
                    raise OopaoError('The source object zenith ('+str(src.coordinates[0])+'") is outside of the telescope fov ('+str(
                        self.fov//2)+'")! You can:\n - Reduce the zenith of the source \n - Re-initialize the atmosphere object using a telescope with a larger fov')
                src.through_atm = True
                src.optical_path.append([self.tag, self])
            # intialize the OPD support
            OPD_support = self.initialize_OPD_support()
            # fill the support for each layer
            for i_layer in range(self.nLayer):
                tmp_layer = getattr(self, 'layer_' + str(i_layer + 1))
                OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
        self.set_OPD(OPD_support)
        self.is_user_defined_opd = False
        return

    def initialize_OPD_support(self):
        OPD_support = []
        for i in range(len(self.src_list)):
            OPD_support.append(xp.zeros([self.telescope.resolution, self.telescope.resolution], dtype=self.precision()))
        return OPD_support

    def fill_OPD_support(self, tmp_layer, OPD_support, i_layer):
        for i_src in range(len(self.src_list)):
            if self.src_list[i_src].altitude <= tmp_layer.altitude:
                raise OopaoError('The source altitude ('+str(self.src_list[i_src].altitude[i_src])+' m) is below or at the same altitude as the atmosphere layer ('+str(tmp_layer.altitude)+' m)')
            _im = tmp_layer.OPD.copy()
            if tmp_layer.extra_sx[i_src] != 0 or tmp_layer.extra_sy[i_src] != 0:
                pixel_size_in = 1
                pixel_size_out = 1
                resolution_out = _im.shape[0]
                _im = xp.squeeze(interpolate_image(_im, pixel_size_in, pixel_size_out,
                                 resolution_out, shift_x=tmp_layer.extra_sx[i_src], shift_y=tmp_layer.extra_sy[i_src]))
            interpolate_cone_effect = False
            if self.src_list[i_src].altitude != np.inf:
                sub_im = xp.reshape(_im[xp.where(tmp_layer.pupil_footprint[i_src] == 1)], [self.telescope.resolution, self.telescope.resolution])
                h = self.src_list[i_src].altitude-tmp_layer.altitude
                if xp.isinf(h):
                    # magnification due to cone effect not considered
                    magnification_cone_effect = 1
                else:
                    # magnification due to cone effect considered
                    magnification_cone_effect = (h)/self.src_list[i_src].altitude
                    interpolate_cone_effect = True
                    pixel_size_in = 1
                    pixel_size_out = pixel_size_in*magnification_cone_effect
                    resolution_out = self.telescope.resolution
                    cube_in = xp.atleast_3d(sub_im).T
            if interpolate_cone_effect:
                _im = xp.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out)).T
            else:
                _im = _im[tmp_layer.pupil_footprint[i_src] == 1].reshape(self.telescope.resolution, self.telescope.resolution)
            _im *= self.wavelength/2/xp.pi
            _im *= xp.sqrt(self.fractionalR0[i_layer])
            OPD_support[i_src] += _im
        return OPD_support

    def set_OPD(self, OPD_support):
        for i, src in enumerate(self.src_list):
            src.OPD_no_pupil = OPD_support[i]
            src.OPD = src.OPD_no_pupil*src.mask
        self.OPD = np.squeeze(np.array(OPD_support))
        return

    def get_covariance_matrices(self, layer):
        # Compute the covariance matrices
        compute_covariance_matrices = True

        if self.fov_rad == 0:
            try:
                c = time.time()
                self.ZZt_r0 = self.ZZt_r0
                d = time.time()
                print('ZZt.. : ' + str(d-c) + ' s')
                self.ZXt_r0 = self.ZXt_r0
                e = time.time()
                print('ZXt.. : ' + str(e-d) + ' s')
                self.XXt_r0 = self.XXt_r0
                f = time.time()
                print('XXt.. : ' + str(f-e) + ' s')
                self.ZZt_inv_r0 = self.ZZt_inv_r0
                print('SCAO system considered: covariance matrices were already computed!')
                compute_covariance_matrices = False
            except:
                compute_covariance_matrices = True
        if compute_covariance_matrices:
            c = time.time()
            self.ZZt = makeCovarianceMatrix(layer.innerZ, layer.innerZ, self)
            if self.param is None:
                self.ZZt_inv = xp.linalg.pinv(self.ZZt)
            else:
                try:
                    print('Loading pre-computed data...')
                    name_data = 'ZZt_inv_spider_L0_'+str(self.L0)+'_m_r0_'+str(self.r0_def)+'_shape_'+str(self.ZZt.shape[0])+'x'+str(self.ZZt.shape[1])+'.json'
                    location_data = self.param['pathInput'] + \
                        self.param['name'] + '/sk_v/'
                    try:
                        with open(location_data+name_data) as f:
                            C = json.load(f)
                        data_loaded = jsonpickle.decode(C)
                    except:
                        createFolder(location_data)
                        with open(location_data+name_data) as f:
                            C = json.load(f)
                        data_loaded = jsonpickle.decode(C)
                    self.ZZt_inv = data_loaded['ZZt_inv']

                except:
                    print('Something went wrong.. re-computing ZZt_inv ...')
                    name_data = 'ZZt_inv_spider_L0_'+str(self.L0)+'_m_r0_'+str(self.r0_def)+'_shape_'+str(self.ZZt.shape[0])+'x'+str(self.ZZt.shape[1])+'.json'
                    location_data = self.param['pathInput'] + \
                        self.param['name'] + '/sk_v/'
                    createFolder(location_data)
                    self.ZZt_inv = xp.linalg.pinv(self.ZZt)
                    print('saving for future...')
                    data = dict()
                    data['pupil'] = self.telescope.pupil
                    data['ZZt_inv'] = self.ZZt_inv
                    data_encoded = jsonpickle.encode(data)
                    with open(location_data+name_data, 'w') as f:
                        json.dump(data_encoded, f)
            d = time.time()
            print('ZZt.. : ' + str(d-c) + ' s')
            self.ZXt = makeCovarianceMatrix(layer.innerZ, layer.outerZ, self)
            e = time.time()
            print('ZXt.. : ' + str(e-d) + ' s')
            self.XXt = makeCovarianceMatrix(layer.outerZ, layer.outerZ, self)
            f = time.time()
            print('XXt.. : ' + str(f-e) + ' s')

            self.ZZt_r0 = self.ZZt*(self.r0_def/self.r0)**(5/3)
            self.ZXt_r0 = self.ZXt*(self.r0_def/self.r0)**(5/3)
            self.XXt_r0 = self.XXt*(self.r0_def/self.r0)**(5/3)
            self.ZZt_inv_r0 = self.ZZt_inv/((self.r0_def/self.r0)**(5/3))
        return self.ZZt, self.ZXt, self.XXt, self.ZZt_inv

    def generateNewPhaseScreen(self, seed=None):
        if seed is None:
            t = time.localtime()
            seed = t.tm_hour*3600 + t.tm_min*60 + t.tm_sec
        OPD_support = self.initialize_OPD_support()
        for i_layer in range(self.nLayer):
            tmp_layer = getattr(self, 'layer_'+str(i_layer+1))

            if self.mode == 1:
                raise DeprecationWarning("The dependency to the aotools package has been deprecated.")
            else:
                if self.mode == 2:
                    # with subharmonics
                    phase = ft_sh_phase_screen(self, tmp_layer.resolution, tmp_layer.D/tmp_layer.resolution, seed=seed+i_layer)
                else:
                    phase = ft_phase_screen(self, tmp_layer.resolution, tmp_layer.D/tmp_layer.resolution, seed=seed+i_layer)

            tmp_layer.OPD = phase
            tmp_layer.randomState = RandomState(seed+i_layer*1000)
            if self.compute_covariance:
                Z = tmp_layer.OPD[tmp_layer.innerMask[1:-1, 1:-1] != 0]
                X = xp.matmul(tmp_layer.A, Z) + xp.matmul(tmp_layer.B, tmp_layer.randomState.normal(size=tmp_layer.B.shape[1]))
                tmp_layer.mapShift[tmp_layer.outerMask != 0] = X
                tmp_layer.mapShift[tmp_layer.outerMask == 0] = xp.reshape(tmp_layer.OPD, tmp_layer.resolution*tmp_layer.resolution)
                tmp_layer.notDoneOnce = True

            setattr(self, 'layer_'+str(i_layer+1), tmp_layer)
            OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
        if self.telescope.isPaired:
            self.telescope.src**self*self.telescope

    def print_atm_at_wavelength(self, wavelength):
        r0_wvl = self.r0*((wavelength/self.wavelength)**(6/5))
        seeingArcsec_wvl = self.rad2arcsec*(wavelength/r0_wvl)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE AT ' +
              str(wavelength)+' nm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('r0 \t\t'+str(r0_wvl) + ' \t [m]')
        print('Seeing \t' + str(xp.round(seeingArcsec_wvl, 2)) + str('\t ["]'))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return r0_wvl, seeingArcsec_wvl

    # kept for backward compatibility
    def __mul__(self, obj):
        if obj.tag == 'telescope' or obj.tag == 'source' or obj.tag == 'asterism':
            if obj.tag == 'telescope':
                if self.fov == obj.fov:
                    self.telescope = obj
                else:
                    print('Re-initializing the atmosphere to match the new telescope fov')
                    self.hasNotBeenInitialized = True
                    self.initializeAtmosphere(obj)
            elif obj.tag == 'source' or obj.tag == 'asterism':
                self.relay(obj)
            return obj
        else:
            raise OopaoError('The atmosphere can be multiplied only with a Telescope or a Source object!')

    def display_atm_layers(self, layer_index=None, fig_index=None, list_src=None):
        if self.hasNotBeenInitialized:
            raise OopaoError('The atmosphere must be initialized first to make use of the display_atm_layers() method')
        display_cn2 = False
        if layer_index is None:
            layer_index = list(xp.arange(self.nLayer))
            n_sp = len(layer_index)
            display_cn2 = True
        else:
            n_sp = len(layer_index)
            display_cn2 = True
        if type(layer_index) is not list:
            raise OopaoError('layer_index should be a list')
        # when sources not yet propagated through atmosphere, we need to use src_list from telescope
        if len(self.src_list) == 0:
            if len(self.telescope.src_list) >= 1:
                list_src = self.telescope.src_list
            else:
                raise OopaoError('No sources yet associated/propagated either through atmosphere or telescope')
        # when already propagated through atmosphere
        else:
            list_src = self.src_list
        plt.figure(fig_index, figsize=[n_sp*4, 3*(1+display_cn2)], edgecolor=None)
        if display_cn2:
            gs = gridspec.GridSpec(1, n_sp+1, height_ratios=[1], width_ratios=xp.ones(n_sp+1), hspace=0.5, wspace=0.5)
        else:
            gs = gridspec.GridSpec(1, n_sp, height_ratios=xp.ones(1), width_ratios=xp.ones(n_sp), hspace=0.25, wspace=0.25)

        axis_list = []
        for i in range(len(layer_index)):
            axis_list.append(plt.subplot(gs[0, i]))

        if display_cn2:
            ax = plt.subplot(gs[0, -1])
            for i_layer in range(self.nLayer):
                p = ax.barh(self.altitude[i_layer]*1e-3, 100*np.round(self.fractionalR0[i_layer], 2), height=1.5, edgecolor='k', label='Layer '+str(i_layer+1))
                ax.bar_label(p, label_type='center')
            ax.legend()
            plt.xlabel('Fractional Cn2 [%]')
            plt.ylabel('Altitude [km]')

        for i_l, ax in enumerate(axis_list):
            tmp_layer = getattr(self, 'layer_'+str(layer_index[i_l]+1))
            ax.imshow(tmp_layer.OPD, extent=[-tmp_layer.D/2, tmp_layer.D/2, -tmp_layer.D/2, tmp_layer.D/2])
            center = tmp_layer.D/2
            [x_tel, y_tel] = pol2cart(tmp_layer.D_fov/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
            cm = plt.get_cmap('gist_rainbow')
            col = []
            for i_source in range(len(list_src)):
                col.append(cm(1.*i_source/len(list_src)))
                [x_c, y_c] = pol2cart(self.telescope.D/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
                h = list_src[i_source].altitude-tmp_layer.altitude
                if xp.isinf(h):
                    r = self.telescope.D/2
                else:
                    r = (h)/self.src.altitude[i_source]*self.telescope.D/2
                [x_cone, y_cone] = pol2cart(
                    r, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
                if list_src[i_source].chromatic_shift is not None:
                    if len(list_src[i_source].chromatic_shift) == self.nLayer:
                        chromatic_shift = list_src[i_source].chromatic_shift[i_l]
                    else:
                        raise OopaoError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')
                else:
                    chromatic_shift = 0
                [x_z, y_z] = pol2cart(tmp_layer.altitude*xp.tan((list_src[i_source].coordinates[0] + chromatic_shift)/self.rad2arcsec), xp.deg2rad(list_src[i_source].coordinates[1]))
                center = 0
                [x_c, y_c] = pol2cart(tmp_layer.D_fov/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
                nm = (list_src[i_source].type) + '@' + str(list_src[i_source].coordinates[0])+'"'
                ax.plot(x_cone+x_z+center, y_cone+y_z+center, '-', color=col[i_source], label=nm)
                ax.fill(x_cone+x_z+center, y_cone+y_z+center,
                        y_z+center, alpha=0.25, color=col[i_source])
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.set_title('Altitude '+str(tmp_layer.altitude)+' m')
            ax.plot(x_tel+center, y_tel+center, '--', color='k')
            ax.legend(loc='upper left')
            makeSquareAxes(plt.gca())

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, val):
        self._r0 = val
        if self.hasNotBeenInitialized is False:
            print('Updating the Atmosphere covariance matrices...')
            self.seeingArcsec = self.rad2arcsec*(self.wavelength/val)
            self.cn2 = (self.r0**(-5. / 3) / (0.423 * (2*np.pi/self.wavelength)**2))/np.max([1, np.max(self.altitude)])  # Cn2 m^(-2/3)
            if self.compute_covariance:
                for i_layer in range(self.nLayer):
                    tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                    tmp_layer.ZZt_r0 = tmp_layer.ZZt*(self.r0_def/self.r0)**(5/3)
                    tmp_layer.ZXt_r0 = tmp_layer.ZXt*(self.r0_def/self.r0)**(5/3)
                    tmp_layer.XXt_r0 = tmp_layer.XXt*(self.r0_def/self.r0)**(5/3)
                    tmp_layer.ZZt_inv_r0 = tmp_layer.ZZt_inv / ((self.r0_def/self.r0)**(5/3))
                    BBt = tmp_layer.XXt_r0 - xp.matmul(tmp_layer.A, tmp_layer.ZXt_r0)
                    tmp_layer.B = xp.linalg.cholesky(BBt).astype(self.precision())

    @property
    def L0(self):
        return self._L0

    @L0.setter
    def L0(self, val):
        self._L0 = val
        if self.hasNotBeenInitialized is False:
            print('Updating the Atmosphere covariance matrices...')
            self.hasNotBeenInitialized = True
            del self.ZZt
            del self.XXt
            del self.ZXt
            del self.ZZt_inv
            self.initializeAtmosphere(self.telescope)

    @property
    def windSpeed(self):
        return self._windSpeed

    @windSpeed.setter
    def windSpeed(self, val):
        self._windSpeed = val

        if self.hasNotBeenInitialized is False:
            if len(val) != self.nLayer:
                raise OopaoError('Wrong value for the wind-speed! Make sure that you inpute a wind-speed for each layer')
            else:
                print('Updating the wind speed...')
                self.V0 = (np.sum(np.asarray(self.fractionalR0) * np.asarray(self.windSpeed))**(5/3))**(3/5)  # computation of equivalent wind speed, Roddier 1982
                self.tau0 = 0.31 * self.r0 / self.V0  # Coherence time of atmosphere, Roddier 1981
                for i_layer in range(self.nLayer):
                    tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                    tmp_layer.windSpeed = val[i_layer]
                    tmp_layer.vY = tmp_layer.windSpeed * xp.cos(xp.deg2rad(tmp_layer.direction))
                    tmp_layer.vX = tmp_layer.windSpeed * xp.sin(xp.deg2rad(tmp_layer.direction))
                    ps_turb_x = tmp_layer.vX*self.telescope.samplingTime
                    ps_turb_y = tmp_layer.vY*self.telescope.samplingTime
                    tmp_layer.ratio[0] = ps_turb_x/tmp_layer.pixel_size
                    tmp_layer.ratio[1] = ps_turb_y/tmp_layer.pixel_size
                    setattr(self, 'layer_'+str(i_layer+1), tmp_layer)

    @property
    def windDirection(self):
        return self._windDirection

    @windDirection.setter
    def windDirection(self, val):
        self._windDirection = val

        if self.hasNotBeenInitialized is False:
            if len(val) != self.nLayer:
                raise OopaoError('Wrong value for the wind-speed! Make sure that you inpute a wind-direction for each layer')
            else:
                print('Updating the wind direction...')
                for i_layer in range(self.nLayer):
                    tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                    tmp_layer.direction = val[i_layer]
                    tmp_layer.vY = tmp_layer.windSpeed * xp.cos(xp.deg2rad(tmp_layer.direction))
                    tmp_layer.vX = tmp_layer.windSpeed * xp.sin(xp.deg2rad(tmp_layer.direction))
                    ps_turb_x = tmp_layer.vX*self.telescope.samplingTime
                    ps_turb_y = tmp_layer.vY*self.telescope.samplingTime
                    tmp_layer.ratio[0] = ps_turb_x/tmp_layer.pixel_size
                    tmp_layer.ratio[1] = ps_turb_y/tmp_layer.pixel_size
                    setattr(self, 'layer_'+str(i_layer+1), tmp_layer)

    @property
    def fractionalR0(self):
        return self._fractionalR0

    @fractionalR0.setter
    def fractionalR0(self, val):
        self._fractionalR0 = val
        if self.hasNotBeenInitialized is False:
            if len(val) != self.nLayer:
                raise OopaoError('Wrong value for the fractional r0 ! Make sure that you inpute a fractional r0 for each layer!' +
                                 ' If you want to change the number of layer, re-generate a new atmosphere object.')
            else:
                print('Updating the fractional R0...BEWARE COMPLETE THE RECOMPUTATION...NOT ONLY V0 and Tau0 !')
                self.V0 = (np.sum(np.asarray(self.fractionalR0) * np.asarray(self.windSpeed))**(5/3))**(3/5)  # computation of equivalent wind speed, Roddier 1982
                self.tau0 = 0.31 * self.r0 / self.V0  # Coherence time of atmosphere, Roddier 1981

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['parameters'] = f"{'Layer':^7s}|{'Direction':^11s}|{'Speed':^7s}|{'Altitude':^10s}|{'Frac Cn²':^10s}|{'Diameter':^10s}|"
        self.prop['units'] = f"{'':^7s}|{'[°]':^11s}|{'[m/s]':^7s}|{'[m]':^10s}|{'[%]':^10s}|{'[m]':^10s}|"
        for i in range(self.nLayer):
            if i % 2 == 0:
                self.prop['layer_%02d' % i] = f"\033[00m{i+1:^7d}|{self.windDirection[i]:^11.0f}|{self.windSpeed[i]:^7.1f}|{self.altitude[i]:^10.0e}|{self.fractionalR0[i]*100:^10.0f}|{getattr(self, 'layer_'+str(i+1)).D:^10.3f}|"
            else:
                self.prop['layer_%02d' % i] = f"\033[47m{i+1:^7d}|{self.windDirection[i]:^11.0f}|{self.windSpeed[i]:^7.1f}|{self.altitude[i]:^10.0e}|{self.fractionalR0[i]*100:^10.0f}|{getattr(self, 'layer_'+str(i+1)).D:^10.3f}|"
        self.prop['delimiter'] = ''
        self.prop['r0'] = f"{'r0 @ 500 nm [m]':<16s}|{self.r0:^10.2f}"
        self.prop['L0'] = f"{'L0 [m]':<16s}|{self.L0:^10.1f}"
        self.prop['tau0'] = f"{'Tau0 [s]':<16s}|{self.tau0:^10.4f}"
        self.prop['V0'] = f"{'V0 [m/s]':<16s}|{self.V0:^10.2f}"
        self.prop['frequency'] = f"{'Frequency [Hz]':<16s}|{1/self.telescope.samplingTime:^10.1f}"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len)) - len('\033[00m')
        self.prop['delimiter'] = f'\033[00m{"":=^{n_char}}'
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Atmosphere ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
