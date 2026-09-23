# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:59:02 2020

@author: cheritie
"""
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
from .runtime import array_backend, gpu_resident, precision_bits
from .tools.gpuTransforms import translate_cubic
from .tools.separableInterpolation import SeparableTaps, shift_crop_zoom_taps, stack_taps
import scipy.fft
import scipy.linalg
xp, global_gpu_flag = array_backend()
from .runtime import backend_of as _backend_of, to_backend as _to_backend


def _flat_view(array):
    """Writable 1D view of a C-contiguous array (a[idx] = v on it writes into `array`)."""
    if not array.flags.c_contiguous:
        raise OopaoError('Internal error: expected a C-contiguous phase-screen support.')
    return array.reshape(-1)


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
                 elevation: float = 90.0,
                 mode: float = 2,
                 angular_spectrum_propagation: bool = False,
                 geometric_phase_backup: bool = False,
                 unwrap_diffractive_phase=False,
                 rytov_var: float = None,
                 rytov_wvl: float = 500e-9,
                 t_boiling=None):
        """ ATMOSPHERE.
        An Atmosphere is made of one or several layer of turbulence that follow the Van Karmann statistics.
        Each layer is considered to be independant to the other ones and has its own properties (direction, speed, etc.)
        The Atmosphere object can be defined for a single Source object (default) or multi Source Object.
        The Source coordinates allow to span different areas in the field (defined by the telescope.fov).
        If the source type is an LGS the cone effect is considered using an interpolation.
        NGS and LGS can be combined together in the Asterism object.
        The convention chosen is that all the wavelength-dependant atmosphere parameters are expressed at 500 nm.
        The atmosphere is computed at the elevation defined by the user (by default at 90° zenith) and can be updated on the fly setting the atmosphere.elevation value.
        The atmosphere has three propagation modes :
            1 : geometric propagation only (no scintillation, no diffractive effects) -> atm.angular_spectrum_propagation = False and atm.geometric_phase_backup = False
            2 : diffractive propagation with scintillation -> atm.angular_spectrum_propagation = True and atm.geometric_phase_backup = False
            3 : diffractive propagation with scintillation and geometric phase backup -> atm.angular_spectrum_propagation = True and atm.geometric_phase_backup = True
        Parameters
        ----------
        telescope : Telescope
            The telescope object to which the Atmosphere is associated.
            This object carries the phase, flux, pupil information and sampling time as well as the type of source (NGS/LGS, source/asterism).
        r0 : float
            the Fried Parameter in [m], expressed at 500 nm and at the elevation defined by the user.
        L0 : float
            Outer scale parameter in [m].
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
            If the system contains multiple source, an asterism should be input to the atmosphere object.
            The default is None.
        elevation : float, optional
            Elevation of the source in degrees. This is used to compute the effective r0 and altitude of the layers.
            The default is 90 deg (i.e. zenith).
        angular_spectrum_propagation : bool, optional
            If True, the scintillation is computed using a diffractive propagation of the wavefront through the layers.
            If False, only the geometric phase is computed (No Amplitude effect).
            The default is False.
        geometric_phase_backup : bool, optional
            If True, the geometric phase is kept as a backup and can be used to compare the effect of the scintillation on the final phase.
            The default is False.
        rytov_var : float, optional
            If not None, the variance of the Rytov fluctuations is set to this value. This allows to simulate specific scintillation conditions.
            The default is None, in which case the Rytov variance is computed from the Cn2 profile and the wavelength using the formula from Tatarskii (1961).
        rytov_wvl : float, optional
            Wavelength at which the Rytov variance is computed in [m]. The default is 500 nm.
        t_boiling : float or list, optional
            Boiling (atmospheric turbulence decorrelation) time-constant in [s]. By default (None) the layers follow the frozen-flow (Taylor) hypothesis only.
            When set, each phase screen additionally decorrelates in time following a first-order auto-regressive (AR1) process with characteristic time t_boiling:
                phi(t+dt) = alpha * phi(t) + sqrt(1 - alpha**2) * phi_noise 
            with 
            alpha = exp(-dt / t_boiling) and dt = telescope.samplingTime.
            phi_noise is a fresh, independent phase screen sharing the same (r0, L0) Van-Karman statistics, so the spatial variance is preserved while the temporal
            auto-correlation decays as exp(-tau / t_boiling).
            A scalar is applied to every layer; a list provides one value per layer.
            Boiling and frozen-flow translation are applied simultaneously, and boiling also works for layers with zero wind-speed.
            The default is None.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** PROPERTIES **************************

        The main properties of the Atmosphere object are listed here:

        _ atm.OPD : Optical Path Difference in [m] truncated by the telescope pupil. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.OPD_no_pupil : Optical Path Difference in [m]. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.r0
        _ atm.L0
        _ atm.nLayer                                : number of turbulence layers
        _ atm.seeingArcsec                          : seeing in arcsec at 500 nm
        _ atm.layer_X                               : access the child object corresponding to the layer X where X starts at 0
        _ atm.elevation                             : elevation of the source in degrees.
        _ atm.angular_spectrum_propagation          : if True, the propagation is computed using a diffractive propagation of the wavefront through the layers. If False, only the geometric phase is computed.
        _ atm.geometric_phase_backup                : if True, the geometric phase is kept as a backup and can be used to compare the effect of the scintillation on the final phase.

        The main properties of the object can be displayed using :
            atm.print_properties()

        the following properties can be updated on the fly:
            _ atm.r0
            _ atm.windSpeed
            _ atm.windDirection
            _ atm.elevation
            _ atm.t_boiling
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
        precision = precision_bits()
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is np.float32:
            self.precision_complex = np.complex64
        else:
            self.precision_complex = np.complex128
        self.gpu_available = global_gpu_flag
        self.gpu_resident = gpu_resident()
        if self.gpu_available:
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
        else:
            self.convert_for_gpu = lambda a: a
            self.convert_for_numpy = lambda a: a
        self.hasNotBeenInitialized = True
        # caches: pupil masks on the working backend (CPU/GPU), angular-spectrum kernels
        self._mask_cache = {}
        self._asm_cache = {}
        self._asm_batch_cache = {}
        # propagate all the sources of an asterism at once (False: one after the other)
        self.parallel_sources = True
        # number of CPU threads for the batched FFTs (-1: all the cores)
        self.fft_workers = -1
        # inversion of the covariance matrices of the phase screens: 'cholesky' or 'pinv' (SVD)
        self.covariance_inversion = 'cholesky'
        # Elevation initialization
        self.angular_spectrum_propagation = angular_spectrum_propagation
        self.geometric_phase_backup = geometric_phase_backup
        self.unwrap_diffractive_phase = unwrap_diffractive_phase
        self.altitude = altitude  # altitude of the atmospheric layers
        self.wavelength = 500*1e-9  # Wavelengt used to define the properties of the atmosphere
        self._elevation = max(5.0, float(elevation))
        el_rad = np.radians(self._elevation)
        self.sampling_checked = False
        self._saturation_warned = False  # whether the Rytov saturation warning was already emitted
        self.altitude_zenith = [h * np.sin(el_rad) for h in self.altitude]
        self.r0_def = 0.15  # Default Fried Parameter in m at 500 nm to build covariance matrices once and scale them later
        self.r0 = r0  # User input Fried Parameter in m at 500 nm
        self.rad2arcsec = (180. / np.pi) * 3600
        self.fractionalR0 = fractionalR0      # Fractional Cn2 profile in percentage
        self.cn2 = (self.r0**(-5. / 3) / (0.423 * (2*np.pi/self.wavelength)**2))/np.max([1, np.max(self.altitude)])  # Cn2 m^(-2/3)
        self.L0 = L0                # Outer Scale in m
        self.nLayer = len(fractionalR0)     # number of layer
        self.windSpeed = windSpeed         # wind speed of the layers in m/s
        self.windDirection = windDirection     # wind direction in degrees
        self.t_boiling = t_boiling             # boiling (AR1 decorrelation) time-constant(s) in [s], None = frozen flow only
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
        # Rytov initialization
        self.rytov_wvl = float(rytov_wvl)
        self.update_rytov_variance()
        if rytov_var is not None:
            self.update_rytov_variance()
            self.rytov_var = float(rytov_var)

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
                tmp_layer.OPD *= self.wavelength/2/np.pi
        else:
            print('Re-setting the atmosphere to its initial state...')
            self.r0 = self.initial_r0
            for i_layer in range(self.nLayer):
                print('Updating layer' + str(i_layer+1) + '/' + str(self.nLayer) + ' ...')
                # access each layer to modify its properties
                tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                # re-load the saved initial OPD
                tmp_layer.OPD = tmp_layer.initial_OPD/self.wavelength*2*np.pi
                # reset the random state to its initial value
                tmp_layer.randomState = RandomState(42+i_layer*1000)
                # reset the boiling noise seed counter for reproducibility
                tmp_layer.boiling_seed = 100000 + i_layer * 100000
                # reset the phase-screen support from the initial OPD
                self._reset_map_shift(tmp_layer)
                # set back the flag to its default value
                tmp_layer.notDoneOnce = True
                # attribute to each layer the modified layer
                setattr(self, 'layer_'+str(i_layer+1), tmp_layer)
                # re-intialise the atmosphere phase-support
                OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
                # wavelength scaling
                tmp_layer.OPD *= self.wavelength/2/np.pi
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
        layer.vY = layer.windSpeed*np.cos(np.deg2rad(layer.direction))
        layer.vX = layer.windSpeed*np.sin(np.deg2rad(layer.direction))
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov = self.telescope.D+2*np.tan(self.fov_rad/2)*layer.altitude
        layer.resolution_fov = int(np.ceil((self.telescope.resolution/self.telescope.D)*layer.D_fov))
        # 4 pixels are added as a margin for the edges
        layer.resolution = layer.resolution_fov + 4
        layer.center = layer.resolution//2
        # diameter of the layer in [m]
        layer.D = layer.resolution * self.telescope.D / self.telescope.resolution
        # pupil footprint of each source in the layer (no chromatic shift at creation)
        self._compute_footprints(layer, i_layer, [0] * len(self.src_list))
        # layer pixel size in [m]
        layer.pixel_size = layer.D/layer.resolution
        # number of extra pixel for the phase screens computation
        layer.n_extra_pixel = self.n_extra_pixel
        layer.nPixel = int(1+np.round(layer.D/layer.pixel_size))
        print('-> Computing the initial phase screen...')
        a = time.time()
        layer.OPD = ft_sh_phase_screen(atm=self,
                                       resolution=layer.resolution,
                                       pixel_size=layer.D/layer.resolution,
                                       seed=i_layer)
        layer.initial_OPD = layer.OPD.copy()
        layer.seed = i_layer
        # boiling (AR1 temporal decorrelation) state for this layer
        layer.t_boiling = self.t_boiling[i_layer]
        # dedicated seed counter so the boiling noise screens are independent of the
        # frozen-flow extrusion randomness and reproducible across runs
        layer.boiling_seed = 100000 + i_layer * 100000
        b = time.time()
        print('initial phase screen : ' + str(b-a) + ' s')

        # Outer ring of pixel for the phase screens update
        layer.outerMask = np.ones([layer.resolution+layer.n_extra_pixel, layer.resolution+layer.n_extra_pixel], dtype=self.precision())
        layer.outerMask[1:-1, 1:-1] = 0

        # inner pixels that contains the phase screens
        layer.innerMask = np.ones([layer.resolution+layer.n_extra_pixel, layer.resolution+layer.n_extra_pixel], dtype=self.precision())
        layer.innerMask -= layer.outerMask
        layer.innerMask[1+layer.n_extra_pixel:-1-layer.n_extra_pixel,
                        1+layer.n_extra_pixel:-1-layer.n_extra_pixel] = 0

        x = np.linspace(0, layer.resolution+1, layer.resolution + 2, dtype=self.precision()) * layer.D/(layer.resolution-1)
        u, v = np.meshgrid(x, x)

        layer.innerZ = u[layer.innerMask != 0] + 1j*v[layer.innerMask != 0]
        layer.outerZ = u[layer.outerMask != 0] + 1j*v[layer.outerMask != 0]
        if self.compute_covariance:

            layer.ZZt, layer.ZXt, layer.XXt, layer.ZZt_inv = self.get_covariance_matrices(layer)

            layer.ZZt_r0 = self.ZZt_r0.copy()
            layer.ZXt_r0 = self.ZXt_r0.copy()
            layer.XXt_r0 = self.XXt_r0.copy()
            layer.ZZt_inv_r0 = self.ZZt_inv_r0.copy()

            layer.A = np.matmul(layer.ZXt_r0.T, layer.ZZt_inv_r0)
            layer.BBt = layer.XXt_r0 - np.matmul(layer.A, layer.ZXt_r0)
            layer.B = np.linalg.cholesky(layer.BBt)
            layer.mapShift = np.zeros([layer.resolution+self.n_extra_pixel, layer.resolution+self.n_extra_pixel], dtype=self.precision())
            Z = layer.OPD[layer.innerMask[1:-1, 1:-1] != 0]
            X = np.matmul(layer.A, Z) + np.matmul(layer.B, layer.randomState.normal(size=layer.B.shape[1]))

            layer.mapShift[layer.outerMask != 0] = X
            layer.mapShift[layer.outerMask == 0] = np.reshape(layer.OPD, layer.resolution*layer.resolution)
            layer.notDoneOnce = True
            layer.A = layer.A.astype(self.precision())
            layer.B = layer.B.astype(self.precision())
            print('Done!')
            if self.gpu_available:
                layer.A = self.convert_for_gpu(layer.A)
                layer.B = self.convert_for_gpu(layer.B)
                layer.outerMask = self.convert_for_gpu(layer.outerMask)
                layer.innerMask = self.convert_for_gpu(layer.innerMask)
                layer.mapShift = self.convert_for_gpu(layer.mapShift)
        return layer

    def get_t_boiling(self, val):
        if val is None:
            return [None] * self.nLayer
        if np.isscalar(val):
            return [val] * self.nLayer
        val = list(val)
        if len(val) != self.nLayer:
            raise OopaoError('Wrong value for t_boiling! Provide either a scalar '
                             '(applied to each layer) or a list with a value per layer.')
        return val

    def apply_boiling(self, layer):
        t_b = layer.t_boiling
        # frozen flow: no boiling requested or degenerate value
        if t_b is None or not np.isfinite(t_b) or t_b <= 0:
            return False
        dt = self.telescope.samplingTime
        alpha = float(np.exp(-dt / t_b))
        # alpha == 1 means perfect correlation (frozen). Skip the screen
        # generation when the per-step decorrelation is negligible (t_boiling >> dt).
        if alpha >= 1.0 - 1e-6:
            return False
        beta = float(np.sqrt(max(0.0, 1.0 - alpha ** 2)))
        # new independent phase screen on the same support as mapShift
        layer.boiling_seed += 1
        screen_resolution = layer.mapShift.shape[0]
        pixel_size = layer.D / layer.resolution
        backend = _backend_of(layer.mapShift)
        if self.mode == 2:
            phi_boiling = ft_sh_phase_screen(self, screen_resolution, pixel_size, seed=layer.boiling_seed, backend=backend)
        else:
            phi_boiling = ft_phase_screen(self, screen_resolution, pixel_size, seed=layer.boiling_seed, backend=backend)
        layer.mapShift = (alpha * layer.mapShift + beta * phi_boiling).astype(self.precision())
        return True

    def _support_indices(self, layer):
        """Flat indices of the inner ring, outer ring and centre of the phase-screen support.

        Boolean-mask indexing forces a GPU synchronization at every call (the size of the result must be
        read back); these integer indices are computed once per layer, on the backend of the masks.
        """
        cache = getattr(layer, '_support_idx', None)
        if cache is None or cache['outer'] is not layer.outerMask or cache['inner'] is not layer.innerMask:
            backend = _backend_of(layer.outerMask)
            inner = _to_backend(layer.innerMask, np)
            outer = _to_backend(layer.outerMask, np)
            cache = {'outer': layer.outerMask, 'inner': layer.innerMask,
                     'inner_idx': backend.asarray(np.flatnonzero(inner[1:-1, 1:-1] != 0)),
                     'outer_idx': backend.asarray(np.flatnonzero(outer != 0)),
                     'center_idx': backend.asarray(np.flatnonzero(outer == 0))}
            layer._support_idx = cache
        return cache

    def _reset_map_shift(self, layer, cast_random=False):
        """Rebuild the phase-screen support from layer.OPD: the OPD fills the centre and the outer ring is extruded."""
        backend = _backend_of(layer.A)
        idx = self._support_indices(layer)
        opd = backend.asarray(layer.OPD).ravel()
        rand = layer.randomState.normal(size=layer.B.shape[1])
        if cast_random:
            rand = rand.astype(self.precision())
        X = layer.A@opd[idx['inner_idx']] + layer.B@backend.asarray(rand)
        flat = _flat_view(layer.mapShift)
        flat[idx['outer_idx']] = X
        flat[idx['center_idx']] = opd

    def add_row(self, layer, stepInPixel, map_full=None):
        if map_full is None:
            map_full = layer.mapShift
        xp_ = _backend_of(map_full)
        if self.gpu_available:
            shifted = translate_cubic(map_full, stepInPixel)
        else:
            shiftMatrix = translationImageMatrix(map_full, stepInPixel)
            shifted = globalTransformation(map_full, shiftMatrix)
        onePixelShiftedPhaseScreen = shifted[1:-1, 1:-1]
        idx = self._support_indices(layer)
        Z = onePixelShiftedPhaseScreen.ravel()[idx['inner_idx']]
        rand_vec = xp_.asarray(layer.randomState.normal(size=layer.B.shape[1]).astype(self.precision()))
        X = layer.A@Z + layer.B@rand_vec
        flat = _flat_view(map_full)
        flat[idx['outer_idx']] = X
        flat[idx['center_idx']] = onePixelShiftedPhaseScreen.ravel()
        return onePixelShiftedPhaseScreen

    def _compute_footprints(self, layer, i_layer, chromatic_shifts):
        """Pupil footprint of each source in `layer`: sub-pixel shift, centre and crop slices.
        The footprint is a square block of the layer, so the part of the layer seen by a source is a plain slice
        (layer.footprint_slices), which works on both backends. The boolean maps (layer.pupil_footprint) are kept
        for backward compatibility. Nothing is recomputed while the sources and their shifts are unchanged.
        """
        key = tuple((tuple(float(c) for c in src.coordinates), float(cs)) for src, cs in zip(self.src_list, chromatic_shifts))
        if getattr(layer, '_footprint_key', None) == key:
            return
        n = self.telescope.resolution
        layer.pupil_footprint = []
        layer.extra_sx = []
        layer.extra_sy = []
        layer.footprint_slices = []
        for src, chromatic_shift in zip(self.src_list, chromatic_shifts):
            [x_z, y_z] = pol2cart(layer.altitude*np.tan((src.coordinates[0]+chromatic_shift)/self.rad2arcsec) * layer.resolution / layer.D, np.deg2rad(src.coordinates[1]))
            layer.extra_sx.append(int(x_z)-x_z)
            layer.extra_sy.append(int(y_z)-y_z)
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
            pupil_footprint_support = np.zeros([layer.resolution, layer.resolution], dtype=self.precision())
            pupil_footprint_support[center_x-n//2:center_x+n//2, center_y-n//2:center_y+n//2] = 1
            layer.pupil_footprint.append(pupil_footprint_support)
            layer.footprint_slices.append((slice(center_x-n//2, center_x+n//2), slice(center_y-n//2, center_y+n//2)))
        layer._footprint_key = key

    def set_pupil_footprint(self):
        for i_layer in range(self.nLayer):
            layer = getattr(self, 'layer_'+str(i_layer+1))
            chromatic_shifts = []
            for src in self.src_list:
                if src.chromatic_shift is not None:
                    if len(src.chromatic_shift) == self.nLayer:
                        chromatic_shifts.append(src.chromatic_shift[i_layer])
                    else:
                        raise OopaoError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')
                else:
                    chromatic_shifts.append(0)
            self._compute_footprints(layer, i_layer, chromatic_shifts)

    def updateLayer(self, layer, shift=None):
        if self.compute_covariance is False:
            raise OopaoError('The computation of the covariance matrices was set to False in the atmosphere initialisation. Set it to True to provide moving layers.')
        layer.pixel_scale = layer.D / (layer.resolution)
        ps_turb_x = layer.vX*self.telescope.samplingTime
        ps_turb_y = layer.vY*self.telescope.samplingTime

        # Boiling: temporally decorrelate the persistent phase-screen support (AR1 model).
        # Applied to layer.mapShift *before* the frozen-flow translation below so that the
        # temporal memory accumulates in the persistent state and composes with the wind shift.
        boiling_active = self.apply_boiling(layer)

        if layer.vX == 0 and layer.vY == 0 and shift is None:
            if boiling_active:
                # No wind: the screen still evolves through boiling only. Deliver the boiled
                # screen by re-extracting the interior of the (just updated) support.
                # Preserve a device layer in explicit GPU-resident mode.
                layer_opd = layer.mapShift.ravel()[self._support_indices(layer)['center_idx']].reshape(layer.resolution, layer.resolution)
                layer.OPD = layer_opd if self.gpu_resident else self.convert_for_numpy(layer_opd)
            else:
                layer.OPD = layer.OPD
        else:
            if layer.notDoneOnce:
                layer.notDoneOnce = False
                layer.ratio = np.zeros(2)
                layer.buff = np.zeros(2)
            if shift is None:
                layer.ratio[0] = ps_turb_x/layer.pixel_size
                layer.ratio[1] = ps_turb_y/layer.pixel_size
                ratio = layer.ratio
            else:
                ratio = shift    # shift in pixels
            tmpRatio = np.abs(ratio)
            tmpRatio[np.isinf(tmpRatio)] = 0
            nScreens = (tmpRatio)
            nScreens = nScreens.astype('int')
            stepInPixel = np.zeros(2)
            stepInSubPixel = np.zeros(2)
            for i in range(nScreens.min()):
                stepInPixel[0] = 1
                stepInPixel[1] = 1
                stepInPixel = stepInPixel*np.sign(ratio)
                layer.OPD = self.add_row(layer, stepInPixel)
            for j in range(nScreens.max()-nScreens.min()):
                stepInPixel[0] = 1
                stepInPixel[1] = 1
                stepInPixel = stepInPixel*np.sign(ratio)
                stepInPixel[np.where(nScreens == nScreens.min())] = 0
                layer.OPD = self.add_row(layer, stepInPixel)
            stepInSubPixel[0] = (np.abs(ratio[0]) % 1)*np.sign(ratio[0])
            stepInSubPixel[1] = (np.abs(ratio[1]) % 1)*np.sign(ratio[1])
            layer.buff += stepInSubPixel
            if np.abs(layer.buff[0]) >= 1 or np.abs(layer.buff[1]) >= 1:
                stepInPixel[0] = 1*np.sign(layer.buff[0])
                stepInPixel[1] = 1*np.sign(layer.buff[1])
                stepInPixel[np.where(np.abs(layer.buff) < 1)] = 0
                layer.OPD = self.add_row(layer, stepInPixel)
            layer.buff[0] = (np.abs(layer.buff[0]) % 1)*np.sign(layer.buff[0])
            layer.buff[1] = (np.abs(layer.buff[1]) % 1)*np.sign(layer.buff[1])
            if self.gpu_available:
                shifted = translate_cubic(layer.mapShift, layer.buff)
                layer.OPD = shifted[1:-1, 1:-1] if self.gpu_resident else self.convert_for_numpy(shifted[1:-1, 1:-1])
            else:
                shiftMatrix = translationImageMatrix(layer.mapShift, layer.buff)
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
            backend = xp if self.gpu_resident else np
            OPD_support = [_to_backend(self.user_defined_opd, backend)]*len(self.src_list)
            warning('User-Defined OPD are only propagated once in the Atmosphere class.')
            self.set_OPD(OPD_support)
            self.is_user_defined_opd = False
            return
        # compute the pupil footprint
        self.set_pupil_footprint()
        for src in self.src_list:
            if src.coordinates[0] > self.fov/2:
                raise OopaoError(f'Source zenith ({src.coordinates[0]}") outside of fov!')
            src.through_atm = True
            src.optical_path.append([self.tag, self])
        # geometric baseline
        needs_geometric = (not self.angular_spectrum_propagation) or self.geometric_phase_backup or self.unwrap_diffractive_phase
        OPD_support = self.initialize_OPD_support()
        if needs_geometric:
            for i_layer in range(self.nLayer):
                tmp_layer = getattr(self, 'layer_' + str(i_layer + 1))
                OPD_support = self.fill_OPD_support(tmp_layer, OPD_support, i_layer)
        # diffractive propagation
        if self.angular_spectrum_propagation:
            self.check_fresnel_sampling()
            # sort layers by altitude (top to bottom)
            layers_data = [{'layer': getattr(self, 'layer_'+str(i+1)), 'idx': i, 'alt': getattr(self, 'layer_'+str(i+1)).altitude} for i in range(self.nLayer)]
            layers_sorted = sorted(layers_data, key=lambda x: x['alt'], reverse=True)
            scintillation_support = self.initialize_scintillation_support()
            for k, item in enumerate(layers_sorted):
                dist = item['alt'] - layers_sorted[k+1]['alt'] if k < len(layers_sorted)-1 else item['alt']
                scintillation_support = self.fill_scintillation_support(item['layer'], scintillation_support, dist, item['idx'])
            # extract maps and apply setters
            self.set_scintillation_support(scintillation_support, OPD_support)
        else:
            # apply standard geometric maps if scintillation is disabled
            backend = xp if self.gpu_resident else np
            intensity_support = [backend.ones((self.telescope.resolution, self.telescope.resolution), dtype=self.precision()) for _ in self.src_list]
            self.set_scintillation(intensity_support)
            self.set_OPD(OPD_support)
        self.is_user_defined_opd = False
        return

    def initialize_OPD_support(self):
        OPD_support = []
        backend = xp if self.gpu_resident else np
        for i in range(len(self.src_list)):
            OPD_support.append(backend.zeros([self.telescope.resolution, self.telescope.resolution], dtype=self.precision()))
        return OPD_support

    def fill_OPD_support(self, tmp_layer, OPD_support, i_layer):
        for src in self.src_list:
            if src.altitude <= tmp_layer.altitude:
                raise OopaoError('The source altitude ('+str(src.altitude)+' m) is below or at the same altitude as the atmosphere layer ('+str(tmp_layer.altitude)+' m)')
        OPD_batch = None
        if self.parallel_sources and len(self.src_list) > 0:
            OPD_batch = self._get_OPD_batch(tmp_layer, i_layer)
        if OPD_batch is None:
            return self._fill_OPD_support_sequential(tmp_layer, OPD_support, i_layer)
        OPD_batch = _to_backend(OPD_batch, _backend_of(OPD_support[0]))
        for i_src in range(len(self.src_list)):
            OPD_support[i_src] += OPD_batch[i_src]
        return OPD_support

    def _get_taps(self, layer):
        """Interpolation taps of every source for the layer (footprint, sub-pixel shift, cone effect)."""
        backend = _backend_of(layer.OPD)
        n = self.telescope.resolution
        n_layer = layer.OPD.shape[0]
        key = (getattr(layer, '_footprint_key', None), tuple(float(src.altitude) for src in self.src_list),
               float(layer.altitude), n, n_layer, backend.__name__, np.dtype(layer.OPD.dtype).str)
        cache = getattr(layer, '_taps_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        row_taps = []
        col_taps = []
        for i_src, src in enumerate(self.src_list):
            rows, cols = layer.footprint_slices[i_src]
            magnification = 1 if src.altitude == np.inf else (src.altitude-layer.altitude)/src.altitude
            taps = shift_crop_zoom_taps(n_layer, rows, cols, layer.extra_sx[i_src], layer.extra_sy[i_src], magnification, n)
            if taps is None:
                layer._taps_cache = (key, None)
                return None
            row_taps.append(taps[0])
            col_taps.append(taps[1])
        taps = SeparableTaps(stack_taps(row_taps), stack_taps(col_taps), backend, layer.OPD.dtype)
        layer._taps_cache = (key, taps)
        return taps

    def _get_OPD_batch(self, layer, i_layer):
        """OPD of the layer seen by every source, as a (n_src, n, n) array (None if it cannot be batched)."""
        taps = self._get_taps(layer)
        if taps is None:
            return None
        OPD = taps.apply(layer.OPD)
        OPD = OPD * (self.wavelength/2/np.pi)
        OPD = OPD * np.sqrt(self.fractionalR0[i_layer])
        return OPD

    def _fill_OPD_support_sequential(self, tmp_layer, OPD_support, i_layer):
        backend = xp if self.gpu_resident else np
        for i_src in range(len(self.src_list)):
            src = self.src_list[i_src]
            rows, cols = tmp_layer.footprint_slices[i_src]
            off_axis_shift = tmp_layer.extra_sx[i_src] != 0 or tmp_layer.extra_sy[i_src] != 0
            cone_effect = src.altitude != np.inf
            if off_axis_shift or cone_effect:
                # the general sub-pixel shift and the cone-effect interpolation use CPU helpers
                _im = _to_backend(tmp_layer.OPD, np)
                if off_axis_shift:
                    _im = np.squeeze(interpolate_image(_im, 1, 1, _im.shape[0],
                                                       shift_x=tmp_layer.extra_sx[i_src], shift_y=tmp_layer.extra_sy[i_src]))
                _im = _im[rows, cols]
                if cone_effect:
                    # magnification due to cone effect
                    magnification_cone_effect = (src.altitude-tmp_layer.altitude)/src.altitude
                    _im = np.squeeze(interpolate_cube(np.atleast_3d(_im).T, 1, magnification_cone_effect, self.telescope.resolution)).T
                _im = backend.asarray(_im)
            else:
                # on-axis NGS: plain crop, on the device that holds the layer
                _im = _to_backend(tmp_layer.OPD, backend)[rows, cols]
            # (not in place: _im can be a view of the layer)
            _im = _im * (self.wavelength/2/np.pi)
            _im = _im * np.sqrt(self.fractionalR0[i_layer])
            OPD_support[i_src] += _to_backend(_im, _backend_of(OPD_support[i_src]))
        return OPD_support

    def _mask_on(self, mask, backend):
        """src.mask on `backend`, uploaded once per mask object (NumPy and CuPy arrays cannot be mixed)."""
        if np.isscalar(mask) or _backend_of(mask) is backend:
            return mask
        cached = self._mask_cache.get(id(mask))
        if cached is None or cached[0] is not mask or _backend_of(cached[1]) is not backend:
            # Telescope.relay gives each source a new mask at every propagation: keep only the masks of the
            # current sources (and the telescope pupil) so that old copies are released
            if len(self._mask_cache) >= 2*len(self.src_list) + 2:
                self._mask_cache.clear()
            cached = (mask, _to_backend(mask, backend))
            self._mask_cache[id(mask)] = cached
        return cached[1]

    def set_OPD(self, OPD_support):
        for i, src in enumerate(self.src_list):
            src.OPD_no_pupil = OPD_support[i]
            src.OPD = src.OPD_no_pupil*self._mask_on(src.mask, _backend_of(src.OPD_no_pupil))
        backend = xp if self.gpu_resident else np
        self.OPD = backend.squeeze(backend.stack([_to_backend(opd, backend) for opd in OPD_support]))
        return

    def set_scintillation(self, scintillation_support):
        for i, src in enumerate(self.src_list):
            src.scintillation_no_pupil = scintillation_support[i]
            src.scintillation = src.scintillation_no_pupil*self._mask_on(src.mask, _backend_of(src.scintillation_no_pupil))
        backend = xp if self.gpu_resident else np
        self.scintillation_map = backend.squeeze(backend.stack([_to_backend(m, backend) for m in scintillation_support]))
        return

    def initialize_scintillation_support(self):
        scintillation_support = []
        for i in range(len(self.src_list)):
            scintillation_support.append((xp if self.gpu_resident else np).ones([self.telescope.resolution, self.telescope.resolution], dtype=self.precision_complex))
        return scintillation_support

    def fill_scintillation_support(self, tmp_layer, scintillation_support, distance, i_layer):
        if hasattr(self.telescope, 'pixelSize'):
            pxl_scale = self.telescope.pixelSize
        else:
            pxl_scale = self.telescope.D / self.telescope.resolution
        # extract the geometric OPD for this layer only
        # (on the working backend: GPU with GPU residency)
        layer_opd = self.fill_OPD_support(tmp_layer, self.initialize_OPD_support(), i_layer)
        if self.parallel_sources and len(self.src_list) > 1:
            return self._fill_scintillation_support_batch(scintillation_support, layer_opd, distance, pxl_scale)
        for i_src, src in enumerate(self.src_list):
            wvl = src.wavelength
            backend = _backend_of(scintillation_support[i_src])
            # apply phase screen
            phi = _to_backend(layer_opd[i_src], backend) * (2 * np.pi / wvl)
            scintillation_support[i_src] = scintillation_support[i_src] * backend.exp(1j * phi)
            # propagate using angular spectrum
            if distance > 1e-6:
                scintillation_support[i_src] = self.ASM(
                    scintillation_support[i_src], wvl, pxl_scale, pxl_scale, distance)
        return scintillation_support

    def _fill_scintillation_support_batch(self, scintillation_support, layer_opd, distance, pxl_scale):
        backend = _backend_of(scintillation_support[0])
        fields = backend.stack([_to_backend(f, backend) for f in scintillation_support])
        opd = backend.stack([_to_backend(o, backend) for o in layer_opd])
        wavenumber = backend.asarray([2*np.pi/src.wavelength for src in self.src_list], dtype=opd.dtype)
        # apply phase screen
        fields = fields * backend.exp(1j * (opd * wavenumber[:, None, None]))
        # propagate using angular spectrum
        if distance > 1e-6:
            fields = self._ASM_batch(fields, [src.wavelength for src in self.src_list], pxl_scale, pxl_scale, distance)
        return list(fields)

    def _ASM_batch(self, fields, wavelengths, input_pitch, output_pitch, distance):
        """Angular-spectrum propagation of a (n_src, N, N) stack of fields (one wavelength per field)."""
        if distance == 0:
            return fields
        backend = _backend_of(fields)
        if backend is np:
            fft, fft_kw = scipy.fft, {'workers': self.fft_workers}
        else:
            fft, fft_kw = backend.fft, {}
        phase_1, phase_2, phase_3, m = self._asm_kernels_batch(backend, fields.shape[-1], fields.real.dtype,
                                                               tuple(wavelengths), input_pitch, output_pitch, distance)
        axes = (-2, -1)
        field_freq = fft.fft2(backend.fft.ifftshift(fields * phase_1, axes=axes), axes=axes, **fft_kw)
        field_out = backend.fft.fftshift(fft.ifft2(field_freq * phase_2, axes=axes, **fft_kw), axes=axes)
        return field_out * phase_3 / m

    def _asm_kernels_batch(self, backend, N, dtype, wavelengths, input_pitch, output_pitch, distance):
        """ASM kernels of every source, stacked along the first axis (shared if one wavelength)."""
        key = (backend.__name__, N, np.dtype(dtype).str, wavelengths, input_pitch, output_pitch, distance)
        kernels = self._asm_batch_cache.get(key)
        if kernels is not None:
            return kernels
        if len(self._asm_batch_cache) >= self.nLayer + 1:
            self._asm_batch_cache.clear()
        kernels_src = [self._asm_kernels(backend, N, dtype, wvl, input_pitch, output_pitch, distance) for wvl in wavelengths]
        if len(set(wavelengths)) == 1:
            kernels = kernels_src[0]
        else:
            def stack(items):
                if all(np.isscalar(item) for item in items):
                    return backend.asarray(items)[:, None, None] if len(set(items)) > 1 else items[0]
                return backend.stack([item * backend.ones((N, N), dtype=dtype) for item in items])
            kernels = tuple(stack([k[i] for k in kernels_src]) for i in range(4))
        self._asm_batch_cache[key] = kernels
        return kernels

    def set_scintillation_support(self, scintillation_support, OPD_support):
        intensity_support = []
        for i, src in enumerate(self.src_list):
            E_field = scintillation_support[i]
            wvl = src.wavelength
            backend = _backend_of(E_field)
            # Extract intensity
            intensity = backend.abs(E_field)**2
            intensity_support.append(intensity)
            # --- Phase Extraction Routing ---
            if self.geometric_phase_backup:
                # Case 1: Pure geometric OPD (OPD_support is already populated)
                pass
            elif self.unwrap_diffractive_phase:
                # Case 2: Hybrid geometric guide + diffractive perturbation
                # Convert geometric OPD to phase [rad]
                phi_geo = _to_backend(OPD_support[i], backend) * (2 * np.pi / wvl)

                # Extract the wrapped diffractive residual (delta)
                delta = backend.angle(E_field * backend.exp(-1j * phi_geo))

                # Safety check: warn if the residual itself wraps inside the pupil
                pupil_mask = self._mask_on(self.telescope.pupil, backend) > 0  # Ensure boolean masking
                delta_pupil = delta[pupil_mask]

                if delta_pupil.size > 0:  # Safeguard
                    min_delta = float(delta_pupil.min())
                    max_delta = float(delta_pupil.max())
                    C = max_delta - min_delta
                    if C > 1.9 * np.pi:
                        warning(f"Diffractive residual wrapped inside the pupil! (C = {C:.2f} rad). Scintillation is too strong for perfect unwrapping.")

                # Combine guide and residual, then convert back to OPD [m]
                final_phase = phi_geo + delta
                OPD_support[i] = final_phase * (wvl / (2 * np.pi))
            else:
                # Case 3: Standard ASM phase
                final_phase = backend.angle(E_field)
                OPD_support[i] = final_phase * (wvl / (2 * np.pi))
        # Route to the final setters
        self.set_scintillation(intensity_support)
        self.set_OPD(OPD_support)
        return

    def _invert_covariance(self, M):
        if self.covariance_inversion == 'cholesky':
            potrf, potri = scipy.linalg.lapack.get_lapack_funcs(('potrf', 'potri'), (M,))
            factor, info = potrf(M, lower=1)
            if info == 0:
                inverse, info = potri(factor, lower=1)
            if info == 0:
                # potri only fills the lower triangle
                return np.tril(inverse) + np.tril(inverse, -1).T
            warning('The covariance matrix is not positive definite: using the SVD pseudo-inverse instead.')
        elif self.covariance_inversion != 'pinv':
            raise OopaoError("covariance_inversion must be 'cholesky' or 'pinv'")
        return np.linalg.pinv(M)

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
                self.ZZt_inv = self._invert_covariance(self.ZZt)
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
                    self.ZZt_inv = self._invert_covariance(self.ZZt)
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
            # re-seed the boiling noise stream so a regenerated screen boils reproducibly
            tmp_layer.boiling_seed = 100000 + seed + i_layer * 100000
            if self.compute_covariance:
                # same as initializeAtmosphere's reset branch above
                self._reset_map_shift(tmp_layer)
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
        print('Seeing \t' + str(np.round(seeingArcsec_wvl, 2)) + str('\t ["]'))
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
            layer_index = list(np.arange(self.nLayer))
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
            gs = gridspec.GridSpec(1, n_sp+1, height_ratios=[1], width_ratios=np.ones(n_sp+1), hspace=0.5, wspace=0.5)
        else:
            gs = gridspec.GridSpec(1, n_sp, height_ratios=np.ones(1), width_ratios=np.ones(n_sp), hspace=0.25, wspace=0.25)

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
            ax.imshow(_to_backend(tmp_layer.OPD, np), extent=[-tmp_layer.D/2, tmp_layer.D/2, -tmp_layer.D/2, tmp_layer.D/2])
            center = tmp_layer.D/2
            [x_tel, y_tel] = pol2cart(tmp_layer.D_fov/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
            cm = plt.get_cmap('gist_rainbow')
            col = []
            for i_source in range(len(list_src)):
                col.append(cm(1.*i_source/len(list_src)))
                [x_c, y_c] = pol2cart(self.telescope.D/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
                h = list_src[i_source].altitude-tmp_layer.altitude
                if np.isinf(h):
                    r = self.telescope.D/2
                else:
                    r = (h)/list_src[i_source].altitude*self.telescope.D/2
                [x_cone, y_cone] = pol2cart(
                    r, np.linspace(0, 2*np.pi, 100, endpoint=True))
                if list_src[i_source].chromatic_shift is not None:
                    if len(list_src[i_source].chromatic_shift) == self.nLayer:
                        chromatic_shift = list_src[i_source].chromatic_shift[i_l]
                    else:
                        raise OopaoError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')
                else:
                    chromatic_shift = 0
                [x_z, y_z] = pol2cart(tmp_layer.altitude*np.tan((list_src[i_source].coordinates[0] + chromatic_shift)/self.rad2arcsec), -np.deg2rad(list_src[i_source].coordinates[1]))
                center = 0
                [x_c, y_c] = pol2cart(tmp_layer.D_fov/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
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

    def update_rytov_variance(self):
        if not hasattr(self, 'nLayer') or not hasattr(self, '_fractionalR0') or not hasattr(self, 'altitude'):
            return
        if not hasattr(self, 'rytov_wvl'):
            self.rytov_wvl = 500e-9
        k_target = 2 * np.pi / self.rytov_wvl
        r0_target_wvl = self.r0 * (self.rytov_wvl / self.wavelength)**(6/5)
        total_cn2_integral = (r0_target_wvl**(-5/3)) / (0.423 * k_target**2)
        current_rytov = 0.0
        for i in range(self.nLayer):
            if self.altitude[i] > 1e-3:
                current_rytov += (total_cn2_integral * self.fractionalR0[i]) * (self.altitude[i]**(5/6))
        self._rytov_var = current_rytov * 2.2524 * (k_target**(7/6))
        if self.angular_spectrum_propagation:
            if self._rytov_var > 0.3 and not self._saturation_warned:
                warning(f"Atmospheric conditions degraded. Rytov variance ({self._rytov_var:.2f}) exceeds 0.3 limit! "
                        "Entering saturation regime.")
                self._saturation_warned = True
            elif self._rytov_var <= 0.3:
                self._saturation_warned = False

    def _asm_kernels(self, backend, N, dtype, wavelength, input_pitch, output_pitch, distance):
        """Chirps and transfer kernel of the angular-spectrum propagation, computed once per configuration."""
        key = (backend.__name__, N, np.dtype(dtype).str, wavelength, input_pitch, output_pitch, distance)
        kernels = self._asm_cache.get(key)
        if kernels is not None:
            return kernels
        # one kernel set per (layer distance, wavelength) is in use at a time: when the configuration changes
        # (elevation, sources, telescope...) the kernels of the previous one are released
        if len(self._asm_cache) >= (self.nLayer + 1) * max(1, len(self.src_list)):
            self._asm_cache.clear()
        k = 2 * np.pi / wavelength
        # spatial frequency grids
        delta_f = 1.0 / (N * input_pitch)
        vals = backend.arange(-N/2, N/2, dtype=dtype) * delta_f
        fx, fy = backend.meshgrid(vals, vals)
        f_sq = fx**2 + fy**2
        # spatial grids
        vals_r = backend.arange(-N/2, N/2, dtype=dtype) * input_pitch
        x, y = backend.meshgrid(vals_r, vals_r)
        r_sq = x**2 + y**2
        m = output_pitch / input_pitch
        # input chirp
        if m != 1.0:
            phase_1 = backend.exp(1j * k/2 * (1-m)/distance * r_sq)
        else:
            phase_1 = 1.0
        # transfer kernel (stored ifftshift-ed: multiplying the unshifted spectrum by it is the same as
        # shifting, multiplying by the centred kernel and shifting back)
        phase_2 = backend.fft.ifftshift(backend.exp(-1j * np.pi * wavelength * distance / m * f_sq))
        # output chirp
        if m != 1.0:
            vals_out = backend.arange(-N/2, N/2, dtype=dtype) * output_pitch
            x_out, y_out = backend.meshgrid(vals_out, vals_out)
            r_out_sq = x_out**2 + y_out**2
            phase_3 = backend.exp(1j * k/2 * (m-1)/(m*distance) * r_out_sq)
        else:
            phase_3 = 1.0
        kernels = (phase_1, phase_2, phase_3, m)
        self._asm_cache[key] = kernels
        return kernels

    def ASM(self, input_field, wavelength, input_pitch, output_pitch, distance):
        """Angular-spectrum propagation over `distance`, on the backend of input_field (NumPy or CuPy)."""
        if distance == 0:
            return input_field
        backend = _backend_of(input_field)
        N = input_field.shape[0]
        phase_1, phase_2, phase_3, m = self._asm_kernels(backend, N, input_field.real.dtype,
                                                         wavelength, input_pitch, output_pitch, distance)
        # fft operations
        field_freq = backend.fft.fft2(backend.fft.ifftshift(input_field * phase_1))
        field_out = backend.fft.fftshift(backend.fft.ifft2(field_freq * phase_2))
        return field_out * phase_3 / m

    def check_fresnel_sampling(self):
        """
        Internal verification of the telescope grid for Fresnel propagation.
        Issues warnings if the grid is prone to aliasing or violates ASM limits.
        """
        if not self.angular_spectrum_propagation or self.sampling_checked:
            return
        if hasattr(self.telescope, 'initial_D'):
            D_tel_phys = self.telescope.initial_D
        else:
            D_tel_phys = self.telescope.D
        N_current = self.telescope.resolution
        if hasattr(self.telescope, 'pixelSize'):
            delta = self.telescope.pixelSize
        else:
            delta = self.telescope.D / N_current
        alts = sorted(self.altitude + [0.0], reverse=True)
        max_step = max([alts[i] - alts[i+1] for i in range(len(alts)-1)]) if len(alts) > 1 else 0
        z_max = max(self.altitude) if self.altitude else 0
        for src in self.src_list:
            wvl = src.wavelength
            r0_wvl = self._r0 * (wvl / self.wavelength)**(6/5)
            # Coy's spread (bilateral support)
            D_turb = 4.0 * (wvl * z_max) / r0_wvl
            D_total = D_tel_phys + 2.0 * D_turb
            delta_req = min(r0_wvl / 4.0, (wvl * z_max) / D_total)
            # Strict ASM anti-aliasing (N >= D_tot/delta + lambda*z/delta^2)
            N_min_physique = (D_total / delta) + (wvl * z_max) / (delta**2)
            z_asm = (N_current * delta**2) / wvl
            if delta > delta_req:
                warning(f"Fresnel [wvl={wvl*1e9:.0f}nm] - Pixel scale ({delta*1000:.1f} mm) too large! Limit is {delta_req*1000:.1f} mm.")
            if N_current < N_min_physique:
                N_min_int = int(np.ceil(N_min_physique))
                warning(f"Fresnel Aliasing [wvl={wvl*1e9:.0f}nm] - Grid N={N_current} too small (Physics requires N >= {N_min_int}).")
                warning(f"-> Action required: Pad your telescope to the next multiple of your WFS res_factor >= {N_min_int}.")
            if max_step > z_asm:
                warning(f"Fresnel ASM Limit [wvl={wvl*1e9:.0f}nm] - Distance between layers ({max_step/1000:.1f} km) > limit ({z_asm/1000:.1f} km).")
        self.sampling_checked = True

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, val):
        self.sampling_checked = False
        self._r0 = val
        el_rad = np.radians(self.elevation)
        self.r0_zenith = val * (np.sin(el_rad))**(-3/5)  # r0 at zenith
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
                    xp_ = _backend_of(tmp_layer.A)
                    BBt = tmp_layer.XXt_r0 - self.convert_for_numpy(xp_.matmul(tmp_layer.A, xp_.asarray(tmp_layer.ZXt_r0)))
                    tmp_layer.B = self.convert_for_gpu(np.linalg.cholesky(BBt).astype(self.precision()))
        self.update_rytov_variance()

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
                self.V0 = (np.sum(np.asarray(self.fractionalR0) * np.asarray(self.windSpeed)**(5/3)))**(3/5)  # computation of equivalent wind speed, Roddier 1982
                self.tau0 = 0.31 * self.r0 / self.V0  # Coherence time of atmosphere, Roddier 1981
                for i_layer in range(self.nLayer):
                    tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                    tmp_layer.windSpeed = val[i_layer]
                    tmp_layer.vY = tmp_layer.windSpeed * np.cos(np.deg2rad(tmp_layer.direction))
                    tmp_layer.vX = tmp_layer.windSpeed * np.sin(np.deg2rad(tmp_layer.direction))
                    ps_turb_x = tmp_layer.vX*self.telescope.samplingTime
                    ps_turb_y = tmp_layer.vY*self.telescope.samplingTime
                    # (before the first update the ratio does not exist yet: updateLayer computes it)
                    if hasattr(tmp_layer, 'ratio'):
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
                    tmp_layer.vY = tmp_layer.windSpeed * np.cos(np.deg2rad(tmp_layer.direction))
                    tmp_layer.vX = tmp_layer.windSpeed * np.sin(np.deg2rad(tmp_layer.direction))
                    ps_turb_x = tmp_layer.vX*self.telescope.samplingTime
                    ps_turb_y = tmp_layer.vY*self.telescope.samplingTime
                    # (before the first update the ratio does not exist yet: updateLayer computes it)
                    if hasattr(tmp_layer, 'ratio'):
                        tmp_layer.ratio[0] = ps_turb_x/tmp_layer.pixel_size
                        tmp_layer.ratio[1] = ps_turb_y/tmp_layer.pixel_size
                    setattr(self, 'layer_'+str(i_layer+1), tmp_layer)

    @property
    def t_boiling(self):
        return self._t_boiling

    @t_boiling.setter
    def t_boiling(self, val):
        self._t_boiling = self.get_t_boiling(val)
        if self.hasNotBeenInitialized is False:
            print('Updating the boiling time-constant(s)...')
            for i_layer in range(self.nLayer):
                tmp_layer = getattr(self, 'layer_'+str(i_layer+1))
                tmp_layer.t_boiling = self._t_boiling[i_layer]
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
                self.V0 = (np.sum(np.asarray(self.fractionalR0) * np.asarray(self.windSpeed)**(5/3)))**(3/5)  # computation of equivalent wind speed, Roddier 1982
                self.tau0 = 0.31 * self.r0 / self.V0  # Coherence time of atmosphere, Roddier 1981
        self.update_rytov_variance()

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, val):
        self.sampling_checked = True
        val = float(val)
        if val < 5.0:
            warning("Very low elevation (< 5 deg). Clamping to 5 deg.")
            val = 5.0
        if hasattr(self, '_elevation') and val == self._elevation:
            return  # Do nothing if the exact same elevation is requested again
        self._elevation = val
        el_rad = np.radians(val)
        # airmass = 1.0 / np.sin(el_rad)  # Airmass calculation
        print(f"--- Atmosphere Configuration : Elev={val:.1f}° ---")
        self.altitude = [h_z / np.sin(el_rad) for h_z in self.altitude_zenith]  # Update effective distance of layers
        self._r0 = self.r0_zenith * (np.sin(el_rad))**(3/5)  # Update effective r0 (3/5 power law)
        print(f" -> New r0 : {self._r0:.4f} m (Zenith reference: {self.r0_zenith:.4f} m)")
        if self.hasNotBeenInitialized is False:  # Re-initializationof of the atmosphere
            self.hasNotBeenInitialized = True
            if hasattr(self, 'telescope') and self.telescope is not None:
                self.initializeAtmosphere(self.telescope, compute_covariance=self.compute_covariance)
            else:
                raise OopaoError("Warning: Atmosphere not yet linked to a telescope. Call initializeAtmosphere() manually.")
        self.update_rytov_variance()

    @property
    def rytov_var(self):
        if not self.angular_spectrum_propagation:
            return 0.0
        if not hasattr(self, '_rytov_var'):
            self.update_rytov_variance()
        return self._rytov_var

    @rytov_var.setter
    def rytov_var(self, val):
        if not self.angular_spectrum_propagation:
            warning("Cannot set Rytov variance because atm.angular_spectrum_propagation is False.")
            return
        val = float(val)
        if val <= 0:
            return
        saturation_threshold = 0.3
        if val > saturation_threshold:
            warning(f"Target Rytov variance ({val:.2f}) exceeds the weak fluctuation limit ({saturation_threshold}). "
                    "The simulation is entering the moderate/strong scintillation regime (saturation). "
                    "Phase unwrapping and scaling behaviors may become unstable.")
        self.update_rytov_variance()
        current = self._rytov_var
        if current > 0:
            ratio = val / current
            print(f"--- Scaling Rytov variance to {val:.4f} (Ratio: {ratio:.3f}) ---")
            new_r0 = self.r0 * (ratio**(-3/5))
            self.r0 = new_r0
            scale_factor = float(np.sqrt(ratio))

            for i in range(1, self.nLayer + 1):
                layer_name = f'layer_{i}'
                if hasattr(self, layer_name):
                    layer_obj = getattr(self, layer_name)
                    if hasattr(layer_obj, 'OPD') and layer_obj.OPD is not None:
                        layer_obj.OPD *= scale_factor
            self.update_rytov_variance()
        else:
            warning("Cannot scale Rytov (all layers are at 0m).")

        if self.angular_spectrum_propagation:
            if self._rytov_var > 0.3 and not self._saturation_warned:
                warning(f"Atmospheric conditions degraded. Rytov variance ({self._rytov_var:.2f}) exceeds 0.3 limit! "
                        "Entering saturation regime.")
                self._saturation_warned = True
            elif self._rytov_var <= 0.3:
                self._saturation_warned = False

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        # self.prop['t_boiling'] = f"{'Boiling [s]':<16s}|{tb_str:^10s}"
        self.prop['parameters'] = f"{'Layer':^7s}|{'Direction':^11s}|{'Speed':^7s}|{'Altitude':^10s}|{'Frac Cn²':^10s}|{'Diameter':^10s}|{'Evolution':^20s}|"
        self.prop['units'] = f"{'':^7s}|{'[°]':^11s}|{'[m/s]':^7s}|{'[m]':^10s}|{'[%]':^10s}|{'[m]':^10s}|{'':^20s}|"
        for i in range(self.nLayer):
            if self.t_boiling[i] is None:
                tb_str = f"{'Frozen Flow':^20s}"
            else:
                tb_str = f"{f'Boiling: t = {self._t_boiling[i]:.4g} s':^20s}"
            if i % 2 == 0:
                self.prop['layer_%02d' % i] = f"\033[00m{i+1:^7d}|{self.windDirection[i]:^11.0f}|{self.windSpeed[i]:^7.1f}|{self.altitude[i]:^10.0e}|{self.fractionalR0[i]*100:^10.0f}|{getattr(self, 'layer_'+str(i+1)).D:^10.3f}|{tb_str}|"
            else:
                self.prop['layer_%02d' % i] = f"\033[47m{i+1:^7d}|{self.windDirection[i]:^11.0f}|{self.windSpeed[i]:^7.1f}|{self.altitude[i]:^10.0e}|{self.fractionalR0[i]*100:^10.0f}|{getattr(self, 'layer_'+str(i+1)).D:^10.3f}|{tb_str}|"
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
        title = f'\n{" Atmosphere @ Elevation = " + str(self._elevation) + "°":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table