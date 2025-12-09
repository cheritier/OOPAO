# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import time
import numpy as np
from .Detector import Detector
from .tools.tools import bin_ndarray, gaussian_2D, warning, OopaoError
from joblib import Parallel, delayed
import scipy as sp
import sys
import matplotlib.pyplot as plt
try:
    import cupy as xp
    global_gpu_flag = True
    xp = np  # for now
except ImportError or ModuleNotFoundError:
    xp = np

class ShackHartmann:
    def __init__(self, nSubap: float,
                 telescope,
                 lightRatio: float,
                 threshold_cog: float = 0.01,
                 is_geometric: bool = False,
                 binning_factor: int = 1,
                 pixel_scale: float = None,
                 threshold_convolution: float = 0.05,
                 shannon_sampling: bool = False,
                 n_pixel_per_subaperture: int = None,
                 half_pixel_shift: bool = False,
                 padding_extension_factor: int = None):
        """SHACK-HARTMANN
        A Shack Hartmann object consists in defining a 2D grd of lenslet arrays located in the pupil plane of the telescope to estimate the local tip/tilt seen by each lenslet.
        By default the Shack Hartmann detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)
        It requires the following parameters:

        Parameters
        ----------
        nSubap : float
            The number of subapertures (micro-lenses) along the diameter defined by the telescope.pupil.
        telescope : TYPE
            The telescope object to which the Shack Hartmann is associated.
            This object carries the phase, flux and pupil information.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        threshold_cog : float, optional
            Threshold (with respect to the maximum value of the image)
            to apply to compute the center of gravity of the spots.
            The default is 0.01.
        is_geometric : bool, optional
            Flag to enable the geometric WFS.
            - If True, enables the geometric Shack Hartmann (direct measurement of gradient).
               The signal units are in units of pixel-scale (see wfs.pixel_scale).
            - If False, the diffractive computation is considered.
                The signal units are in units of pixel-scale (see wfs.pixel_scale).
            The default is False.
        binning_factor : int, optional
            Binning factor of the detector.
            The default is 1.
        pixel_scale : float, optional
                Pixel scale in [arcsec] requested by the user. The spots will be either zero-padded and binned accordingly to provide the closest pixel-scale available.
                The default is the shannon sampling of the spots.
        threshold_convolution : float, optional
            Threshold considered to force the gaussian spots (elungated spots) to go to zero on the edges (to speed up the concvolution operations).
            The default is 0.05.
        shannon_sampling : bool, optional
            This parameter is only used if the pixel_scale parameter is set to None.
            If True, the lenslet array spots are sampled at the same sampling as the FFT (2 pix per FWHM).
            If False, the sampling is 1 pix per FWHM (default).
            The default is False.
        unit_P2V : bool, optional
                If True, the slopes units are calibrated using a Tip/Tilt normalized to 2 Pi peak-to-valley.
                If False, the slopes units are calibrated using a Tip/Tilt normalized to 1 in the pupil (Default). In that case the slopes are expressed in [rad].
                The default is False.
        n_pixel_per_subaperture : int, optional
                Number of pixel per subaperture of size defined by the pixel_scale parameter.
                The maximum FoV in pixel is driven by the resolution of the telescope and is stored in the n_pix_subap_init property.
                If n_pixel_per_subaperture < n_pix_subap_init, the subapertures are cropped to the right number of pixels
                If n_pixel_per_subaperture > n_pix_subap_init, the subapertures are zero-padded to provide the right number of pixels.
                    A warning is displayed as only the FoV defined by n_pix_subap_init contains signal and wrapping effects may occur.
                The default is None and corresponds to n_pixel_per_subaperture = n_pixel_per_subap_init.
        pixel_scale : float, optional
                sampling of the detector pixels in [arcsec]. This parameter overwrites the shannon_sampling parameter.
                The effective pixel_scale value is obtained by taking the closest value after considering the FFT sampling
                and the possible binning factor:
                    - If the pixel-scale is too large for the lenslet FoV, the diffractive spots are zero-padded.
                    A warning is displayed in this situation.
        half_pixel_shift : bool,optional
                half pixel shift (in pixel scale unit) of the SH spots in the focal plane to center the SH spots on 1 or 4 pixels.
                The default is False and corresponds to a spot centered on 4 pixels.
            
        padding_extension_factor : int, optional
            DEPRECATED

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** PROPAGATING THE LIGHT TO THE SH OBJECT **************************
        The light can be propagated from a telescope object tel through the Shack Hartmann object wfs using the * operator:
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the Shack Hartmann detector (phase and flux)
            _ binning of the SH signals
            _ addition of eventual photon noise and readout noise
            _ computation of the Shack Hartmann signals

        ************************** WEIGHTED CENTER OF GRAVITY **************************
        The weighted center of gravity can be set using the wfs.set_weighted_centroiding_map method.
        Once set, the units of the shack-hartman are re-calibrated using the wfs.set_slopes_units method.

        ************************** PROPERTIES **************************

        The main properties of a Telescope object are listed here:
        _ wfs.signal                     : signal measured by the Shack Hartmann
        _ wfs.signal_2D                  : 2D map of the signal measured by the Shack Hartmann
        _ wfs.fov_lenslet_arcsec         : Field of View of the subapertures in arcsec
        _ wfs.fov_pixel_binned_arcsec    : Field of View of the pixel in arcsec
        _ wfs.slopes_units               : normalisation factor to provide cog in pixel-scale units.
                                           For example, a wfs.signal value of 1 corresponds to a shift of 1 pixel at the lenslet level.
                                           For the Geometric WFS, the units are calibrated to provide the same values as for a diffractive WFS
                                           As such, they also depends on the wfs.pixel_scale value.

        The main properties of the object can be displayed using :
            wfs.print_properties()

        the following properties can be updated on the fly:
            _ wfs.is_geometric          : switch between diffractive and geometric shackHartmann
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value

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
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128

        self.tag = 'shackHartmann'
        self.telescope = telescope
        if self.telescope.src is None:
            raise OopaoError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        if telescope.src.type == 'LGS':
            self.is_LGS = True
            self.convolution_tag = 'FFT'
        else:
            self.is_LGS = False
        # save wavelength used for the calibration of the Pyramid to avoid conflicts
        self.wavelength_calibration = self.telescope.src.wavelength
        # flag to switch between geometric and diffractive ShackHartmann
        self.is_geometric = is_geometric
        # Number of subaperture
        self.nSubap = nSubap
        # Size of the subaperture in [m]
        self.d_subap = telescope.D/self.nSubap
        # light ratio used to select the valid pixels
        self.lightRatio = lightRatio
        # binning factor of the WFS detector
        self.binning_factor = binning_factor
        # initial zero-padding for the ShackHartmann spot computation
        self.zero_padding = 2
        # flag to shift the SH spots of half a pixel
        self.half_pixel_shift = half_pixel_shift
        # threshold value to speed up the LGS spots convolution
        self.threshold_convolution = threshold_convolution
        # threshold to compute the center of gravity
        self.threshold_cog = threshold_cog
        # flag to consider shannon-sampling or shannon-sampling/2 for the spot sampling (over-written by the pixel_scale)
        self.shannon_sampling = shannon_sampling
        # property to be used for the weighted center of gravity
        self.weighting_map = 1
        # conversion of the radians in arcsec
        self.rad2arcsec = 1 / (np.pi / 180 / 3600)
        # original pixel scale assuming a zeropadding factor of 2
        self.pixel_scale_init = self.rad2arcsec*self.telescope.src.wavelength/self.d_subap / self.zero_padding
        if padding_extension_factor is not None:
            raise DeprecationWarning('The use of the the padding_extension_factor parameter is deprecated. ' +
                                     'The pixel scale and FoV of the subaperture can be set using the pixel_scale and the n_pixel_per_subaperture parameters.')
        if pixel_scale is None:
            print('No user-input pixel scale - using shannon_sampling input value:'+str((1+self.shannon_sampling))+' pixel(s) per spot FWHM')
            self.pixel_scale = self.pixel_scale_init*(2-self.shannon_sampling)
            self.binning_pixel_scale = int(2-self.shannon_sampling)
        else:
            print('User input pixel scale - The shannon_sampling input value is ignored')
            # find the closest value of pixel-scale possible considering the sampling of the telescope
            binning_pixel_scale = pixel_scale/self.pixel_scale_init
            if binning_pixel_scale < 0.95:
                while binning_pixel_scale < 0.95:
                    self.zero_padding += 1
                    self.pixel_scale_init = self.rad2arcsec*self.telescope.src.wavelength/self.d_subap / self.zero_padding
                    binning_pixel_scale = pixel_scale/self.pixel_scale_init

            binning_pixel_scale = [np.floor(pixel_scale/self.pixel_scale_init), np.ceil(pixel_scale/self.pixel_scale_init)]
            ind_ = (np.argmin([np.abs(binning_pixel_scale[0]*self.pixel_scale_init - pixel_scale), np.abs(binning_pixel_scale[1]*self.pixel_scale_init-pixel_scale)]))
            binning_pixel_scale = binning_pixel_scale[ind_]

            self.pixel_scale = self.pixel_scale_init*binning_pixel_scale
            self.binning_pixel_scale = int(binning_pixel_scale)
            if pixel_scale != self.pixel_scale:
                warning('The requested pixel-scale is: '+str(pixel_scale)+' arcsec\n' +
                        'The effective pixel-scale is: '+str(self.pixel_scale)+' arcsec')

        # different resolutions needed:
        # 1) The number of pixel per subaperture
        self.n_pix_subap_init = self.telescope.resolution // self.nSubap
        if n_pixel_per_subaperture is None:
            self.n_pix_subap = self.n_pix_subap_init
        else:
            if n_pixel_per_subaperture % 2 != 0:
                raise OopaoError('n_pixel_per_subaperture can only be an even number.')
            self.n_pix_subap = n_pixel_per_subaperture
            if n_pixel_per_subaperture*self.pixel_scale > self.n_pix_subap_init*self.pixel_scale_init*self.zero_padding and self.is_LGS is False:
                warning('The requested number of pixel per subaperture is too large!\n' +
                        'The SH spots will be zero-padded to provide the desired number but will not contain any signal.\n' +
                        'The FoV is limited to +/- '+str(self.n_pix_subap_init*self.pixel_scale_init/2)+' arcsec. Any signal further will be wrapping.\n' +
                        'To avoid this effect, use a larger resolution for the telescope to increase the technical FoV.')
        # 2) The number of pixel required to extend the subaperture to fit the full LGS spots
        self.extra_pixel = (self.n_pix_subap-self.n_pix_subap_init)//2
        # 3) The number of pixel per subaperture for the initial optical propagation
        self.n_pix_lenslet_init = self.n_pix_subap_init*self.zero_padding
        # 4) The number of pixel per subaperture associated to the extended fov
        self.n_pix_lenslet = self.n_pix_subap*self.zero_padding
        # maximum field of view for off-axis sources when propagating asterism
        self.max_fov_arcsec = self.pixel_scale*self.n_pix_subap/2
        # associated centers for each case
        self.center = self.n_pix_lenslet//2
        self.center_init = self.n_pix_lenslet_init//2
        self.outerMask = np.ones(
            [self.n_pix_subap_init*self.zero_padding, self.n_pix_subap_init*self.zero_padding])
        self.outerMask[1:-1, 1:-1] = 0
        # Compute camera frame in case of multiple measurements
        self.get_raw_data_multi = False

        # WFS detector object
        self.cam = Detector(round(nSubap*self.n_pix_subap))
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0

        # joblib parameter
        self.nJobs = 1
        self.joblib_prefer = 'processes'

        # camera frame
        self.raw_data = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,
                                 self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)

        self.cube_flux = np.zeros([self.nSubap**2,
                                   self.n_pix_subap_init,
                                   self.n_pix_subap_init], dtype=(float))
        self.index_x = []
        self.index_y = []

        # phasor to center spots in the center of the lenslets
        [xx, yy] = np.meshgrid(np.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init),
                               np.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init))
        self.phasor = np.exp(-(1j*np.pi*(self.n_pix_lenslet_init+1+(self.pixel_scale/self.pixel_scale_init)*self.half_pixel_shift) /
                             self.n_pix_lenslet_init)*(xx+yy))
        self.phasor_tiled = np.moveaxis(
            np.tile(self.phasor[:, :, None], self.nSubap**2), 2, 0)
        # get the flux per subaperture
        self.initialize_flux()
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
        self.current_nPhoton = self.telescope.src.nPhoton
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)
        print('Valid subapertures selection based on flux considerations')
        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture, [self.nSubap, self.nSubap])
        self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap, self.nSubap])
        self.valid_subapertures_1D = np.reshape(self.valid_subapertures, [self.nSubap**2])
        [self.validLenslets_x, self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures, self.valid_subapertures))

        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        self.nSignal = 2*self.nValidSubaperture
        if self.is_LGS:
            self.shift_x_buffer, self.shift_y_buffer, self.spot_kernel_elongation_fft, self.spot_kernel_elongation = self.get_convolution_spot(compute_fft_kernel=True)
        # WFS initialization
        self.initialize_wfs()

    def initialize_wfs(self):
        # storage of the current telescope OPD and WFS detector noise
        tmp_opd = self.telescope.OPD.copy()
        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0
        self.isInitialized = False
        self.isCalibrated = False
        # reference signal initialization
        self.SX = np.zeros([self.nSubap, self.nSubap])
        self.SY = np.zeros([self.nSubap, self.nSubap])
        self.reference_slopes_maps = np.zeros([self.nSubap*2, self.nSubap])
        # slopes units initialization
        self.slopes_units = 1
        print('Acquiring reference slopes...', end=' ')
        self.telescope.resetOPD()
        self.wfs_measure(self.telescope.src.phase)
        self.reference_slopes_maps = np.copy(self.signal_2D)
        self.isInitialized = True
        print('Done!')
        # if self.is_geometric:
        #     self.slopes_units = self.nSubap*np.pi/2/self.zero_padding  # units in self.pixel_scale unit
        # else:
        #     self.slopes_units = self.pixel_scale_init/self.pixel_scale
        self.set_slopes_units()
        # re-apply telescope OPD and WFS detector noise
        self.cam.photonNoise = photonNoise
        self.cam.readoutNoise = readoutNoise
        self.telescope.OPD = tmp_opd
        self.print_properties()

    def set_weighted_centroiding_map(self, is_lgs: bool, is_gaussian: bool, fwhm_factor):
        """
        This function allows to compute a 2D map to perform a weighted center of gravity.
        The ShackHartmann property weighting_map is set after the execution of the function
        Parameters
        ----------
        is_lgs : bool
            Flag to make use of the LGS spots properties to compute the weighting map.
            If set to True, the map computed is based on the LGS spot elongation profile and
            can be based on the Na Profile or assymetric 2D gaussian (see is_gaussian parameter).
            If set to False, the map computed is a gaussian function that is the same for each lenslet
        is_gaussian : bool
            Flag used when the is_lgs flag is set to True.
            If set to True, a gaussian map is computed for each lenslet based on the elongation of each LGS spot.
            If set to False, a map based on the LGS Na Profile is computed for each lenslet.
            In both cases, the fwhm_factor is used as a multiplicative factor for the fwhm of the weighting map.
        fwhm_factor : scalar/list
            fwhm of the weighting map.
            if is_lgs is True:
                - fwhm_factor must be a scalar to be used as a multiplicative factor based
                    on the LGS spots elongation
            if is_lgs is False:
                - fwhm_factor can be a scalar (symmetric gaussian) or a list (assymetric)
                    expressed in fraction of pixel scale
        Returns
        -------
        None.

        """
        if is_lgs:
            print('The weighting map is based on the LGS spots kernel')
            _, _, _, weighting_map = self.get_convolution_spot(fwhm_factor=fwhm_factor, compute_fft_kernel=False, is_gaussian=is_gaussian)
            # bin the resulting image to the right pixel scale
            weighting_map = bin_ndarray(weighting_map,
                                        [weighting_map.shape[0],
                                         weighting_map.shape[1]//self.binning_pixel_scale,
                                         weighting_map.shape[1]//self.binning_pixel_scale], operation='sum')

            # crop the resulting spots to the right number of pixels
            n_crop = (weighting_map.shape[1] - self.n_pix_subap)//2
            if n_crop > 0:
                weighting_map = weighting_map[:, n_crop:-n_crop, n_crop:-n_crop]
        else:
            if np.isscalar(fwhm_factor):
                fwhm_factor = [fwhm_factor, fwhm_factor]
            print('The weighting map is a centerd gaussian 2D function with a FWHM of ' + str(fwhm_factor) + ' px')
            weighting_map = np.tile(gaussian_2D(resolution=self.n_pix_subap, fwhm=fwhm_factor), [self.nValidSubaperture, 1, 1])
        warning('A new weighting map is now considered.')
        self.weighting_map = weighting_map
        self.set_slopes_units()
        return

    def set_slopes_units(self):
        # storage of the current telescope OPD and WFS detector noise
        print('Calibrating the slopes units...', end=' ')
        tmp_opd = self.telescope.OPD.copy()
        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0
        # reference signal
        self.SX = np.zeros([self.nSubap, self.nSubap])
        self.SY = np.zeros([self.nSubap, self.nSubap])
        # flux per subaperture
        self.slopes_units = 1
        # apply a Tip/Tilt of 1 pixel in X & Y direction
        [Tip, Tilt] = np.meshgrid(np.linspace(-np.pi, np.pi, self.telescope.resolution, endpoint=False),
                                  np.linspace(-np.pi, np.pi, self.telescope.resolution, endpoint=False))
        self.telescope.OPD = ((Tip+Tilt)*self.telescope.src.wavelength/2/np.pi) \
            * self.pixel_scale / (self.telescope.src.wavelength / self.telescope.D) / self.rad2arcsec
        self.wfs_measure(self.telescope.src.phase)
        self.slopes_units = np.mean(self.signal)
        # re-apply the WFS detector noise and telescope OPD
        self.cam.photonNoise = photonNoise
        self.cam.readoutNoise = readoutNoise
        self.telescope.OPD = tmp_opd
        self.isCalibrated = True
        print('Done')
        return

    def centroid(self, image, threshold=0.01):

        if np.ndim(image) <= 2:
            im = np.reshape(image.copy(), (1, np.shape(image)[0], np.shape(image)[1]))
        else:
            im = np.atleast_3d(image.copy())

        im[im < (threshold*im.max())] = 0
        centroid_out = np.zeros([im.shape[0], 2])
        X_map, Y_map = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[2]))
        X_coord_map = np.atleast_3d(X_map).T
        Y_coord_map = np.atleast_3d(Y_map).T
        norma = np.sum(np.sum(im, axis=1), axis=1)
        centroid_out[:, 0] = np.sum(np.sum(im*X_coord_map, axis=1), axis=1)/norma
        centroid_out[:, 1] = np.sum(np.sum(im*Y_coord_map, axis=1), axis=1)/norma

        return centroid_out

    def initialize_flux(self, input_flux_map=None):
        if self.telescope.tag != 'asterism':
            if input_flux_map is None:
                input_flux_map = self.telescope.src.fluxMap.T
            tmp_flux_h_split = np.hsplit(input_flux_map, self.nSubap)
            self.cube_flux = np.zeros([self.nSubap**2,
                                       self.n_pix_lenslet_init,
                                       self.n_pix_lenslet_init], dtype=float)
            for i in range(self.nSubap):
                tmp_flux_v_split = np.vsplit(tmp_flux_h_split[i], self.nSubap)
                self.cube_flux[i*self.nSubap:(i+1)*self.nSubap,
                               self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init // 2,
                               self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = np.asarray(tmp_flux_v_split)
            self.photon_per_subaperture = np.apply_over_axes(
                np.sum, self.cube_flux, [1, 2])
            self.current_nPhoton = self.telescope.src.nPhoton
        return

    def get_lenslet_em_field(self, phase):
        tmp_phase_h_split = np.hsplit(phase.T, self.nSubap)
        self.cube_em = np.zeros([self.nSubap**2,
                                 self.n_pix_lenslet_init,
                                 self.n_pix_lenslet_init], dtype=complex)
        for i in range(self.nSubap):
            tmp_phase_v_split = np.vsplit(tmp_phase_h_split[i], self.nSubap)
            self.cube_em[i*self.nSubap:(i+1)*self.nSubap,
                         self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2,
                         self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = np.exp(1j*np.asarray(tmp_phase_v_split))
        self.cube_em *= np.sqrt(self.cube_flux)*self.phasor_tiled
        return self.cube_em

    def fill_raw_data(self, ind_x, ind_y, intensity, index_frame=None):
        if index_frame is None:
            self.raw_data[ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,
                          ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = intensity
        else:
            self.raw_data[index_frame,
                          ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,
                          ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = intensity

    def merge_data_cube(self, cube):
        # save current raw data
        tmp_raw_data = self.raw_data.copy()

        def joblib_fill_raw_data():
            Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.fill_raw_data)(i, j, k) for i, j, k in zip(self.index_x[self.valid_subapertures_1D],
                                                                                                               self.index_y[self.valid_subapertures_1D],
                                                                                                               cube))
            return Q
        joblib_fill_raw_data()
        output_raw_data = self.raw_data.copy()
        # re-assign raw_data
        self.raw_data = tmp_raw_data.copy()
        return output_raw_data

    def split_raw_data(self,input_frame=None):
        if input_frame is None:
            input_frame = self.cam.frame
        raw_data_h_split = np.vsplit((input_frame), self.nSubap)
        maps_intensity = np.zeros([self.nSubap**2,
                                        self.n_pix_subap,
                                        self.n_pix_subap], dtype=float)
        center = self.n_pix_subap//2
        for i in range(self.nSubap):
            raw_data_v_split = np.hsplit(raw_data_h_split[i], self.nSubap)
            maps_intensity[i*self.nSubap:(i+1)*self.nSubap,
                                center - self.n_pix_subap//self.binning_factor//2:center+self.n_pix_subap//self.binning_factor // 2,
                                center - self.n_pix_subap//self.binning_factor//2:center+self.n_pix_subap//self.binning_factor//2] = np.asarray(raw_data_v_split)
        maps_intensity = maps_intensity[self.valid_subapertures_1D, :, :]
        return maps_intensity

    def compute_raw_data_multi(self, intensity):
        self.ind_frame = np.zeros(intensity.shape[0], dtype=(int))
        index_x = np.tile(
            self.index_x[self.valid_subapertures_1D], self.phase_buffer.shape[0])
        index_y = np.tile(
            self.index_y[self.valid_subapertures_1D], self.phase_buffer.shape[0])

        for i in range(self.phase_buffer.shape[0]):
            self.ind_frame[i *
                           self.nValidSubaperture:(i+1)*self.nValidSubaperture] = i

        def joblib_fill_raw_data():
            Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.fill_raw_data)(i, j, k, l) for i, j, k, l in zip(index_x, index_y, intensity, self.ind_frame))
            return Q

        joblib_fill_raw_data()
        return

    def gradient_2D(self, arr):
        arr[~self.telescope.pupil] = np.nan

        res_x = (np.gradient(arr, axis=0, edge_order=1)/self.telescope.pixelSize) * self.telescope.pupil
        res_x = np.nan_to_num(res_x)
        res_y = (np.gradient(arr, axis=1, edge_order=1)/self.telescope.pixelSize) * self.telescope.pupil
        res_y = np.nan_to_num(res_y)

        return res_x, res_y

    def lenslet_propagation_geometric(self, arr):

        [SLx, SLy] = self.gradient_2D(arr)
        sy = bin_ndarray(ndarray=SLx, new_shape=(self.nSubap, self.nSubap), operation="mean", ignore_zeros=True)
        sx = bin_ndarray(ndarray=SLy, new_shape=(self.nSubap, self.nSubap), operation="mean", ignore_zeros=True)

        return np.concatenate((sx, sy))

    def convolve_direct(self, A_in, B_in):
        return sp.signal.convolve(A_in, B_in, mode='same', method='direct')

    def get_convolution_spot(self, fwhm_factor=1, compute_fft_kernel=False, is_gaussian=False):
        # compute the projection of the LGS on the subaperture to simulate
        #  the spots elongation using a convulotion with gaussian spot
        # coordinates of the LLT in [m] from the center (sign convention adjusted to match display position on camera)
        [X0, Y0] = [self.telescope.src.laser_coordinates[1], -self.telescope.src.laser_coordinates[0]]

        # 3D coordinates
        coordinates_3D = np.zeros([3, len(self.telescope.src.Na_profile[0, :])])
        coordinates_3D_ref = np.zeros([3, len(self.telescope.src.Na_profile[0, :])])

        # variable to store the shift due to the elongation
        delta_dx = np.zeros([2, len(self.telescope.src.Na_profile[0, :])])
        delta_dy = np.zeros([2, len(self.telescope.src.Na_profile[0, :])])

        # coordinates of the subapertures
        x_subap = np.linspace(-self.telescope.D//2, self.telescope.D//2, self.nSubap)
        y_subap = np.linspace(-self.telescope.D//2, self.telescope.D//2, self.nSubap)
        # grid of coordinates
        [X, Y] = np.meshgrid(x_subap, y_subap)
        # number of pixel
        n_pix = self.n_pix_lenslet
        # size of a pixel in m
        d_pix = (self.telescope.D/self.nSubap)/self.n_pix_lenslet_init
        # cartesian grid on whoch to compute the 2D spots
        v = np.linspace(-n_pix*d_pix/2, n_pix*d_pix/2, n_pix)
        [alpha_x, alpha_y] = np.meshgrid(v, v)
        # FWHM of gaussian converted into pixel in arcsec
        sigma_spot = fwhm_factor*self.telescope.src.FWHM_spot_up/self.pixel_scale_init
        for i in range(len(self.telescope.src.Na_profile[0, :])):
            coordinates_3D[:2, i] = (self.telescope.D/4)*([X0, Y0]/self.telescope.src.Na_profile[0, i])
            coordinates_3D[2, i] = self.telescope.D**2. / (8.*self.telescope.src.Na_profile[0, i])/(2.*np.sqrt(3.))
            coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(self.telescope.src.Na_profile[0, :])//2]
        distance_to_llt = np.sqrt((X-X0)**2 + (Y-Y0)**2)
        tmp = (np.where(distance_to_llt == np.max(distance_to_llt)))
        if len(tmp) > 1:
            tmp = tmp[0]
        x_max = x_subap[tmp[0]]
        y_max = y_subap[tmp[0]]
        shift_X = np.zeros(len(self.telescope.src.Na_profile[0, :]))
        shift_Y = np.zeros(len(self.telescope.src.Na_profile[0, :]))
        for i in range(len(self.telescope.src.Na_profile[0, :])):
            coordinates_3D[:2, i] = (self.telescope.D/4) * ([X0, Y0]/self.telescope.src.Na_profile[0, i])
            coordinates_3D[2, i] = self.telescope.D**2. / (8. * self.telescope.src.Na_profile[0, i])/(2.*np.sqrt(3.))
            coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(self.telescope.src.Na_profile[0, :])//2]
            # shift in the focal planee (in rad) associated to the LGS
            delta_dx[0, i] = coordinates_3D_ref[0, i] * (4/self.telescope.D)
            delta_dy[0, i] = coordinates_3D_ref[1, i] * (4/self.telescope.D)
            delta_dx[1, i] = coordinates_3D_ref[2, i] * (np.sqrt(3)*(4/self.telescope.D)**2) * x_max
            delta_dy[1, i] = coordinates_3D_ref[2, i] * (np.sqrt(3)*(4/self.telescope.D)**2) * y_max
            # resulting shift + conversion from radians to arcsec
            shift_X[i] = self.rad2arcsec*(delta_dx[0, i] + delta_dx[1, i])/self.pixel_scale_init
            shift_Y[i] = self.rad2arcsec*(delta_dy[0, i] + delta_dy[1, i])/self.pixel_scale_init

        # maximum elongation in number of pixel
        r_max = np.sqrt((shift_X[0]-shift_X[-1])**2+(shift_Y[0]-shift_Y[-1])**2)
        self.max_elongation_arcsec = r_max*self.pixel_scale_init
        # ensure to have an even number after binning
        n_even = int(np.ceil(1*r_max/self.n_pix_subap_init)*self.n_pix_subap_init)
        if n_even/self.binning_pixel_scale % 2 != 0:
            n_even = int((np.ceil(n_even/self.binning_pixel_scale/2))*2 * self.binning_pixel_scale)
        # consider the maximum number of pixel between the required pixel scale and the LGS elongation
        n_pix = np.max([n_even, self.n_pix_subap*self.binning_pixel_scale])
        spot_kernel_elongation_fft = []
        shift_x_buffer = []
        shift_y_buffer = []
        spot_kernel_elongation = []
        l_spot = []
        theta_spot = []
        count = -1
        for i_subap in range(len(x_subap)):
            for j_subap in range(len(y_subap)):
                count += 1
                if self.valid_subapertures_1D[count]:
                    intensity = np.zeros([n_pix, n_pix], dtype=(float))
                    shift_X = np.zeros(len(self.telescope.src.Na_profile[0, :]))
                    shift_Y = np.zeros(len(self.telescope.src.Na_profile[0, :]))
                    for i in range(len(self.telescope.src.Na_profile[0, :])):
                        coordinates_3D[:2, i] = (self.telescope.D/4) * ([X0, Y0]/self.telescope.src.Na_profile[0, i])
                        coordinates_3D[2, i] = self.telescope.D**2. / (8. * self.telescope.src.Na_profile[0, i])/(2.*np.sqrt(3.))
                        coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(self.telescope.src.Na_profile[0, :])//2]
                        # shift in the focal planee (in rad) associated to the LGS
                        delta_dx[0, i] = coordinates_3D_ref[0, i] * (4/self.telescope.D)
                        delta_dy[0, i] = coordinates_3D_ref[1, i] * (4/self.telescope.D)
                        delta_dx[1, i] = coordinates_3D_ref[2, i] * (np.sqrt(3)*(4/self.telescope.D)**2) * x_subap[i_subap]
                        delta_dy[1, i] = coordinates_3D_ref[2, i] * (np.sqrt(3)*(4/self.telescope.D)**2) * y_subap[j_subap]
                        # resulting shift + conversion from radians to arcsec
                        shift_X[i] = self.rad2arcsec*(delta_dx[0, i] + delta_dx[1, i])/self.pixel_scale_init
                        shift_Y[i] = self.rad2arcsec*(delta_dy[0, i] + delta_dy[1, i])/self.pixel_scale_init
                        # sum the 2D spots
                        if is_gaussian is False:
                            intensity += self.telescope.src.Na_profile[1, :][i] * gaussian_2D(resolution=n_pix,
                                                                                              fwhm=sigma_spot,
                                                                                              position=[shift_X[i], shift_Y[i]])
                    # length of the LGS spot in arcsec
                    r = (np.sqrt((np.max(shift_X*self.pixel_scale_init)-np.min(shift_X*self.pixel_scale_init))**2 +
                                 (np.max(shift_Y*self.pixel_scale_init)-np.min(shift_Y*self.pixel_scale_init))**2))
                    theta = np.pi+(np.pi/2+(np.arctan(-shift_X[-1]/shift_Y[-1]))) + np.pi/2
                    if is_gaussian:
                        intensity = gaussian_2D(resolution=n_pix,
                                                fwhm=[fwhm_factor*r/self.pixel_scale_init, fwhm_factor*self.telescope.src.FWHM_spot_up/self.pixel_scale_init],
                                                position=[0, 0],
                                                theta=theta).T
                    # truncation of the wings of the gaussian to speed up the convolution
                    intensity[intensity < self.threshold_convolution*intensity.max()] = 0
                    # normalization to conserve energy
                    intensity /= intensity.sum()
                    l_spot.append(r)
                    theta_spot.append(theta)
                    # save
                    shift_x_buffer.append(shift_X)
                    shift_y_buffer.append(shift_Y)
                    spot_kernel_elongation.append(intensity.T)
                    if compute_fft_kernel:
                        spot_kernel_elongation_fft.append((np.fft.fft2(intensity.T)))

        return np.asarray(shift_x_buffer), np.asarray(shift_y_buffer), np.asarray(spot_kernel_elongation_fft), np.asarray(spot_kernel_elongation)

    def sh_measure(self, phase_in):
        # backward compatibility with previous version
        self.wfs_measure(phase_in=phase_in)
        return

    def wfs_integrate(self):
        # propagate to detector to add noise and detector effects
        self*self.cam
        self.maps_intensity = self.split_raw_data()

        # compute the centroid on valid subaperture
        self.centroid_lenslets = self.centroid(self.maps_intensity*self.weighting_map, self.threshold_cog)

        # discard nan and inf values
        val_inf = np.where(np.isinf(self.centroid_lenslets))
        val_nan = np.where(np.isnan(self.centroid_lenslets))

        if np.shape(val_inf)[1] != 0:
            warning('Some subapertures are giving inf values!')
            self.centroid_lenslets[np.where(
                np.isinf(self.centroid_lenslets))] = 0

        if np.shape(val_nan)[1] != 0:
            warning('Some subapertures are giving nan values!')
            self.centroid_lenslets[np.where(
                np.isnan(self.centroid_lenslets))] = 0

        # compute slopes-maps
        self.SX[self.validLenslets_x, self.validLenslets_y] = self.centroid_lenslets[:, 0]
        self.SY[self.validLenslets_x, self.validLenslets_y] = self.centroid_lenslets[:, 1]

        signal_2D = np.concatenate((self.SX, self.SY)) - self.reference_slopes_maps
        signal_2D[~self.valid_slopes_maps] = 0

        signal_2D = signal_2D/self.slopes_units
        signal = signal_2D[self.valid_slopes_maps]

        return signal_2D, signal

    def wfs_measure(self, phase_in=None, integrate=True):
        if phase_in is not None:
            self.telescope.src.phase = phase_in

        if self.current_nPhoton != self.telescope.src.nPhoton:
            print('updating the flux of the SHWFS object')
            self.initialize_flux()

        if self.isInitialized:
            if self.wavelength_calibration != self.telescope.src.wavelength:
                raise OopaoError('A change in wavelength was detected in the WFS object \n' +
                                 'Make sure that the correct source is propagated in the WFS object or re-calibrate with the correct source.')

        if self.is_geometric is False:
            if np.ndim(phase_in) == 2:
                # -- case with a single wave-front to sense--
                # reset camera frame to be filled up
                self.raw_data = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,
                                         self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)

                # normalization for the FFT
                norma = self.cube_flux.shape[1]

                # compute spot intensity
                if self.telescope.spatialFilter is None:
                    phase = self.telescope.src.phase
                    self.initialize_flux()
                else:
                    phase = self.telescope.phase_filtered
                    self.initialize_flux(((self.telescope.amplitude_filtered)**2).T*self.telescope.src.fluxMap.T)
                intensity = (np.abs(np.fft.fft2(np.asarray(self.get_lenslet_em_field(phase)), axes=[1, 2])/norma)**2)
                # reduce to valid subaperture
                intensity = intensity[self.valid_subapertures_1D, :, :]

                self.sum_intensity = np.sum(intensity, axis=0)
                self.edge_subaperture_criterion = np.sum(intensity*self.outerMask)/np.sum(intensity)
                if self.edge_subaperture_criterion > 0.05:
                    warning('The light in the subaperture is probably wrapping!\n'+str(np.round(100*self.edge_subaperture_criterion, 1)) +
                            ' % of the total flux detected on the edges of the subapertures.\n' +
                            'You may want to lower the seeing value or increase the number of pixel per subaperture')

                # in case of LGS sensor, convolve with LGS spots kernel to create spot elungation
                if self.is_LGS:
                    if self.convolution_tag == 'FFT':
                        # zero pad the spot intensity to match LGS spot size for the FFT product
                        extra_pixel = (self.spot_kernel_elongation_fft.shape[1] - intensity.shape[1])//2

                        intensity = np.pad(intensity,
                                           [[0, 0],
                                            [extra_pixel, extra_pixel],
                                            [extra_pixel, extra_pixel]])

                        # compute convolution using the FFT
                        intensity = np.fft.fftshift(np.abs((np.fft.ifft2(np.fft.fft2(intensity)*self.spot_kernel_elongation_fft))), axes=[1, 2])
                        # bin the resulting image to the right pixel scale
                        intensity = bin_ndarray(intensity,
                                                [intensity.shape[0],
                                                 intensity.shape[1]//self.binning_pixel_scale,
                                                 intensity.shape[1]//self.binning_pixel_scale], operation='sum')

                        # crop the resulting spots to the right number of pixels
                        n_crop = (intensity.shape[1] - self.n_pix_subap)//2
                        if n_crop > 0:
                            intensity = intensity[:, n_crop:-n_crop, n_crop:-n_crop]
                    elif self.convolution_tag == 'direct':
                        n_crop = intensity.shape[1]//4
                        intensity = intensity[:, n_crop:-n_crop, n_crop:-n_crop]
                        # parallelization of the direct convolution using joblib

                        def joblib_convolve_direct():
                            Q = Parallel(n_jobs=12, prefer='threads')(delayed(self.convolve_direct)(i, j) for i, j in zip(self.spot_kernel_elongation, intensity))
                            return Q
                        intensity = np.asarray(joblib_convolve_direct())
                        # # bin the resulting image
                        intensity = bin_ndarray(intensity,
                                                [intensity.shape[0],
                                                 intensity.shape[1] // self.binning_pixel_scale,
                                                 intensity.shape[1] // self.binning_pixel_scale], operation='sum')
                        # crop the resulting spots
                        n_crop = (intensity.shape[1] - self.n_pix_subap)//2
                        if n_crop > 0:
                            intensity = intensity[:, n_crop:-n_crop, n_crop:-n_crop]
                else:
                    # set the sampling of the spots
                    if self.pixel_scale == self.pixel_scale_init:
                        intensity = intensity
                    elif self.pixel_scale < self.pixel_scale_init:
                        raise OopaoError('The smallest pixel scale value is ' + str(self.pixel_scale_init) + ' "')
                    else:
                        # pad the intensity to provide the right number of pixel before binning
                        self.extra_pixel = (self.binning_pixel_scale*self.n_pix_subap_init - intensity.shape[1])//2
                        intensity = np.pad(intensity,
                                           [[0, 0],
                                            [self.extra_pixel, self.extra_pixel],
                                            [self.extra_pixel, self.extra_pixel]])

                        # bin the spots to get the requested pixel scale
                        intensity = bin_ndarray(intensity, [
                                                          intensity.shape[0],
                                                          self.n_pix_subap_init,
                                                          self.n_pix_subap_init], operation='sum')

                # crop to the right number of pixel
                n_crop = (intensity.shape[1] - self.n_pix_subap)//2
                if n_crop > 0:
                    intensity = intensity[:, n_crop:-n_crop, n_crop:-n_crop]
                elif n_crop < 0:
                    intensity = np.pad(intensity, [[0, 0],
                                                   [-n_crop, -n_crop],
                                                   [-n_crop, -n_crop]])
                if self.binning_factor > 1:
                    intensity = bin_ndarray(intensity, [
                                                      intensity.shape[0],
                                                      self.n_pix_subap//self.binning_factor,
                                                      self.n_pix_subap//self.binning_factor], operation='sum')
                else:
                    if self.binning_factor != 1:
                        raise OopaoError('The binning factor must be a scalar >= 1')

                # fill camera frame with computed intensity (only valid subapertures)
                def joblib_fill_raw_data():
                    Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.fill_raw_data)(i, j, k) for i, j, k in zip(self.index_x[self.valid_subapertures_1D],
                                                                                                                       self.index_y[self.valid_subapertures_1D],
                                                                                                                       intensity))
                    return Q
                joblib_fill_raw_data()
                self.maps_intensity = intensity
                if integrate:
                    self.signal_2D, self.signal = self.wfs_integrate()
                    return [self.signal_2D, self.signal, intensity]
            else:
                # -- case with multiple wave-fronts to sense--
                # set phase buffer
                self.phase_buffer = np.moveaxis(self.telescope.src.phase, -1, 0)

                # reset camera frame
                self.raw_data = np.zeros([self.phase_buffer.shape[0],
                                          self.n_pix_subap*(self.nSubap)//self.binning_factor,
                                          self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)

                # compute 2D intensity for multiple input wavefronts
                def compute_diffractive_signals_multi():
                    Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.wfs_measure)(i) for i in self.phase_buffer)
                    return Q
                # compute the WFS maps and WFS signals
                m = compute_diffractive_signals_multi()

                # re-organization of signals into the right properties according to number of wavefronts considered
                self.signal_2D = np.zeros([m[0][0].shape[0],
                                           m[0][0].shape[1],
                                           self.phase_buffer.shape[0]])
                self.signal = np.zeros([m[0][1].shape[0],
                                        self.phase_buffer.shape[0]])
                self.maps_intensity = np.zeros([self.phase_buffer.shape[0],
                                                m[0][2].shape[0],
                                                m[0][2].shape[1],
                                                m[0][2].shape[2]])
                for i in range(self.phase_buffer.shape[0]):
                    self.signal_2D[:, :, i] = m[i][0]
                    self.signal[:, i] = m[i][1]
                    self.maps_intensity[i, :, :, :] = m[i][2]

                # fill up camera frame if requested (default is False)
                if self.get_raw_data_multi is True:
                    self.compute_raw_data_multi(self.maps_intensity)

        else:
            # Geometric SH with single WF
            if np.ndim(self.telescope.src.phase) == 2:
                self.signal_2D = self.lenslet_propagation_geometric(self.telescope.src.phase)*self.valid_slopes_maps/self.slopes_units
                self.signal = self.signal_2D[self.valid_slopes_maps]

            # Geometric SH with multiple WFS
            else:
                self.phase_buffer = np.moveaxis(self.telescope.src.phase, -1, 0)

                def compute_geometric_signals():
                    Q = Parallel(n_jobs=1, prefer='processes')(
                        delayed(self.lenslet_propagation_geometric)(i) for i in self.phase_buffer)
                    return Q
                maps = compute_geometric_signals()
                self.signal_2D = np.asarray(maps)/self.slopes_units
                self.signal = self.signal_2D[:, self.valid_slopes_maps].T

    @property
    def is_geometric(self):
        return self._is_geometric

    @is_geometric.setter
    def is_geometric(self, val):
        self._is_geometric = val
        if hasattr(self, 'isInitialized'):
            if self.isInitialized:
                print('Re-initializing WFS...')
                self.initialize_wfs()

    @property
    def half_pixel_shift(self):
        return self._half_pixel_shift

    @half_pixel_shift.setter
    def half_pixel_shift(self, val):
        self._half_pixel_shift = val
        if hasattr(self, 'isInitialized'):
            if self.isInitialized:
                # recompute the phasor to center spots in the center of the lenslets on 1 or 4 pixels
                [xx, yy] = np.meshgrid(np.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init),
                                       np.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init))
                self.phasor = np.exp(-(1j*np.pi*(self.n_pix_lenslet_init+1+(self.pixel_scale/self.pixel_scale_init)*self.half_pixel_shift) /
                                     self.n_pix_lenslet_init)*(xx+yy))
                self.phasor_tiled = np.moveaxis(
                    np.tile(self.phasor[:, :, None], self.nSubap**2), 2, 0)

    @property
    def lightRatio(self):
        return self._lightRatio

    @lightRatio.setter
    def lightRatio(self, val):
        self._lightRatio = val
        if hasattr(self, 'isInitialized'):
            if self.isInitialized:
                print('Selecting valid subapertures based on flux considerations..')

                self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(
                    self.photon_per_subaperture), [self.nSubap, self.nSubap])

                self.valid_subapertures_1D = np.reshape(
                    self.valid_subapertures, [self.nSubap**2])

                [self.validLenslets_x, self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))

                # index of valid slopes X and Y
                self.valid_slopes_maps = np.concatenate((self.valid_subapertures, self.valid_subapertures))

                # number of valid lenslet
                self.nValidSubaperture = int(np.sum(self.valid_subapertures))

                self.nSignal = 2*self.nValidSubaperture

                print('Re-initializing WFS...')
                self.initialize_wfs()
                print('Done!')

    def __mul__(self, obj):
        if obj.tag == 'detector':
            obj._integrated_time += self.telescope.samplingTime
            obj.integrate(self.raw_data)
        else:
            raise OopaoError('The light is propagated to the wrong type of object')
        return -1

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['subapertures'] = f"{'Subapertures [lenslets]':<25s}|{self.nSubap:^9d}"
        self.prop['subapertures_sky'] = f"{'Subaperture Pitch [m]':<25s}|{self.telescope.D/self.nSubap:^9.2f}"
        self.prop['fov'] = f"{'Subaperture FoV [arcsec]':<25s}|{self.pixel_scale*self.n_pix_subap:^9.2f}"
        self.prop['fov_pix'] = f"{'Pixel Scale [arcsec]':<25s}|{self.pixel_scale:^9.3f}"
        self.prop['n_valid_pixels'] = f"{'Valid Subapertures':<25s}|{self.nSignal:^9.0f}"
        if self.is_LGS:
            self.prop['spot_sampling'] = f"{'Spot Sampling [pix]':<25s}|{self.telescope.src.FWHM_spot_up/self.pixel_scale:^9.3f}"
            self.prop['spot_elongation'] = f"{'Max Elongation [arcsec]':<25s}|{self.max_elongation_arcsec:^9.3f}"
        else:
            self.prop['spot_sampling'] = f"{'Spot Sampling [pix]':<25s}|{self.zero_padding*self.pixel_scale_init/self.pixel_scale:^9.2f}"
        self.prop['geometric'] = f"{'Geometric WFS':<25s}|{str(self.is_geometric):^9s}"
        if self.is_geometric:
            warning('All Detector Noises are disables with the geometric WFS')
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Shack-Hartmann WFS ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
