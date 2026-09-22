# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""
import numpy as np
from .Detector import Detector
from .tools.tools import bin_ndarray, gaussian_2D, warning, OopaoError, emptyClass, get_array_module
from joblib import Parallel, delayed
from scipy import signal as sg
from .OPD_map import OPD_map

from .runtime import array_backend, precision_bits
xp, global_gpu_flag = array_backend()
from .runtime import to_numpy as _to_numpy

if global_gpu_flag:
    from cupyx.scipy import signal as csg
else:
    csg = sg


class ShackHartmann:
    def __init__(self, nSubap: int,
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
                 em_field_transform=None):
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
        precision = precision_bits()
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        # bridge to move the per-subaperture EM field / FFT propagation onto
        # GPU when cupy is available, mirroring Pyramid's convert_for_gpu /
        # convert_for_numpy pattern; both are no-ops on a plain numpy array,
        # so this is harmless when cupy isn't installed
        self.gpu_available = global_gpu_flag
        if self.gpu_available:
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
        else:
            self.convert_for_gpu = lambda a: a
            self.convert_for_numpy = lambda a: a
        # -------------------- INPUTS -------------------------
        # Number of subapertures (micro-lenses) across telescope diameter
        self.nSubap = nSubap
        # Telescope object carrying pupil, and flux information
        self.telescope = telescope
        # Flux ratio threshold used to determine valid subapertures
        self.lightRatio = lightRatio
        # Threshold (relative to max spot intensity) used for CoG computation
        self.threshold_cog = threshold_cog
        # If True → geometric WFS (direct gradient measurement)
        # If False → diffractive SH model
        self.is_geometric = is_geometric
        # Detector binning factor
        self.binning_factor = binning_factor
        # Requested detector pixel scale [arcsec]
        # If None → derived from shannon_sampling
        self.pixel_scale_requested = pixel_scale
        # Threshold forcing elongated LGS spots to zero at edges
        # (accelerates convolution)
        self.threshold_convolution = threshold_convolution
        # If pixel_scale is None:
        # True  → 2 pixels per FWHM (Shannon sampling)
        # False → 1 pixel per FWHM
        self.shannon_sampling = shannon_sampling
        # Number of pixels per subaperture FoV
        # If None → use telescope-driven default
        # If larger → zero-padding
        # If smaller → cropping
        self.n_pixel_per_subaperture = n_pixel_per_subaperture
        # If True → apply half-pixel shift to center spots
        # False → spots centered on 4 pixels
        self.half_pixel_shift = half_pixel_shift
        # Optional transformation applied to the EM field
        # (see FieldTransformer class)
        self.em_field_transform = em_field_transform
        # -----------------------------------------------------
        self.tag = 'shackHartmann'
        self.src = self.telescope.src
        if self.telescope.src is None:
            raise OopaoError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        if self.src.type == 'asterism':
            if self.src.src[0].type == 'LGS':
                self.is_LGS = True
                self.convolution_tag = 'FFT'
            else:
                self.is_LGS = False
        else:
            if self.src.type == 'LGS':
                self.is_LGS = True
                self.convolution_tag = 'FFT'
            else:
                self.is_LGS = False
        self.wavelength_calibration = self.src.wavelength
        # Size of the subaperture in [m]
        self.d_subap = telescope.D/self.nSubap
        # initial zero-padding for the ShackHartmann spot computation
        self.zero_padding = 2
        # conversion of the radians in arcsec
        self.rad2arcsec = 1 / (np.pi / 180 / 3600)
        # original pixel scale assuming a zeropadding factor of 2
        self.pixel_scale_init = self.rad2arcsec*self.telescope.src.wavelength/self.d_subap / self.zero_padding
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
        # xp (not np): multiplied against the on-GPU intensity cube in
        # wfs_measure, and cupy refuses to mix cupy/numpy arrays in arithmetic
        self.outerMask = xp.ones([self.n_pix_subap_init*self.zero_padding, self.n_pix_subap_init*self.zero_padding], dtype=self.precision)
        self.outerMask[1:-1, 1:-1] = 0
        # Compute camera frame in case of multiple measurements
        self.get_raw_data_multi = False
        # WFS detector object
        self.cam = Detector(round(nSubap*self.n_pix_subap),
                            output_precision=np.float32 if precision == 32 else None)
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0
        # memory budget (bytes) for one batch of wave-fronts when sensing a cube on the CPU (e.g. interaction matrix)
        self.cpu_batch_memory = 1e9
        # joblib parameter
        self.nJobs = 1
        self.joblib_prefer = 'processes'
        # camera frame
        self.raw_data = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor, self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)
        # phasor to center spots in the center of the lenslets
        # xp (not np): multiplied against the on-GPU cube_em in
        # get_lenslet_em_field, same cupy/numpy mixing restriction as above
        [xx, yy] = xp.meshgrid(xp.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init), xp.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init))
        # (at the working precision: a complex128 phasor would turn the lenslet FFTs into double precision)
        self.phasor = xp.exp(-(1j*xp.pi*(self.n_pix_lenslet_init+1+(self.pixel_scale/self.pixel_scale_init)*self.half_pixel_shift) / self.n_pix_lenslet_init)*(xx+yy)).astype(self.precision_complex)
        self.phasor_tiled = xp.moveaxis(xp.tile(self.phasor[:, :, None], self.nSubap**2), 2, 0)
        if self.src.tag == 'source':
            self.src_list = [self.src]
        elif self.src.tag == 'asterism':
            self.src_list = self.src.src
        self.nSignal = 0
        self.index_x = []
        self.index_y = []
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)
        self.src**self.telescope*self.em_field_transform
        self.sh_data = dict()
        for i_src, src in enumerate(self.src_list):
            self.sh_data['src_'+str(i_src)] = emptyClass()
            # self.sh_data.id = self.id
            # get the flux per subaperture
            self.initialize_flux(src=src, sh_data=self.sh_data['src_'+str(i_src)])
            # set the valid lenslets accordingly
            self.set_valid_subaperture(src=src, sh_data=self.sh_data['src_'+str(i_src)])
            # Set the pupil masks for each SH (needed for the geometric SH implementation with EM field transform)
            intensity_cpu = _to_numpy(src.intensity)
            self.sh_data['src_' + str(i_src)].pupil_mask = intensity_cpu == np.max(intensity_cpu)
            # initialize the weithing map for the CoG computation
            self.sh_data['src_'+str(i_src)].weighting_map = 1
            # store the number of signal
            self.nSignal += 2 * self.sh_data['src_'+str(i_src)].nValidSubaperture
            # compute the LGS kernels if required
            if src.type == "LGS" and self.is_geometric is False:
                self.is_LGS = True
                _, _, self.sh_data['src_'+str(i_src)].spot_kernel_elongation_fft, self.sh_data['src_'+str(i_src)].spot_kernel_elongation = self.get_convolution_spot(compute_fft_kernel=True,
                                                                                                                                                                     src=src,
                                                                                                                                                                     sh_data=self.sh_data['src_'+str(i_src)])
            else:
                self.is_LGS = False
        # initialize the WFS (reference signal acquisition and slopes units calibration)
        self.initialize_wfs()
        return

    def initialize_flux(self, src, sh_data, flux_map=None):
        # this stays on numpy (its only output feeds the valid-subaperture
        # threshold mask, inherently a host-side control-flow decision) but
        # the per-subaperture split is vectorized the same way as
        # get_lenslet_em_field: one reshape/transpose instead of a
        # per-subaperture Python loop
        # (src.intensity is a CuPy array with GPU residency: bring it to the host first)
        flux = _to_numpy(src.intensity if flux_map is None else flux_map)
        flux_tiles = self._lenslet_tiles(flux)
        npx = self.n_pix_subap_init
        sh_data.cube_flux = np.zeros([self.nSubap ** 2,
                                      self.n_pix_lenslet_init,
                                      self.n_pix_lenslet_init], dtype=float)
        c = self.center_init - npx // 2
        sh_data.cube_flux[:, c:c+npx, c:c+npx] = flux_tiles
        # required to compute the valid subapertures
        # np.apply_over_axes has no cupy equivalent; xp.sum(..., keepdims=True)
        # is the direct, backend-agnostic replacement (same shape/values)
        xp_local = get_array_module(sh_data.cube_flux)
        sh_data.photon_per_subaperture = xp_local.sum(sh_data.cube_flux, axis=(1, 2), keepdims=True)
        sh_data.current_nPhoton = src.nPhoton
        return

    def set_valid_subaperture(self, src, sh_data):
        self.photon_per_subaperture_2D = np.reshape(sh_data.photon_per_subaperture, [self.nSubap, self.nSubap])
        sh_data.valid_subapertures = np.reshape(sh_data.photon_per_subaperture >= self.lightRatio*np.max(sh_data.photon_per_subaperture), [self.nSubap, self.nSubap])
        sh_data.valid_subapertures_1D = np.reshape(sh_data.valid_subapertures, [self.nSubap**2])
        [sh_data.validLenslets_x, sh_data.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(sh_data.photon_per_subaperture))
        # index of valid slopes X and Y
        sh_data.valid_signal_2D = np.concatenate((sh_data.valid_subapertures, sh_data.valid_subapertures))
        # number of valid lenslet
        sh_data.nValidSubaperture = int(np.sum(sh_data.valid_subapertures))
        # required to compute LGS kernels
        sh_data.valid_subapertures_1D = sh_data.valid_subapertures_1D.copy()
        sh_data.valid_signal_2D = sh_data.valid_signal_2D.copy()
        # copy for backward compatibility
        for name, value in sh_data.__dict__.items():
            if not name.startswith("__"):  # skip built-ins
                setattr(self, name, value)
        return

    def initialize_wfs(self):
        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0
        self.isInitialized = False

        self.src**self.telescope*self.em_field_transform
        if self.src.tag == 'source':
            self.src_list = [self.src]
        elif self.src.tag == 'asterism':
            self.src_list = self.src.src
        # reference signal initialization
        self.SX = np.zeros([self.nSubap, self.nSubap])
        self.SY = np.zeros([self.nSubap, self.nSubap])
        # flux per subaperture
        self.slopes_units = 1
        # self.reference_signal_2D = []
        print('Acquiring reference slopes..')
        self.reference_signal_2D = []
        for i_src, src in enumerate(self.src_list):
            self.sh_data['src_'+str(i_src)].reference_signal_2D = 0
        self.relay(self.src)
        self.reference_signal_2D = np.squeeze(np.asarray(self.signal_2D))
        if len(self.src_list) == 1:
            self.sh_data['src_'+str(i_src)].reference_signal_2D = self.signal_2D
        else:
            for i_src, src in enumerate(self.src_list):
                self.sh_data['src_'+str(i_src)].reference_signal_2D = self.signal_2D[i_src, :, :]

        self.isInitialized = True
        print('Done!')
        # Calibrate the the WFS units
        self.set_slopes_units()
        self.cam.photonNoise = photonNoise
        self.cam.readoutNoise = readoutNoise
        # print SHWFS propeties
        self.print_properties()
        return

    def relay(self, src):
        if src.tag == 'source':
            self.src_list = [src]
        elif src.tag == 'asterism':
            self.src_list = src.src
        signal_2D_list = []
        signal_list = []
        frames_list = []
        for i_src, src in enumerate(self.src_list):
            src = self.src_list[i_src]
            src.optical_path.append([self.tag, self])
            # initialize flux for each source to capture an eventual change
            self.initialize_flux(src, sh_data=self.sh_data['src_'+str(i_src)])
            # compute the WFS measurements for the given src
            self.wfs_measure(phase_in=src.phase, src=src, sh_data=self.sh_data['src_'+str(i_src)])
            signal_2D_list.append(np.squeeze(self.signal_2D))
            signal_list.append(np.squeeze(self.signal))
            if self.is_geometric is False:
                frames_list.append(self.cam.frame)
        # print(self.signal_2D.shape)
        self.signal_2D = np.asarray(np.squeeze(signal_2D_list))
        self.signal = np.concatenate(signal_list)
        if self.is_geometric is False:
            self.cam.frame = np.squeeze(np.array(frames_list))
        return

    def wfs_integrate(self, src, sh_data):
        # propagate to detector to add noise and detector effects
        self*self.cam
        self.maps_intensity = self.split_raw_data(sh_data=sh_data)
        # compute the centroid on valid subaperture
        self.centroid_lenslets = self.centroid(self.maps_intensity*sh_data.weighting_map, self.threshold_cog)
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
        self.SX[sh_data.validLenslets_x, sh_data.validLenslets_y] = self.centroid_lenslets[:, 0]
        self.SY[sh_data.validLenslets_x, sh_data.validLenslets_y] = self.centroid_lenslets[:, 1]
        signal_2D = np.concatenate((self.SX, self.SY)) - sh_data.reference_signal_2D
        signal_2D[~sh_data.valid_signal_2D] = 0
        signal_2D = signal_2D/self.slopes_units
        signal = signal_2D[sh_data.valid_signal_2D]
        return signal_2D, signal

    def wfs_measure(self, src, sh_data, phase_in=None, integrate=True):
        if phase_in is not None:
            src.phase = phase_in
        if self.isInitialized:
            if self.wavelength_calibration != self.telescope.src.wavelength:
                raise OopaoError('A change in wavelength was detected in the WFS object \n' +
                                 'Make sure that the correct source is propagated in the WFS object or re-calibrate with the correct source.')
        if self.is_geometric is False:
            if np.ndim(src.phase) == 2:
                # -- case with a single wave-front to sense--
                # reset camera frame to be filled up
                self.raw_data = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,
                                         self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)
                # normalization for the FFT
                norma = sh_data.cube_flux.shape[1]
                # compute spot intensity
                if src.phase_filtered is None:
                    phase = src.phase
                    amplitude = None
                else:
                    # spatially filtered wave-front (SpatialFilter): the filtered field replaces the incoming one.
                    # Its amplitude already carries the flux, so the lenslet fields use it directly and the
                    # flux per subaperture is its square
                    phase = src.phase_filtered
                    amplitude = _to_numpy(src.amplitude_filtered)
                    self.initialize_flux(src, sh_data, flux_map=amplitude**2)
                # get_lenslet_em_field uploads to GPU internally when available;
                # the FFT propagation (the dominant cost) then runs on-device,
                # for the valid subapertures only
                em_field = self.get_lenslet_em_field(src=src, sh_data=sh_data, phase=phase, amplitude=amplitude)
                intensity = (xp.abs(xp.fft.fft2(em_field[self._valid_index(sh_data)], axes=[1, 2])/norma)**2)
                self.sum_intensity = xp.sum(intensity, axis=0)
                self.edge_subaperture_criterion = float(xp.sum(intensity*self.outerMask)/xp.sum(intensity))
                self._check_edge_criterion(self.edge_subaperture_criterion)
                intensity = self._process_spots(intensity, sh_data)
                self.raw_data = self._spots_to_frame(intensity, sh_data)
                self.maps_intensity = intensity
                if integrate:
                    self.signal_2D, self.signal = self.wfs_integrate(src=src, sh_data=sh_data)
                    return [self.signal_2D, self.signal, intensity]
            else:
                # -- case with multiple wave-fronts to sense (cube with the modes along the last axis) --
                self._wfs_measure_cube(src, sh_data)
        else:
            # Geometric SH with single WF
            if np.ndim(src.phase) == 2:
                self.signal_2D = self.lenslet_propagation_geometric(_to_numpy(src.phase), sh_data.pupil_mask)*sh_data.valid_signal_2D/self.slopes_units
                self.signal = self.signal_2D[sh_data.valid_signal_2D]
            # Geometric SH with multiple WFS
            else:
                self.phase_buffer = np.moveaxis(_to_numpy(src.phase), -1, 0)

                def compute_geometric_signals():
                    Q = Parallel(n_jobs=1, prefer='processes')(
                        delayed(self.lenslet_propagation_geometric)(arr=i, pupil_mask=sh_data.pupil_mask) for i in self.phase_buffer)
                    return Q
                maps = compute_geometric_signals()
                self.signal_2D = np.asarray(maps)/self.slopes_units
                self.signal = self.signal_2D[:, sh_data.valid_signal_2D].T
        return

    def _lenslet_tiles(self, array):
        """Split (..., n, n) pupil-plane maps into (..., nSubap**2, npx, npx) lenslet tiles.

        Single reshape/transpose (tile ordering i*nSubap+j, as the original hsplit/vsplit loop); works on
        NumPy and CuPy arrays and on stacks of maps.
        """
        npx = self.n_pix_subap_init
        lead = array.shape[:-2]
        k = len(lead)
        tiles = array.swapaxes(-1, -2).reshape(lead + (self.nSubap, npx, self.nSubap, npx))
        tiles = tiles.transpose(tuple(range(k)) + (k+2, k, k+1, k+3))
        return tiles.reshape(lead + (self.nSubap**2, npx, npx))

    def _valid_index(self, sh_data):
        """Integer indices of the valid subapertures on the working backend (computed once per selection)."""
        cache = getattr(sh_data, '_valid_index_cache', None)
        if cache is None or cache[0] is not sh_data.valid_subapertures_1D:
            cache = (sh_data.valid_subapertures_1D, xp.asarray(np.flatnonzero(sh_data.valid_subapertures_1D)))
            sh_data._valid_index_cache = cache
        return cache[1]

    def _check_edge_criterion(self, criterion):
        if criterion > 0.05:
            warning('The light in the subaperture is probably wrapping!\n'+str(np.round(100*criterion, 1)) +
                    ' % of the total flux detected on the edges of the subapertures.\n' +
                    'You may want to lower the seeing value or increase the number of pixel per subaperture')

    def _process_spots(self, intensity, sh_data):
        """From the (nValid, N, N) lenslet intensities to the spots on the detector pixels.

        LGS elongation, pixel scale, crop/padding to n_pix_subap and binning.
        """
        # in case of LGS sensor, convolve with LGS spots kernel to create spot elungation
        if self.is_LGS:
            # LGS kernels (get_convolution_spot) are precomputed on the
            # host; keep this less-common path on numpy rather than
            # threading GPU support through per-subaperture kernels
            # that differ from one subaperture to the next
            intensity = self.convert_for_numpy(intensity)
            if self.convolution_tag == 'FFT':
                # zero pad the spot intensity to match LGS spot size for the FFT product
                extra_pixel = (sh_data.spot_kernel_elongation_fft.shape[1] - intensity.shape[1])//2
                intensity = np.pad(intensity,
                                   [[0, 0],
                                    [extra_pixel, extra_pixel],
                                    [extra_pixel, extra_pixel]])
                # compute convolution using the FFT
                intensity = np.fft.fftshift(np.abs((np.fft.ifft2(np.fft.fft2(intensity)*sh_data.spot_kernel_elongation_fft))), axes=[1, 2])
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
                    Q = Parallel(n_jobs=12, prefer='threads')(delayed(self.convolve_direct)(i, j) for i, j in zip(sh_data.spot_kernel_elongation, intensity))
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
                xp_ = get_array_module(intensity)
                intensity = xp_.pad(intensity, [[0, 0], [self.extra_pixel, self.extra_pixel], [self.extra_pixel, self.extra_pixel]])
                # bin the spots to get the requested pixel scale
                intensity = bin_ndarray(intensity, [intensity.shape[0], self.n_pix_subap_init, self.n_pix_subap_init], operation='sum')
        # crop to the right number of pixel (backend-agnostic: dispatches on
        # whatever produced `intensity` above -- numpy for the LGS branch,
        # xp for the common non-LGS branch)
        xp_ = get_array_module(intensity)
        n_crop = (intensity.shape[1] - self.n_pix_subap)//2
        if n_crop > 0:
            intensity = intensity[:, n_crop:-n_crop, n_crop:-n_crop]
        elif n_crop < 0:
            intensity = xp_.pad(intensity, [[0, 0],
                                            [-n_crop, -n_crop],
                                            [-n_crop, -n_crop]])
        if self.binning_factor > 1:
            intensity = bin_ndarray(intensity, [intensity.shape[0], self.n_pix_subap//self.binning_factor, self.n_pix_subap//self.binning_factor], operation='sum')
        else:
            if self.binning_factor != 1:
                raise OopaoError('The binning factor must be a scalar >= 1')
        return intensity

    def _spots_to_frame(self, intensity, sh_data):
        """Detector frame (NumPy) with the spots of the valid subapertures, zero elsewhere."""
        # one vectorized scatter + reshape/transpose instead of a per-subaperture loop
        xp_ = get_array_module(intensity)
        tile = intensity.shape[1]
        full_cube = xp_.zeros((self.nSubap**2, tile, tile), dtype=intensity.dtype)
        full_cube[sh_data.valid_subapertures_1D if xp_ is np else self._valid_index(sh_data)] = intensity
        return self.convert_for_numpy(
            full_cube.reshape(self.nSubap, self.nSubap, tile, tile)
                     .transpose(0, 2, 1, 3)
                     .reshape(self.nSubap*tile, self.nSubap*tile))

    def _modes_per_batch(self, sh_data):
        """Number of wave-fronts propagated together when sensing a cube, from the memory available."""
        n_lenslet = self.n_pix_lenslet_init
        # peak usage per wave-front: EM field, its FFT and the intensity of the valid lenslets, plus the tiles
        item_bytes = 4 * max(1, sh_data.nValidSubaperture) * n_lenslet**2 * np.dtype(self.precision_complex).itemsize
        if self.gpu_available:
            free_bytes = xp.cuda.runtime.memGetInfo()[0] + xp.get_default_memory_pool().free_bytes()
            return int(max(1, 0.5 * free_bytes // item_bytes))
        return int(max(1, self.cpu_batch_memory // item_bytes))

    def _wfs_measure_cube(self, src, sh_data):
        """Diffractive SH on a (n, n, n_modes) cube of phases (e.g. interaction matrix).

        The lenslet FFTs of several wave-fronts run as one batched call (sized from the memory available);
        the detector and centroiding then run frame by frame, exactly as for a single wave-front.
        """
        phase = src.phase
        n_modes = phase.shape[-1]
        norma = sh_data.cube_flux.shape[1]
        index = self._valid_index(sh_data)
        amp_tiles = self._lenslet_tiles(xp.sqrt(xp.asarray(src.intensity)))[index]
        phasor = self.phasor_tiled[index]
        c = self.center_init - self.n_pix_subap_init//2
        npx = self.n_pix_subap_init
        batch = self._modes_per_batch(sh_data)
        signal_2D, signal, maps = None, None, []
        for start in range(0, n_modes, batch):
            stop = min(start + batch, n_modes)
            phase_tiles = self._lenslet_tiles(xp.moveaxis(xp.asarray(phase[:, :, start:stop]), -1, 0))[:, index]
            cube_em = xp.zeros((stop - start, index.shape[0], self.n_pix_lenslet_init, self.n_pix_lenslet_init), dtype=self.precision_complex)
            cube_em[..., c:c+npx, c:c+npx] = amp_tiles * xp.exp(1j*phase_tiles)
            del phase_tiles
            cube_em *= phasor
            intensity = xp.abs(xp.fft.fft2(cube_em, axes=[-2, -1])/norma)**2
            del cube_em
            criterion = self.convert_for_numpy(xp.sum(intensity*self.outerMask, axis=(1, 2, 3))/xp.sum(intensity, axis=(1, 2, 3)))
            self._check_edge_criterion(float(np.max(criterion)))
            self.edge_subaperture_criterion = float(criterion[-1])
            for i in range(stop - start):
                spots = self._process_spots(intensity[i], sh_data)
                self.raw_data = self._spots_to_frame(spots, sh_data)
                signal_2D_i, signal_i = self.wfs_integrate(src=src, sh_data=sh_data)
                if signal_2D is None:
                    signal_2D = np.zeros([signal_2D_i.shape[0], signal_2D_i.shape[1], n_modes])
                    signal = np.zeros([signal_i.shape[0], n_modes])
                signal_2D[:, :, start + i] = signal_2D_i
                signal[:, start + i] = signal_i
                maps.append(self.convert_for_numpy(spots))
            self.sum_intensity = xp.sum(intensity[-1], axis=0)
            del intensity
        self.signal_2D = signal_2D
        self.signal = signal
        self.maps_intensity = np.asarray(maps)
        # fill up camera frames if requested (default is False)
        if self.get_raw_data_multi is True:
            self.raw_data = np.zeros([n_modes, self.n_pix_subap*(self.nSubap)//self.binning_factor, self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype=float)
            self.compute_raw_data_multi(intensity=self.maps_intensity.reshape((-1,) + self.maps_intensity.shape[2:]), sh_data=sh_data)
        if self.gpu_available:
            try:
                xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                warning('could not free the memory')

    def set_weighted_centroiding_map(self, src, is_lgs: bool, is_gaussian: bool, fwhm_factor, sh_data=None):
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
        if src.tag == 'asterism':
            for i_src, src_ast in enumerate(src.src):
                self.set_weighted_centroiding_map(is_lgs=is_lgs, is_gaussian=is_gaussian, fwhm_factor=fwhm_factor, src=src_ast, sh_data=self.sh_data['src_'+str(i_src)])
                print('Re-calibrating the reference signal with the nex weighting map')
            self.initialize_wfs()
        else:
            if sh_data is None:
                sh_data = self.sh_data['src_0']
            if is_lgs:
                print('The weighting map is based on the LGS spots kernel')
                _, _, _, weighting_map = self.get_convolution_spot(fwhm_factor=fwhm_factor, compute_fft_kernel=False, is_gaussian=is_gaussian, src=src, sh_data=sh_data)
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
                weighting_map = np.tile(gaussian_2D(resolution=self.n_pix_subap, fwhm=fwhm_factor), [sh_data.nValidSubaperture, 1, 1])
            warning('A new weighting map is now considered.')
            sh_data.weighting_map = weighting_map
            self.weighting_map = weighting_map
            if self.src.tag == 'source':
                print('Re-calibrating the reference signal with the nex weighting map')
                self.initialize_wfs()
        return

    def set_slopes_units(self, src=None, tomographic_reconstructor=None, dm=None):
        if src is None:
            src = self.src
        print('Calibrating the slopes units')
        [self.Tip, self.Tilt] = np.meshgrid(np.linspace(-np.pi, np.pi, self.telescope.resolution, endpoint=False),
                                            np.linspace(-np.pi, np.pi, self.telescope.resolution, endpoint=False))

        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        self.cam.photonNoise = 0
        self.cam.readoutNoise = 0
        self.slopes_units = 1
        if tomographic_reconstructor is None:
            TT_in = OPD_map(((self.Tip+self.Tilt)*src.wavelength/2/np.pi) * self.pixel_scale / (src.wavelength / self.telescope.D) / self.rad2arcsec)
            src**self.telescope*TT_in*self.em_field_transform
            self.relay(src)
            self.slopes_units = np.mean(self.signal)
        else:
            if src.type != 'asterism':
                raise OopaoError('Only asterisms objects can be used to calibrate the SH WFS with a tomographic reconsctrutor')
            TT_out = OPD_map(self.Tip*0)
            for i in range(2):
                TT_in = OPD_map((self.Tip - TT_out.OPD)*self.telescope.pupil*1e-9)
                src**self.telescope*TT_in*self.em_field_transform
                self.relay(src)
                wfs_signal = np.hstack(self.signal)
                rec_commands = tomographic_reconstructor@wfs_signal
                TT_out = OPD_map((dm.modes@rec_commands).reshape([self.telescope.resolution, self.telescope.resolution])*self.telescope.pupil)
                self.slopes_units *= np.std(TT_out.OPD) / np.std(TT_in.OPD)
        self.cam.photonNoise = photonNoise
        self.cam.readoutNoise = readoutNoise
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

    def get_lenslet_em_field(self, src, sh_data, phase, amplitude=None):
        # split the pupil-plane phase/amplitude into the nSubap x nSubap lenslet tiles and upload once
        # to GPU when available -- the FFT propagation right after this is the dominant cost.
        # amplitude defaults to sqrt(src.intensity).
        if amplitude is None:
            amplitude = xp.sqrt(xp.asarray(src.intensity))
        phase_tiles = self._lenslet_tiles(xp.asarray(phase))
        amp_tiles = self._lenslet_tiles(xp.asarray(amplitude))
        npx = self.n_pix_subap_init
        # Keep the original complex128 path in default precision. Single
        # precision uses complex64 for the dominant lenslet FFT allocation.
        self.cube_em = xp.zeros([self.nSubap**2,
                                 self.n_pix_lenslet_init,
                                 self.n_pix_lenslet_init], dtype=self.precision_complex)
        c = self.center_init - npx//2
        self.cube_em[:, c:c+npx, c:c+npx] = amp_tiles * xp.exp(1j*phase_tiles)
        self.cube_em *= self.phasor_tiled

        return self.cube_em

    def fill_raw_data(self, ind_x, ind_y, intensity, index_frame=None):
        if index_frame is None:
            self.raw_data[ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,
                          ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = intensity
        else:
            self.raw_data[index_frame,
                          ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,
                          ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = intensity

    def merge_data_cube(self, cube, sh_data):
        # save current raw data
        tmp_raw_data = self.raw_data.copy()

        def joblib_fill_raw_data():
            Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.fill_raw_data)(i, j, k) for i, j, k in zip(self.index_x[sh_data.valid_subapertures_1D],
                                                                                                               self.index_y[sh_data.valid_subapertures_1D],
                                                                                                               cube))
            return Q
        joblib_fill_raw_data()
        output_raw_data = self.raw_data.copy()
        # re-assign raw_data
        self.raw_data = tmp_raw_data.copy()
        return output_raw_data

    def split_raw_data(self, sh_data=None, input_frame=None):
        if sh_data is None:
            sh_data = self.sh_data['src_0']
        if input_frame is None:
            input_frame = self.cam.frame
        # single reshape/transpose instead of a per-subaperture Python loop
        # (validated to reproduce the old vsplit/hsplit loop's tile ordering
        # exactly: i*nSubap+j)
        tile = self.n_pix_subap // self.binning_factor
        tiles = input_frame.reshape(self.nSubap, tile, self.nSubap, tile).transpose(0, 2, 1, 3).reshape(self.nSubap**2, tile, tile)
        if tile == self.n_pix_subap:
            maps_intensity = tiles
        else:
            maps_intensity = np.zeros([self.nSubap**2,
                                       self.n_pix_subap,
                                       self.n_pix_subap], dtype=float)
            center = self.n_pix_subap//2
            maps_intensity[:, center-tile//2:center+tile//2, center-tile//2:center+tile//2] = tiles
        maps_intensity = maps_intensity[sh_data.valid_subapertures_1D, :, :]
        return maps_intensity

    def compute_raw_data_multi(self, intensity, sh_data):
        n_frames = intensity.shape[0] // sh_data.nValidSubaperture
        self.ind_frame = np.zeros(intensity.shape[0], dtype=(int))
        index_x = np.tile(self.index_x[sh_data.valid_subapertures_1D], n_frames)
        index_y = np.tile(self.index_y[sh_data.valid_subapertures_1D], n_frames)
        for i in range(n_frames):
            self.ind_frame[i*sh_data.nValidSubaperture:(i+1)*sh_data.nValidSubaperture] = i

        def joblib_fill_raw_data():
            Q = Parallel(n_jobs=1, prefer='processes')(delayed(self.fill_raw_data)(i, j, k, l) for i, j, k, l in zip(index_x, index_y, intensity, self.ind_frame))
            return Q
        joblib_fill_raw_data()
        return

    def gradient_2D(self, arr, pupil_mask):
        arr[~pupil_mask] = np.nan
        res_x = (np.gradient(arr, axis=0, edge_order=1)/self.telescope.pixelSize) * pupil_mask
        res_x = np.nan_to_num(res_x)
        res_y = (np.gradient(arr, axis=1, edge_order=1)/self.telescope.pixelSize) * pupil_mask
        res_y = np.nan_to_num(res_y)
        return res_x, res_y

    def lenslet_propagation_geometric(self, arr, pupil_mask):
        [SLx, SLy] = self.gradient_2D(arr, pupil_mask)
        sy = bin_ndarray(ndarray=SLx, new_shape=(self.nSubap, self.nSubap), operation="mean", ignore_zeros=True)
        sx = bin_ndarray(ndarray=SLy, new_shape=(self.nSubap, self.nSubap), operation="mean", ignore_zeros=True)
        # self.lighted_subap = np.concatenate((sx.astype(bool), sy.astype(bool)))
        return np.concatenate((sx, sy))

    def convolve_direct(self, A_in, B_in):
        # dispatch on the actual array backend rather than on whether cupy is
        # merely importable: cupyx.scipy.signal.convolve rejects plain numpy
        # input outright, so this stays correct regardless of the state of the
        # rest of the pipeline (still numpy-only for now).
        signal_module = sg if get_array_module(A_in) is np else csg
        return signal_module.convolve(A_in, B_in, mode='same', method='direct')

    def get_convolution_spot(self, src, sh_data, fwhm_factor=1, compute_fft_kernel=False, is_gaussian=False):
        print('Computing LGS spots convolution kernels...')
        # compute the projection of the LGS on the subaperture to simulate
        # the spots elongation using a convulotion with gaussian spot
        # the src object is required to extract the coordinates of the LLT
        # (sign convention adjusted to match display position on camera)
        [X0, Y0] = [src.laser_coordinates[1], -src.laser_coordinates[0]]

        # 3D coordinates
        coordinates_3D = np.zeros([3, len(src.Na_profile[0, :])])
        coordinates_3D_ref = np.zeros([3, len(src.Na_profile[0, :])])

        # variable to store the shift due to the elongation
        delta_dx = np.zeros([2, len(src.Na_profile[0, :])])
        delta_dy = np.zeros([2, len(src.Na_profile[0, :])])

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
        sigma_spot = fwhm_factor*src.fwhm_spot_up/self.pixel_scale_init
        for i in range(len(src.Na_profile[0, :])):
            coordinates_3D[:2, i] = (self.telescope.D/4)*([X0, Y0]/src.Na_profile[0, i])
            coordinates_3D[2, i] = self.telescope.D**2. / (8.*src.Na_profile[0, i])/(2.*np.sqrt(3.))
            coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(src.Na_profile[0, :])//2]
        distance_to_llt = np.sqrt((X-X0)**2 + (Y-Y0)**2)
        tmp = (np.where(distance_to_llt == np.max(distance_to_llt)))
        if len(tmp) > 1:
            tmp = tmp[0]
        x_max = x_subap[tmp[0]]
        y_max = y_subap[tmp[0]]
        shift_X = np.zeros(len(src.Na_profile[0, :]))
        shift_Y = np.zeros(len(src.Na_profile[0, :]))
        for i in range(len(src.Na_profile[0, :])):
            coordinates_3D[:2, i] = (self.telescope.D/4) * ([X0, Y0]/src.Na_profile[0, i])
            coordinates_3D[2, i] = self.telescope.D**2. / (8. * src.Na_profile[0, i])/(2.*np.sqrt(3.))
            coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(src.Na_profile[0, :])//2]
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
                if sh_data.valid_subapertures_1D[count]:
                    intensity = np.zeros([n_pix, n_pix], dtype=(float))
                    shift_X = np.zeros(len(src.Na_profile[0, :]))
                    shift_Y = np.zeros(len(src.Na_profile[0, :]))
                    for i in range(len(src.Na_profile[0, :])):
                        coordinates_3D[:2, i] = (self.telescope.D/4) * ([X0, Y0]/src.Na_profile[0, i])
                        coordinates_3D[2, i] = self.telescope.D**2. / (8. * src.Na_profile[0, i])/(2.*np.sqrt(3.))
                        coordinates_3D_ref[:, i] = coordinates_3D[:, i] - coordinates_3D[:, len(src.Na_profile[0, :])//2]
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
                            intensity += src.Na_profile[1, :][i] * gaussian_2D(resolution=n_pix,
                                                                               fwhm=sigma_spot,
                                                                               position=[shift_X[i], shift_Y[i]])
                    # length of the LGS spot in arcsec
                    r = (np.sqrt((np.max(shift_X*self.pixel_scale_init)-np.min(shift_X*self.pixel_scale_init))**2 +
                                 (np.max(shift_Y*self.pixel_scale_init)-np.min(shift_Y*self.pixel_scale_init))**2))
                    theta = np.pi+(np.pi/2+(np.arctan(-shift_X[-1]/shift_Y[-1]))) + np.pi/2
                    if is_gaussian:
                        intensity = gaussian_2D(resolution=n_pix,
                                                fwhm=[fwhm_factor*r/self.pixel_scale_init, fwhm_factor*self.src.fwhm_spot_up/self.pixel_scale_init],
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

    # def get_valid_actuators(self):
    #     n_elements = 2 * self.nSubap + 1  # Linear number of lenslet+actuator
    #     valid_lenslet_actuator = np.zeros((n_elements, n_elements), dtype=int)
    #     index = np.arange(1, n_elements, 2)  # Lenslet index
    #     # valid_lenslet_actuator[np.ix_(index, index)] = src.valid_subapertures
    #     valid_lenslet_actuator[np.ix_(index, index)] = src.valid_subapertures
    #     for x_lenslet in index:
    #         for y_lenslet in index:
    #             if valid_lenslet_actuator[x_lenslet, y_lenslet] == 1:
    #                 x_actuator_indices = [x_lenslet - 1, x_lenslet - 1, x_lenslet + 1, x_lenslet + 1]
    #                 y_actuator_indices = [y_lenslet - 1, y_lenslet + 1, y_lenslet + 1, y_lenslet - 1]
    #                 for x_act, y_act in zip(x_actuator_indices, y_actuator_indices):
    #                     valid_lenslet_actuator[x_act, y_act] = 1
    #     index = np.arange(0, n_elements, 2)
    #     val = valid_lenslet_actuator[np.ix_(index, index)].astype(bool)
    #     return val

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
                # xp (not np): see the matching comment in __init__
                [xx, yy] = xp.meshgrid(xp.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init),
                                       xp.linspace(0, self.n_pix_lenslet_init-1, self.n_pix_lenslet_init))
                self.phasor = xp.exp(-(1j*xp.pi*(self.n_pix_lenslet_init+1+(self.pixel_scale/self.pixel_scale_init)*self.half_pixel_shift) /
                                     self.n_pix_lenslet_init)*(xx+yy)).astype(self.precision_complex)
                self.phasor_tiled = xp.moveaxis(
                    xp.tile(self.phasor[:, :, None], self.nSubap**2), 2, 0)

    def __mul__(self, obj):
        if obj.tag == 'detector':
            obj._integrated_time += self.telescope.samplingTime
            obj.integrate(self.raw_data)
        else:
            raise OopaoError('The light is propagated to the wrong type of object')

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['subapertures'] = f"{'Subapertures [lenslets]':<25s}|{self.nSubap:^9d}"
        self.prop['subapertures_sky'] = f"{'Subaperture Pitch [m]':<25s}|{self.telescope.D/self.nSubap:^9.2f}"
        self.prop['fov'] = f"{'Subaperture FoV [arcsec]':<25s}|{self.pixel_scale*self.n_pix_subap:^9.2f}"
        self.prop['fov_pix'] = f"{'Pixel Scale [arcsec]':<25s}|{self.pixel_scale:^9.3f}"
        # n_signal_str = ", ".join(f"{v:.0f}" for v in self.nSignal)
        # self.prop['n_valid_pixels'] = f"{'Valid Subapertures':<25s}|{", ".join(f"{v:.0f}" for v in self.nSignal):^9s}"
        self.prop['n_valid_pixels'] = f"{'Valid Subapertures':<25s}|{self.nSignal:^9.0f}"
        if self.is_LGS and self.is_geometric is False:
            for src in self.src_list:
                self.prop['spot_sampling'] = f"{'Spot Sampling [pix]':<25s}|{src.fwhm_spot_up/self.pixel_scale:^9.3f}"
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