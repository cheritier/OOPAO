# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:35:18 2020

@author: cheritie
"""

import multiprocessing
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft
from joblib import Parallel, delayed
from .tools.tools import warning, OopaoError
from .Detector import Detector
from .runtime import array_backend, gpu_resident, precision_bits
xp, global_gpu_flag = array_backend()
from .runtime import backend_of as _backend_of, fft_kwargs as _fft_kwargs, stack_squeeze as _stack_squeeze, to_backend as _to_backend

if global_gpu_flag:
    fft2 = xp.fft.fft2
    ifft2 = xp.fft.ifft2
    fftshift = xp.fft.fftshift
else:
    fft2 = scipy.fft.fft2
    ifft2 = scipy.fft.ifft2
    fftshift = scipy.fft.fftshift


class BioEdge:
    def __init__(
        self,
        nSubap: float,
        telescope,
        modulation: float,
        lightRatio: float,
        postProcessing: str = "slopesMaps",
        n_pix_separation: float = 2.0,
        n_pix_edge: float = None,
        calibModulation: float = 50.0,
        extraModulationFactor: int = 0,
        binning: int = 1,
        nTheta_user_defined: int = None,
        userValidSignal: bool = None,
        grey_width: float = 0,
        grey_length: bool = False,
        delta_theta: float = 0.0,
        user_modulation_path: list = None,
        quadrants_numbering: list = [0, 1, 2, 3],
        polarization_leakage_factor: float = 0.0,
    ):
        """Bi-O Edge
        A Bi-O object consists in defining 4 2D phase mask located at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field.
        By default the Bi-O edge detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)

        Parameters
        ----------
        nSubap : float
            The number of subapertures (ie the diameter of the Pyramid Pupils in pixels).
        telescope : Telescope object
            The telescope object to which the Pyramid is associated. This object carries the phase, flux and pupil information.
        modulation : float
            The Tip-Tilt modulation in [lambda/D] where lambda is the NGS wavelength and D the telescope diameter.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        postProcessing : str, optional
            Processing of the WFS signals:
                -'fullFrame' or 'slopesMaps':
                    - Use of full detector or slopes-maps computation.
                    - Normalization is done using the mean value of the whole frame.
                    -'fullFrame_incidence_flux','slopesMaps_incidence_flux':
                    - Use of full detector or slopes-maps computation.
                    - Normalization is done using the mean value of the incident flux in photons.
                -''fullFrame_sum_flux','slopesMaps_sum_flux':
                    - Use of full detector or slopes-maps computation.
                    - Normalization is done using the sum of the incident flux in photons.
            The default is 'slopesMaps'.
        n_pix_separation : float, optional
            Number of pixels separating the Pyramid Pupils in number of pixels of the detector.
            The default is 2.
        n_pix_edge : float, optional
            number of pixel at the edge of the Pyramid Pupils in number of pixels of the detector.
            The default is None and corresponds to n_pix_separation/2.
        calibModulation : float, optional
            Defines the modulation used to select the valid subapertures.
            The default is 50.
        extraModulationFactor : int, optional
            Extra Factor to increase/reduce the number of modulation point (extraModulationFactor = 1 means 4 modulation points added, 1 for each quadrant).
            The default is 0.
        binning : int, optional
            binning factor of the PWFS detector signals.
            The default is 1.
        nTheta_user_defined : int, optional
            _ nTheta_user_defined   : user-defined number of Tip/Tilt modulation points.
            The default is None and corresponds to using the default value set by the modulation parameter.
        userValidSignal : bool, optional
            User-defined valid pixel mask for the signals computation.
            The default is None.
        grey_width: float
            half-width of the gradient area along the filtering axis for the grey Bi-O edge (Verinaud et al 2023) in [lambda/D].
        grey_length: bool = False,
            half-width of the gradient area along the orthogonal axis for the grey Bi-O edge (Verinaud et al 2023) in [lambda/D].
        delta_theta : float, optional
            delta angle for the modulation points, default value is 0 (on the edge between two sides of the Pyramid).
            The default is 0.
        user_modulation_path : list, optional
            user-defined modulation path ( a list of [x,y] coordinates in lambda/D units is expected).
            The default is None.
        Raises
        ------
        ValueError
            DESCRIPTION.
        AttributeError
            DESCRIPTION.

        Returns
        -------
            None.

        ************************** PROPAGATING THE LIGHT TO THE PYRAMID OBJECT **************************
        The light can be propagated from a telescope object tel through the Pyramid object wfs using the * operator:
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the PWFS detector (phase and flux)
            _ binning of the Pyramid signals
            _ addition of eventual photon noise and readout noise
            _ computation of the Pyramid signals

        If the tel.src object is an asterism of sources with the same wavelength, each source is propagated to the Pyramid.
        The resulting intensities are summed incoherently before being integrated by the Pyramid camera.


        ************************** PROPERTIES **************************

        The main properties of a Pyramid object are listed here:
        _ wfs.nSignal                    : the length of the signal measured by the Pyramid
        _ wfs.signal                     : signal measured by the Pyramid of length wfs.nSignal
        _ wfs.signal_2D                  : 2D map of the signal measured by the Pyramid
        _ wfs.apply_shift_wfs            : apply a tip tilt to each quadrant to move the Pyramid pupils
        _ wfs.random_state_photon_noise  : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time
        _ wfs.random_state_readout_noise : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time
        _ wfs.random_state_background    : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time
        _ wfs.fov                        : Field of View of the Pyramid in arcsec
        _ wfs.raw_data                   : Intensity pattern on the detector before its integration by a detector ("pure" WFS signal)
        _ wfs.pyramidFrame               : DEPRECATED. copy of wfs.raw_data kept for backward compatibility

        The main properties of the object can be displayed using :
            wfs.print_properties()

        the following properties can be updated on the fly:
            _ wfs.modulation            : update the modulation radius and update the reference signal
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value
        The detector noise can be set. (See Detector class for more details.)
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.cam.backgroundNoise   : Background noise can be set to True or False. An Associated wfs.cam.backgroundNoiseMap of the detector frame size must be defined

        """
        self.gpu_available = global_gpu_flag
        self.gpu_resident = gpu_resident()
        if self.gpu_available:
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
            self.nJobs = 1
            self.mempool = xp.get_default_memory_pool()
            from .tools.tools import get_gpu_memory

            self.mem_gpu = get_gpu_memory()

            print("GPU available!")
            for i in range(len(self.mem_gpu)):
                print(
                    "GPU device "
                    + str(i)
                    + " : "
                    + str(self.mem_gpu[i] / 1024)
                    + "GB memory"
                )
        else:
            self.convert_for_gpu = lambda input_matrix: input_matrix
            self.convert_for_numpy = lambda input_matrix: input_matrix

        precision = precision_bits()
        self.precision = np.float64 if precision == 64 else np.float32
        self.precision_complex = xp.complex128 if precision == 64 else xp.complex64
        # initialize the Bi-O Edge Object
        # telescope attached to the wfs
        self.telescope = telescope
        if (
            self.telescope.resolution / nSubap
        ) % 2 != 0 and self.telescope.resolution / nSubap != 1:
            raise OopaoError(
                "The resolution should be an even number and be a multiple of 2**i where i>=2"
            )
        if self.telescope.src is None:
            raise OopaoError(
                "The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel"
            )
        else:
            self.src = self.telescope.src
        # save wavelength used for the calibration of the Pyramid to avoid conflicts
        self.wavelength_calibration = self.src.wavelength
        # delta theta in degree to change the position of the modulation point (default is 0 <=> modulation point on the edge of two sides of the pyramid)
        self.delta_theta = delta_theta
        # user defined number of modulation point
        self.nTheta_user_defined = nTheta_user_defined
        # Extra Factor to increase/reduce the number of modulation point (extraModulationFactor = 1 means 4 modulation points added, 1 for each quadrant)
        self.extraModulationFactor = extraModulationFactor
        # Number of subaperture
        self.nSubap = nSubap
        # Number of pixel on the edges of the PWFS pupils
        self.edgePixel = n_pix_edge
        # Value used for the centering for the slopes-maps computation
        self.centerPixel = 0
        # type of processing of the signals (see self.postProcessing)
        self.postProcessing = postProcessing
        # user defined mask for the valid pixel selection
        self.userValidSignal = userValidSignal
        # tag for the PSF centering on  or 4 pixels
        self.psfCentering = True
        # binning factor for the detector
        self.binning = binning
        # polarization leakage factor to model the polarization leakage in the Bi-O edge masks
        self.polarization_leakage_factor = polarization_leakage_factor
        # list used for the quadrants ordering
        self.quadrants_numbering = quadrants_numbering
        # user defined modulation path
        self.user_modulation_path = user_modulation_path
        # kept for backward compatibility (the 4 masks are now propagated as a single batched operation)
        self.joblib_prefer_masks = "threads"
        self.nJobs_masks = 1
        self.weight_vector = None
        self.n_pix_separation = n_pix_separation
        self.sx = [0, 0, 0, 0]
        self.sy = [0, 0, 0, 0]
        if n_pix_edge is None:
            self.n_pix_edge = self.n_pix_separation // 2
        else:
            self.n_pix_edge = n_pix_edge
            if n_pix_edge != self.n_pix_separation // 2:
                warning(
                    "The recommanded value for n_pix_edge is "
                    + str(self.n_pix_separation // 2)
                    + " instead of "
                    + str(n_pix_edge)
                )
        if self.gpu_available:
            self.joblib_setting = "processes"
        else:
            self.joblib_setting = "threads"
        # half width of the grey area of the mask in [l/D]
        self.grey_width = grey_width  # half width of the grey area of the mask
        self.grey_length = grey_length
        # Case where the zero-padding is not specificed => taking the smallest value ensuring to get edgePixel space from the edge.
        self.resolution = int(
            (self.nSubap * 2 + self.n_pix_separation + self.n_pix_edge * 2)
            * self.telescope.resolution
            / self.nSubap
        )
        # zero-Padding Factor
        self.zeroPaddingFactor = self.resolution / self.telescope.resolution
        # zero-Padding Factor
        self.zeroPadding = (self.resolution - self.telescope.resolution) // 2
        # Tag of the object
        self.tag = "pyramid"
        # WFS detector object (see Detector class)
        self.cam = Detector(round(nSubap * self.zeroPaddingFactor))
        # WFS focal plane detector object (see Detector class)
        self.focal_plane_camera = Detector(
            int((modulation * 4 + 12) * self.zeroPaddingFactor),
            psf_sampling=self.zeroPaddingFactor,
        )
        self.focal_plane_camera.is_focal_plane_camera = True
        # Light ratio for the valid pixels selection
        self.lightRatio = lightRatio
        if calibModulation >= self.telescope.resolution / 2:
            self.calibModulation = self.telescope.resolution / 2 - 1
        else:
            # Modulation used for the valid pixel selection
            self.calibModulation = calibModulation
        # Flag for the initialization of the WFS
        self.isInitialized = False
        # Flag for the initialization of the WFS
        self.isCalibrated = False
        # delta Tip for the modulation
        self.delta_Tip = 0
        # delta Tilt for the modulation
        self.delta_Tilt = 0
        # Center of the zero-Padded array
        self.center = self.resolution // 2
        self.supportPadded = self.convert_for_gpu(
            np.pad(
                self.telescope.pupil.astype(self.precision_complex()),
                (
                    (self.zeroPadding, self.zeroPadding),
                    (self.zeroPadding, self.zeroPadding),
                ),
                "constant",
            )
        )
        # case where a spatial filter is considered
        self.spatialFilter = None
        self.rad2arcsec = (180 / xp.pi) * 3600
        self.fov = (
            self.rad2arcsec
            * self.resolution
            / self.zeroPaddingFactor
            * (self.src.wavelength / self.telescope.D)
        )  # fov in arcsec
        self.fov_l_d = self.resolution / self.zeroPaddingFactor  # fov in arcsec
        # maximum field of view for off-axis sources when propagating asterism
        self.max_fov_arcsec = self.fov / 2

        n_cpu = multiprocessing.cpu_count()
        # joblib settings for parallization
        if self.gpu_available is False:
            if n_cpu > 16:
                # number of jobs for the joblib package
                self.nJobs = 8
            else:
                self.nJobs = 6
            self.n_max = 1e9
        else:
            # quantify GPU max memory usage
            A = np.ones([self.resolution, self.resolution]) + 1j * np.ones(
                [self.resolution, self.resolution]
            )
            self.n_max = int(
                0.75 * (np.min(self.mem_gpu) / 1024) / (A.nbytes / 1024 / 1024 / 1024)
            )
            del A

        # Prepare the Tip Tilt for the modulation -- normalized to apply the modulation in terms of lambda/D
        [self.Tip, self.Tilt] = np.meshgrid(
            np.linspace(-np.pi, np.pi, self.telescope.resolution),
            np.linspace(-np.pi, np.pi, self.telescope.resolution),
        )
        self.Tilt *= (
            self.telescope.pupil
            * self.telescope.resolution
            / self.telescope.initial_resolution
        )
        self.Tip *= (
            self.telescope.pupil
            * self.telescope.resolution
            / self.telescope.initial_resolution
        )

        # compute the phasor to center the PSF on 4 pixels
        [xx, yy] = np.meshgrid(
            np.linspace(0, self.resolution - 1, self.resolution),
            np.linspace(0, self.resolution - 1, self.resolution),
        )
        # phasor for the FFT centering
        # (cast to the working precision: a complex128 phasor would silently turn every FFT into double precision)
        self.phasor = self.convert_for_gpu(
            np.exp(-(1j * np.pi * (self.resolution + 1) / self.resolution) * (xx + yy)).astype(self.precision_complex)
        )

        # Batched propagation settings
        # compute the focal-plane (modulation camera) image during each measurement; set to False to skip that work
        # when wfs.focal_plane_camera is not used (e.g. in a closed loop)
        self.compute_focal_plane = True
        # memory budget (bytes) for one batch of fields when running on the CPU
        self.cpu_batch_memory = 1e9
        # focal-plane intensity summed over the last measurement (read by wfs*wfs.focal_plane_camera)
        self.modulation_camera_intensity = None
        self._focal_plane_sum = None
        # batch size on the GPU, measured once from the free memory (reset when the modulation changes)
        self._max_batch_cache = None

        # Creating the PWFS mask
        self.mask_computation()

        # initialize the reference slopes and units
        self.slopesUnits = 1
        self.referenceSignal = 0
        self.referenceSignal_2D = 0
        self.referencePyramidFrame = 0
        # Modulation radius (in lambda/D)
        self.modulation = modulation

        # Select the valid pixels
        print("Selection of the valid pixels...")
        self.initialization()
        print("Acquisition of the reference slopes and units calibration...")
        # set the modulation radius and propagate light
        self.modulation = modulation
        self.wfs_calibration()
        self.src**self.telescope
        self.wfs_measure(phase_in=self.src.phase)
        print(self)

    def mask_computation(self, sx=[0] * 4, sy=[0] * 4):
        print("Bio-Edge Mask initialization...")
        A = np.zeros([self.resolution, self.resolution], dtype=self.precision_complex)
        n_tot = int(
            (self.nSubap * 2 + self.n_pix_separation + self.n_pix_edge * 2)
            * self.telescope.resolution
            / self.nSubap
        )

        # mask centered on 4 pixel
        lim = np.pi
        norma = (
            (self.nSubap * 2)
            * (self.telescope.resolution / self.nSubap)
            * (1 / (self.nSubap * 2))
        )
        # create a Tip/Tilt combination for each quadrant
        [Tip, Tilt] = np.meshgrid(
            np.linspace(-lim, lim, n_tot, endpoint=False),
            np.linspace(-lim, lim, n_tot, endpoint=False),
        )
        BW = np.zeros([self.resolution])
        BW[0 : self.resolution // 2] = 1.0
        r = int(
            np.round(
                self.zeroPaddingFactor
                * self.grey_width
                * self.telescope.resolution
                / self.telescope.initial_resolution
            )
        )
        if self.grey_width != 0:
            self.gray_gradient = np.hstack(
                [
                    np.linspace(1, 0.5, r, endpoint=True),
                    np.linspace(0.5, 0, r, endpoint=True),
                ]
            )
            BW[self.resolution // 2 - r : self.resolution // 2 + r] = self.gray_gradient
        [X, Y] = np.meshgrid(BW, BW)
        A = np.sqrt(X)
        if self.grey_length is not False:
            r_grey = self.zeroPaddingFactor
            r_length = int(np.round(r_grey * self.grey_length))
            A[0 : self.resolution // 2 - r_length, 0 : self.resolution // 2] = 1.0
            A[self.resolution // 2 + r_length :, 0 : self.resolution // 2] = 1.0
            A[0 : self.resolution // 2 - r_length, self.resolution // 2 :] = 0.0
            A[self.resolution // 2 + r_length :, self.resolution // 2 :] = 0.0
        A = (np.abs(A) ** 2 * (1 - self.polarization_leakage_factor)) ** 0.5
        B = np.sqrt(1 - A**2)
        C = np.copy(A.T)
        D = np.copy(B.T)
        if np.isscalar(sx):
            sx = [sx] * 4
        if np.isscalar(sy):
            sy = [sy] * 4
        if len(sx) == 4 and len(sy) == 4:
            A_TT = -sx[3] * Tip * norma + sy[3] * Tilt * norma
            B_TT = -sx[2] * Tip * norma + sy[2] * Tilt * norma
            C_TT = -sx[1] * Tip * norma + sy[1] * Tilt * norma
            D_TT = -sx[0] * Tip * norma + sy[0] * Tilt * norma
        else:
            A_TT = 0
            B_TT = 0
            C_TT = 0
            D_TT = 0
        mask_TT_ = [A_TT, B_TT, C_TT, D_TT]
        mask_0 = [A, B, C, D]
        mask_ = [
            mask_0[self.quadrants_numbering[0]],
            mask_0[self.quadrants_numbering[1]],
            mask_0[self.quadrants_numbering[2]],
            mask_0[self.quadrants_numbering[3]],
        ]
        self.mask = []
        self.mask_TT = []
        for i_m in range(4):
            self.mask.append(mask_[i_m] * np.exp(1j * mask_TT_[i_m]))
            self.mask_TT.append(mask_TT_[i_m])

        # stack the 4 masks as a (4, N, N) array on the active backend, at the working precision
        self.mask = xp.asarray(np.asarray(self.mask).astype(self.precision_complex))
        # Save a copy of the initial mask
        self.initial_mask = self.mask.copy()
        print("Done!")
        return

    def apply_shift_wfs(self, sx=None, sy=None, mis_reg=None, units="pixels"):
        if sx is None:
            sx = 0
        if sy is None:
            sy = 0
        if mis_reg is not None:
            sx = [mis_reg.dX_1, mis_reg.dX_2, mis_reg.dX_3, mis_reg.dX_4]
            sy = [mis_reg.dY_1, mis_reg.dY_2, mis_reg.dY_3, mis_reg.dY_4]
        # factor to ensure backward compatibility
        f = 2

        # normalization factor to shift the pupil with the correct units
        if units == "pixels":
            factor = f
        if units == "m":
            factor = f / (self.telescope.D / self.nSubap)

        # sx and sy are the units of displacements in pixels
        if np.isscalar(sx) and np.isscalar(sy):
            shift_x = [factor * sx] * 4
            shift_y = [factor * sy] * 4
        else:
            if len(sx) == 4 and len(sy) == 4:
                shift_x = []
                shift_y = []
                [shift_x.append(i_x * factor) for i_x in sx]
                [shift_y.append(i_y * factor) for i_y in sy]
            else:
                raise OopaoError(
                    "Wrong size for sx and/or sy, a list of 4 values is expected."
                )
        if (
            np.max(np.abs(shift_x) / factor) > self.n_pix_edge
            or np.max(np.abs(shift_y) / factor) > self.n_pix_edge
        ):
            warning(
                "The Bi-O Edge pupils have been shifted outside of the detector!"
                + "Wrapping of the signal is currently occuring!!"
            )
        self.mask_computation(sx=sx, sy=sy)
        self.sx = np.asarray(shift_x) / factor
        self.sy = np.asarray(shift_y) / factor
        self.slopesUnits = 1
        self.referenceSignal = 0
        self.referenceSignal_2D = 0
        self.wfs_calibration()
        return

    def initialization(self):
        self.src**self.telescope
        if self.userValidSignal is None:
            if self.lightRatio == 0:
                self.cam.frame = np.ones(
                    [self.cam.resolution * 2, self.cam.resolution * 2]
                )
            else:
                print("The valid pixel are selected on flux considerations")
                # set the modulation to a large value
                self.modulation = self.calibModulation
                self.wfs_measure(phase_in=self.src.phase)
            # save initialization frame
            self.initFrame = self.cam.frame

            # save the number of signals depending on the case
            if self.postProcessing[:10] == "slopesMaps":
                # select the valid pixels of the detector according to the flux (case slopes-maps)
                I1 = self.grabQuadrant(1)
                I2 = self.grabQuadrant(2)
                I3 = self.grabQuadrant(3)
                I4 = self.grabQuadrant(4)

                # sum of the 4 quadrants
                self.I4Q = I1 + I2 + I3 + I4
                # valid pixels to consider for the slopes-maps computation
                self.validI4Q = self.I4Q >= self.lightRatio * self.I4Q.max()
                self.validSignal = np.concatenate((self.validI4Q, self.validI4Q))
                self.nSignal = int(np.sum(self.validSignal))

            if self.postProcessing[:9] == "fullFrame":
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validSignal = (
                    self.initFrame >= self.lightRatio * self.initFrame.max()
                )
                self.nSignal = int(np.sum(self.validSignal))
        else:
            print(
                "You are using a user-defined mask for the selection of the valid pixel"
            )
            if self.postProcessing[:10] == "slopesMaps":
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validI4Q = self.userValidSignal
                self.validSignal = np.concatenate((self.validI4Q, self.validI4Q))
                self.nSignal = int(np.sum(self.validSignal))
            if self.postProcessing[:9] == "fullFrame":
                self.validSignal = self.userValidSignal
                self.nSignal = int(np.sum(self.validSignal))
        # Tag to indicate that the wfs is initialized
        self.isInitialized = True
        return

    def wfs_calibration(self):
        # save current OPD to be re-applied after calibration:
        tmp_OPD = self.src.OPD.copy()
        # reference slopes acquisition
        self.src**self.telescope
        # compute the refrence signals
        self.relay(self.src)
        reference_2D, reference = self.signalProcessing()
        # the references are kept on the CPU (signalProcessing keeps a GPU copy when it needs one)
        self.referenceSignal_2D = _to_backend(reference_2D, np)
        self.referenceSignal = _to_backend(reference, np)

        # 2D reference Frame before binning with detector
        self.referencePyramidFrame = np.copy(_to_backend(self.raw_data, np))
        if self.isCalibrated is False:
            print("WFS calibrated!")
        self.isCalibrated = True
        # re-applied the initial OPD
        self.src.OPD = tmp_OPD
        return

    def convolution(self, mask, em_field_ft):
        """Detector-plane intensity for one mask and one focal-plane field (kept for backward compatibility).

        wfs_measure no longer calls this method: the 4 masks are applied in _bio_edge_transform_block.
        """
        intensity = xp.abs(ifft2(em_field_ft * mask))
        xp.square(intensity, out=intensity)
        return intensity

    def bio_edge_transform(self, phase_in):
        """Propagate a single phase screen through the Bi-O Edge (kept for backward compatibility).

        wfs_measure no longer calls this method: it uses the batched path (_bio_edge_transform_batch).
        Returns the (2N, 2N) detector-plane intensity with the 4 pupils tiled 2x2.
        """
        phase_in = xp.asarray(phase_in, dtype=self.precision)
        # em field corresponding to phase_in
        if np.ndim(self.telescope.OPD) == 2 and self.modulation != 0:
            phase_in = xp.asarray(self.src.phase, dtype=self.precision) + phase_in
        em_field = self.precision(np.sqrt(0.5)) * self.maskAmplitude * xp.exp(1j * phase_in)
        return self._bio_edge_transform_batch(em_field[None])[0]

    def _bio_edge_transform_block(self, fields, workers=-1):
        """Propagate a (B, n, n) stack of pupil-plane EM fields through the 4 Bi-O Edge masks.

        Returns the (B, 2N, 2N) intensities in the detector plane (the 4 pupils tiled 2x2, in mask order) and,
        when compute_focal_plane is set, the (N, N) focal-plane intensity summed over the stack (None otherwise).
        """
        n = self.telescope.resolution
        N = self.resolution
        pupil = slice(self.center - n // 2, self.center + n // 2)
        fft_kw = _fft_kwargs(workers)
        # zero-padding for the FFT computation
        support = xp.zeros((fields.shape[0], N, N), dtype=self.precision_complex)
        support[:, pupil, pupil] = fields
        if self.psfCentering:
            # case with mask centered on 4 pixels
            support *= self.phasor
            em_field_ft = fft2(support, **fft_kw)
        else:
            # case with mask centered on 1 pixel
            em_field_ft = fftshift(fft2(support, **fft_kw), axes=(-2, -1))
        del support
        focal_plane = None
        if self.compute_focal_plane:
            focal_plane = xp.abs(em_field_ft)
            xp.square(focal_plane, out=focal_plane)
            focal_plane = focal_plane.sum(axis=0)
        # Fourier filtering by each of the 4 masks and propagation to the detector plane.
        # The focal-plane field is computed once and shared by the 4 masks.
        intensity = xp.empty((fields.shape[0], 2 * N, 2 * N), dtype=self.precision)
        for k in range(4):
            i, j = divmod(k, 2)
            quadrant = xp.abs(ifft2(em_field_ft * self.mask[k], **fft_kw))
            xp.square(quadrant, out=quadrant)
            intensity[:, i * N:(i + 1) * N, j * N:(j + 1) * N] = quadrant
            del quadrant
        return intensity, focal_plane

    def _bio_edge_transform_batch(self, fields):
        """Propagate a (B, n, n) stack of pupil-plane EM fields; returns the (B, 2N, 2N) detector-plane intensities.

        On the GPU the whole stack is processed in one batched call. On the CPU it is split into nJobs
        blocks run by threads (NumPy and SciPy release the GIL), each using single-threaded FFTs.
        The focal-plane intensity is added to self._focal_plane_sum when it is being accumulated.
        """
        n_fields = fields.shape[0]
        n_blocks = 1 if self.gpu_available else max(1, min(self.nJobs, n_fields))
        if n_blocks == 1:
            results = [self._bio_edge_transform_block(fields, workers=-1)]
        else:
            edges = np.linspace(0, n_fields, n_blocks + 1).astype(int)
            results = Parallel(n_jobs=n_blocks, prefer='threads')(
                delayed(self._bio_edge_transform_block)(fields[a:b], 1) for a, b in zip(edges[:-1], edges[1:]))
        if self._focal_plane_sum is not None:
            for _, focal_plane in results:
                if focal_plane is not None:
                    self._focal_plane_sum += focal_plane
        if len(results) == 1:
            return results[0][0]
        return xp.concatenate([intensity for intensity, _ in results], axis=0)

    def _max_batch(self):
        """Number of fields that can be propagated in one batch with the memory available."""
        # peak usage is about 8 complex (N, N) arrays per field (padded field, focal-plane field, masked field,
        # inverse FFT, the (2N, 2N) real output and FFT workspace)
        item_bytes = 8 * self.resolution**2 * np.dtype(self.precision_complex).itemsize
        if self.gpu_available:
            # the memory query is a driver call: do it once and reuse the result
            if self._max_batch_cache is None:
                free_bytes = xp.cuda.runtime.memGetInfo()[0] + xp.get_default_memory_pool().free_bytes()
                self._max_batch_cache = int(max(1, min(self.n_max, 0.5 * free_bytes // item_bytes)))
            return self._max_batch_cache
        return int(max(1, min(self.n_max, self.cpu_batch_memory // item_bytes)))

    def _modulation_phasors(self, start, stop):
        """exp(1j*TT) for modulation points [start:stop], as a (stop-start, n, n) array on the active backend."""
        if self.modulation_phasors is not None:
            return self.modulation_phasors[start:stop]
        tip_tilt = xp.asarray(self.phaseBuffModulationLowres[start:stop], dtype=self.precision)
        return xp.exp(1j * tip_tilt)

    def _modulated_frames(self, em_fields, weights=None):
        """Bi-O Edge intensities summed over the modulation points, for each field of a (m, n, n) stack.

        Modulation point i multiplies the field by exp(1j*TT_i). Fields and modulation points are propagated
        together in batches sized from the memory available. If weights (length nTheta) are given, the
        weighted sum is returned. Returns a (m, 2N, 2N) array on the active backend.
        """
        n_fields, n = em_fields.shape[0], self.telescope.resolution
        N2 = 2 * self.resolution
        batch = self._max_batch()
        # modulation points per batch, and fields per batch (1 if a full modulation cycle does not fit)
        theta_step = min(self.nTheta, batch)
        field_step = max(1, batch // self.nTheta)
        frames = xp.zeros((n_fields, N2, N2), dtype=self.precision)
        for f0 in range(0, n_fields, field_step):
            f1 = min(f0 + field_step, n_fields)
            for t0 in range(0, self.nTheta, theta_step):
                t1 = min(t0 + theta_step, self.nTheta)
                fields = em_fields[f0:f1, None] * self._modulation_phasors(t0, t1)[None]
                intensity = self._bio_edge_transform_batch(fields.reshape(-1, n, n))
                del fields
                intensity = intensity.reshape(f1 - f0, t1 - t0, N2, N2)
                if weights is None:
                    frames[f0:f1] += intensity.sum(axis=1)
                else:
                    frames[f0:f1] += xp.tensordot(intensity, weights[t0:t1], axes=([1], [0]))
                del intensity
        return frames

    def _phase_modes(self, phase, start, stop):
        """Phase screens [start:stop] of a (n, n, n_modes) cube, as a (stop-start, n, n) array on the active backend."""
        chunk = phase[:, :, start:stop]
        backend = _backend_of(chunk)
        # reorder and cast before any transfer to the GPU
        chunk = backend.ascontiguousarray(backend.moveaxis(chunk, -1, 0), dtype=self.precision)
        return xp.asarray(chunk)

    def _spatial_filter_diagnostics(self, phase):
        """Store the focal-plane field and the spatially filtered pupil-plane field (for inspection only)."""
        n = self.telescope.resolution
        pupil = slice(self.center - n // 2, self.center + n // 2)
        support = xp.zeros((self.resolution, self.resolution), dtype=self.precision_complex)
        support[pupil, pupil] = self.maskAmplitude * xp.exp(1j * xp.asarray(phase, dtype=self.precision))
        em_field_ft = fft2(support * self.phasor)
        self.em_field_spatial_filter = self.convert_for_numpy(em_field_ft)
        self.pupil_plane_spatial_filter = self.convert_for_numpy(ifft2(em_field_ft * self._spatial_filter_xp))

    def _store_focal_plane(self):
        """Keep the accumulated focal-plane intensity for wfs*wfs.focal_plane_camera."""
        if self._focal_plane_sum is None:
            self.modulation_camera_intensity = None
            return
        # factor 4: same scaling as the previous implementation, which stored the focal-plane field once per mask
        intensity = 4 * self._focal_plane_sum
        self.modulation_camera_intensity = intensity if self.gpu_resident and self.isCalibrated else self.convert_for_numpy(intensity)
        self._focal_plane_sum = None

    def _release_gpu_memory(self):
        """Give cached GPU memory (FFT plans, memory-pool blocks) back to the device after a large batched run."""
        if self.gpu_available:
            self._max_batch_cache = None
            try:
                xp.fft.config.get_plan_cache().clear()
                xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                warning("could not free the memory")

    def wfs_integrate(self):
        # propagate to the detector to apply the noise
        self * self.cam
        if self.isInitialized and self.isCalibrated:
            signal_2D, signal = self.signalProcessing()
            return signal_2D, signal
        else:
            return None, None

    def setPhaseBuffer(self, phaseIn):
        B = self.phaseBuffModulationLowres_CPU + phaseIn
        return B

    def wfs_measure(self, phase_in=None, integrate=True):
        if self.isInitialized and self.isCalibrated:
            if self.wavelength_calibration != self.src.wavelength:
                raise OopaoError(
                    "A change in wavelength was detected in the WFS object \n"
                    + "Make sure that the correct source is propagated in the WFS object or re-calibrate with the correct source."
                )
        if phase_in is not None:
            self.src.phase = phase_in
        phase = self.src.phase
        if np.ndim(phase) not in (2, 3):
            raise OopaoError("Wrong dimension for the input phase. Aborting")
        modulated = not (self.modulation == 0 and self.user_modulation_path is None)
        # mask amplitude for the light propagation
        # src.intensity = fluxMap * scintillation, and fluxMap already includes tel.pupilReflectivity
        # (Telescope.relay), so this matches the Pyramid and the Shack-Hartmann
        self.maskAmplitude = xp.sqrt(xp.asarray(self.src.intensity, dtype=self.precision) / self.nTheta)
        # the light is split between the two polarization channels of the Bi-O Edge
        amplitude = self.precision(np.sqrt(0.5)) * self.maskAmplitude

        if self.spatialFilter is not None and np.ndim(phase) == 2:
            self._spatial_filter_diagnostics(phase)
        # focal-plane (modulation camera) intensity, accumulated during the propagation
        if self.compute_focal_plane:
            self._focal_plane_sum = xp.zeros((self.resolution, self.resolution), dtype=self.precision)
        else:
            self._focal_plane_sum = None

        if np.ndim(phase) == 2:
            # single phase screen: all modulation points in as few batched calls as memory allows
            em_field = amplitude * xp.exp(1j * xp.asarray(phase, dtype=self.precision))
            if modulated:
                weights = None if self.weight_vector is None else xp.asarray(self.weight_vector, dtype=self.precision)
                frame = self._modulated_frames(em_field[None], weights)[0]
                if weights is not None:
                    frame /= self.nTheta
            else:
                frame = self._bio_edge_transform_batch(em_field[None])[0]
            # with GPU residency the frame stays on the GPU for the detector and the signal processing
            self.raw_data = frame if self.gpu_resident and self.isCalibrated else self.convert_for_numpy(frame)
            self._store_focal_plane()
            if integrate:
                self.signal_2D, self.signal = self.wfs_integrate()
        else:
            # cube of phase screens (e.g. interaction matrix): processed by chunks of modes
            n_modes = phase.shape[2]
            modes_per_batch = max(1, self._max_batch() // self.nTheta)
            self.signal_2D = np.zeros([self.validSignal.shape[0], self.validSignal.shape[1], n_modes])
            self.signal = np.zeros([self.nSignal, n_modes])
            for start in range(0, n_modes, modes_per_batch):
                stop = min(start + modes_per_batch, n_modes)
                em_fields = amplitude * xp.exp(1j * self._phase_modes(phase, start, stop))
                if modulated:
                    frames = self._modulated_frames(em_fields)
                else:
                    frames = self._bio_edge_transform_batch(em_fields)
                del em_fields
                frames = self.convert_for_numpy(frames)
                for i in range(stop - start):
                    self.raw_data = frames[i]
                    if integrate:
                        self.signal_2D[:, :, start + i], self.signal[:, start + i] = self.wfs_integrate()
                del frames
            self._store_focal_plane()
            self._release_gpu_memory()
        return

    def _processing_arrays(self, backend):
        """Valid-pixel masks, their flat indices and the 2D reference signal on `backend`.

        Boolean-mask indexing (a[mask]) forces a GPU synchronization because the size of the result
        must be read back; integer indices computed once avoid it. The arrays are cached and rebuilt
        whenever validI4Q, validSignal or referenceSignal_2D is replaced (new calibration, lightRatio...).
        """
        valid_I4Q = getattr(self, "validI4Q", None)
        sources = (valid_I4Q, self.validSignal, self.referenceSignal_2D)
        cache = getattr(self, "_processing_cache", None)
        on_gpu = backend is not np
        if cache is not None and cache["on_gpu"] == on_gpu and all(a is b for a, b in zip(cache["sources"], sources)):
            return cache["arrays"]
        arrays = {"reference_2D": _to_backend(self.referenceSignal_2D, backend),
                  "signal_index": _to_backend(np.flatnonzero(_to_backend(self.validSignal, np) == 1), backend)}
        if valid_I4Q is not None:
            arrays["valid_I4Q"] = _to_backend(valid_I4Q, backend)
            arrays["I4Q_index"] = _to_backend(np.flatnonzero(_to_backend(valid_I4Q, np)), backend)
        self._processing_cache = {"on_gpu": on_gpu, "sources": sources, "arrays": arrays}
        return arrays

    def signalProcessing(self, cameraFrame=None):
        # runs on the backend of the camera frame (NumPy, or CuPy with GPU residency) without read-backs
        if cameraFrame is None:
            cameraFrame = self.cam.frame
        backend = _backend_of(cameraFrame)
        arrays = self._processing_arrays(backend)

        if self.postProcessing in ("slopesMaps", "slopesMaps_incidence_flux", "slopesMaps_camera_flux"):
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=cameraFrame) * arrays["valid_I4Q"]
            I2 = self.grabQuadrant(2, cameraFrame=cameraFrame) * arrays["valid_I4Q"]
            I3 = self.grabQuadrant(3, cameraFrame=cameraFrame) * arrays["valid_I4Q"]
            I4 = self.grabQuadrant(4, cameraFrame=cameraFrame) * arrays["valid_I4Q"]
            # global normalisation
            if self.postProcessing == "slopesMaps":
                I4Q = I1 + I2 + I3 + I4
                self.norma = I4Q.ravel()[arrays["I4Q_index"]].mean()
            elif self.postProcessing == "slopesMaps_incidence_flux":
                subArea = (self.telescope.D / self.nSubap) ** 2
                self.norma = np.float64(self.src.nPhoton * self.telescope.samplingTime * subArea)
            else:
                self.norma = self.cam.frame.mean().astype(np.float64)
            # slopesMaps computation cropped to the valid pixels
            Sx = I1 - I2 + I4 - I3
            Sy = I1 - I4 + I2 - I3
            # 2D slopes maps
            slopesMaps = (backend.concatenate((Sx, Sy)) / self.norma - arrays["reference_2D"]) * self.slopesUnits
            # slopes vector
            slopes = slopesMaps.ravel()[arrays["signal_index"]]
            return slopesMaps, slopes

        if self.postProcessing in ("fullFrame_camera_flux", "fullFrame_incidence_flux", "fullFrame_sum_flux", "fullFrame"):
            # global normalization
            if self.postProcessing == "fullFrame_camera_flux":
                self.norma = self.cam.frame.mean().astype(np.float64)
            elif self.postProcessing == "fullFrame_incidence_flux":
                subArea = (self.telescope.D / self.nSubap) ** 2
                self.norma = np.float64(self.src.nPhoton * self.telescope.samplingTime * subArea) / 4
            elif self.postProcessing == "fullFrame_sum_flux":
                self.norma = self.cam.frame.sum().astype(np.float64)
            else:
                self.norma = cameraFrame.ravel()[arrays["signal_index"]].sum()
            # 2D full-frame
            fullFrameMaps = (cameraFrame / self.norma) - arrays["reference_2D"]
            # full-frame vector
            fullFrame = fullFrameMaps.ravel()[arrays["signal_index"]]
            return fullFrameMaps, fullFrame

    def get_modulation_frame(self, radius=6, norma=True):
        if radius <= 0:
            warning(
                "radius for the field of view must be a strictly positive number. Ignoring the input value."
            )
            radius = self.telescope.resolution // 2
        self.modulation_camera_frame = self.convert_for_numpy(self.focal_plane_camera.frame).astype(float)
        N_trunc = int(self.resolution / 2 - radius * self.zeroPaddingFactor)
        if N_trunc <= 0:
            warning(
                "radius Value is too high as the field of view is limited to "
                + str(int(self.fov_l_d / 2))
                + " lambda/D -- ignoring"
            )
            modulation_camera_frame_zoom = self.modulation_camera_frame.copy()
        else:
            modulation_camera_frame_zoom = self.modulation_camera_frame[
                N_trunc:-N_trunc, N_trunc:-N_trunc
            ]

        if norma:
            modulation_camera_frame_zoom /= modulation_camera_frame_zoom.max()

        return modulation_camera_frame_zoom

    def grabQuadrant(self, n, cameraFrame=None):
        centerPixel = int(np.round((self.cam.resolution / self.binning) / 2))
        n_pixels = centerPixel
        if cameraFrame is None:
            cameraFrame = self.cam.frame
        if n == 4:
            quadrant = cameraFrame[
                centerPixel : (centerPixel + n_pixels),
                centerPixel : (centerPixel + n_pixels),
            ]
        if n == 3:
            quadrant = cameraFrame[
                centerPixel : (centerPixel + n_pixels),
                centerPixel - n_pixels : (centerPixel),
            ]
        if n == 1:
            quadrant = cameraFrame[
                centerPixel - n_pixels : (+centerPixel),
                centerPixel - n_pixels : (centerPixel),
            ]
        if n == 2:
            quadrant = cameraFrame[
                centerPixel - n_pixels : (+centerPixel),
                centerPixel : (centerPixel + n_pixels),
            ]
        return quadrant

    def grabFullQuadrant(self, n, cameraFrame=None):

        if cameraFrame is None:
            cameraFrame = self.cam.frame.copy()
        n_tot = cameraFrame.shape[0]

        if n == 4:
            quadrant = cameraFrame[: n_tot // 2, : n_tot // 2]
        if n == 3:
            quadrant = cameraFrame[: n_tot // 2, -n_tot // 2 :]
        if n == 1:
            quadrant = cameraFrame[-n_tot // 2 :, -n_tot // 2 :]
        if n == 2:
            quadrant = cameraFrame[-n_tot // 2 :, : n_tot // 2]
        return quadrant

    @property
    def lightRatio(self):
        return self._lightRatio

    @lightRatio.setter
    def lightRatio(self, val):
        self._lightRatio = val
        if hasattr(self, "isInitialized"):
            if self.isInitialized:
                print("Updating the map if valid pixels ...")
                self.validI4Q = self.I4Q >= self._lightRatio * self.I4Q.max()
                self.validSignal = np.concatenate((self.validI4Q, self.validI4Q))
                self.validPix = self.initFrame >= self.lightRatio * self.initFrame.max()

                # save the number of signals depending on the case
                if self.postProcessing[:10] == "slopesMaps":
                    self.nSignal = np.sum(self.validSignal)
                    # display
                    xPix, yPix = np.where(self.validI4Q == 1)
                    plt.figure()
                    plt.imshow(self.I4Q.T)
                    plt.plot(xPix, yPix, "+")
                if self.postProcessing[:9] == "fullFrame":
                    self.nSignal = np.sum(self.validPix)
                print("Done!")

    @property
    def spatialFilter(self):
        return self._spatialFilter

    @spatialFilter.setter
    def spatialFilter(self, val):
        self._spatialFilter = val
        # copy of the filter on the active backend, at the working precision
        self._spatial_filter_xp = None if val is None else xp.asarray(val).astype(self.precision_complex)
        if self.isInitialized:
            if val is None:
                print("No spatial filter considered")
                self.mask = self.initial_mask
                if self.isCalibrated:
                    print(
                        "Updating the reference slopes and Wavelength Calibration for the new modulation..."
                    )
                    self.slopesUnits = 1
                    self.referenceSignal = 0
                    self.referenceSignal_2D = 0
                    self.wfs_calibration()
                    print("Done!")
            else:
                # a (N, N) filter is applied to the 4 masks, a (4, N, N) filter mask by mask
                if val.shape == self.mask.shape or val.shape == self.mask.shape[1:]:
                    print("A spatial filter is now considered")
                    self.mask = self.initial_mask * self._spatial_filter_xp
                    mask_cpu = self.convert_for_numpy(xp.real(self.mask))
                    plt.figure()
                    plt.imshow(np.block([[mask_cpu[0], mask_cpu[1]], [mask_cpu[2], mask_cpu[3]]]))
                    plt.title("Spatial Filter considered")
                    if self.isCalibrated:
                        print(
                            "Updating the reference slopes and Wavelength Calibration for the new modulation..."
                        )
                        self.slopesUnits = 1
                        self.referenceSignal = 0
                        self.referenceSignal_2D = 0
                        self.wfs_calibration()
                        print("Done!")
                else:
                    warning(
                        "Wrong shape for the spatial filter. No spatial filter attached to the mask"
                    )
                    self.mask = self.initial_mask

    @property
    def delta_Tip(self):
        return self._delta_Tip

    @delta_Tip.setter
    def delta_Tip(self, val):
        self._delta_Tip = val
        if self.isCalibrated:
            self.modulation = self.modulation

    @property
    def delta_Tilt(self):
        return self._delta_Tilt

    @delta_Tilt.setter
    def delta_Tilt(self, val):
        self._delta_Tilt = val
        if self.isCalibrated:
            self.modulation = self.modulation

    @property
    def validSignal(self):
        return self._validSignal

    @validSignal.setter
    def validSignal(self, val):
        self._validSignal = val
        self.valid_signal_2D = val

    @property
    def modulation(self):
        return self._modulation

    @modulation.setter
    def modulation(self, val):
        self._modulation = val
        if self._modulation >= (self.telescope.resolution // 2):
            raise OopaoError(
                "Error the modulation radius is too large for this resolution!"
                + "Consider using a larger telescope resolution!"
            )
        if val != 0 or self.user_modulation_path is not None:
            self.modulation_path = []
            if self.user_modulation_path is not None:
                self.modulation_path = self.user_modulation_path
                self.nTheta = len(self.user_modulation_path)
            else:
                # define the modulation points
                perimeter = np.pi * 2 * self._modulation
                if self.nTheta_user_defined is None:
                    self.nTheta = 4 * int(
                        (self.extraModulationFactor + np.ceil(perimeter / 4))
                    )
                else:
                    self.nTheta = self.nTheta_user_defined
                self.thetaModulation = np.linspace(
                    0 + self.delta_theta,
                    2 * np.pi + self.delta_theta,
                    self.nTheta,
                    endpoint=False,
                )
                for i in range(self.nTheta):
                    dTheta = self.thetaModulation[i]
                    self.modulation_path.append(
                        [
                            self.modulation * np.cos(dTheta) + self.delta_Tip,
                            self.modulation * np.sin(dTheta) + self.delta_Tilt,
                        ]
                    )
            self.phaseBuffModulation = np.zeros(
                [self.nTheta, self.resolution, self.resolution]
            ).astype(xp.float32)
            self.phaseBuffModulationLowres = np.zeros(
                [self.nTheta, self.telescope.resolution, self.telescope.resolution]
            ).astype(xp.float32)
            for i in range(self.nTheta):
                self.TT = (
                    self.modulation_path[i][0] * self.Tip
                    + self.modulation_path[i][1] * self.Tilt
                ) * self.telescope.pupil
                self.phaseBuffModulation[
                    i,
                    self.center
                    - self.telescope.resolution // 2 : self.center
                    + self.telescope.resolution // 2,
                    self.center
                    - self.telescope.resolution // 2 : self.center
                    + self.telescope.resolution // 2,
                ] = self.TT
                self.phaseBuffModulationLowres[i, :, :] = self.TT
            self.phaseBuffModulationLowres_CPU = self.phaseBuffModulationLowres.copy()
            if self.gpu_available:
                if self.nTheta <= self.n_max:
                    self.phaseBuffModulationLowres = self.convert_for_gpu(
                        self.phaseBuffModulationLowres
                    )
            # modulation phasors exp(1j*TT), computed once per modulation change
            # (if they do not fit in memory they are computed on the fly, batch by batch)
            self.modulation_phasors = None
            if self.nTheta <= self.n_max:
                self.modulation_phasors = xp.exp(1j * xp.asarray(self.phaseBuffModulationLowres, dtype=self.precision))
        else:
            self.nTheta = 1
            self.modulation_phasors = None
        # the memory in use has changed: measure the batch size again at the next propagation
        self._max_batch_cache = None

        if hasattr(self, "isCalibrated"):
            if self.isCalibrated:
                print(
                    "Updating the reference slopes and Wavelength Calibration for the new modulation"
                )
                self.slopesUnits = 1
                self.referenceSignal = 0
                self.referenceSignal_2D = 0
                self.wfs_calibration()
                print("Done!")

    def relay(self, src):
        if src.tag == "source":
            src_list = [src]
        elif src.tag == "asterism":
            src_list = src.src
        signal_2D_list = []
        signal_list = []
        frames_list = []

        for src in src_list:
            src.optical_path.append([self.tag, self])
            self.src = src
            self.wfs_measure(phase_in=self.src.phase)
            signal_2D_list.append(self.signal_2D)
            signal_list.append(self.signal)
            frames_list.append(self.cam.frame)

        self.signal_2D = _stack_squeeze(signal_2D_list)
        self.signal = _stack_squeeze(signal_list)
        self.frames = _stack_squeeze(frames_list)
        return

    def __mul__(self, obj):
        if obj.tag == "detector":
            if getattr(obj, "is_focal_plane_camera", False) and self.modulation_camera_intensity is None:
                raise OopaoError("The focal-plane image was not computed. Set wfs.compute_focal_plane = True and propagate the light again.")
            obj._integrated_time += self.telescope.samplingTime
            try:
                if obj.is_focal_plane_camera:
                    camera_backend = xp if self.gpu_resident else np
                    intensity = camera_backend.asarray(self.modulation_camera_intensity)
                    if obj.resolution > self.resolution:
                        frame = intensity
                        warning(
                            "Maximum resolution for focal plane camera is %i, cropping field to this dimension"
                            % self.resolution
                        )
                    else:
                        frame = intensity[
                            intensity.shape[0] // 2
                            - obj.resolution // 2 : intensity.shape[0] // 2
                            + obj.resolution // 2,
                            intensity.shape[0] // 2
                            - obj.resolution // 2 : intensity.shape[0] // 2
                            + obj.resolution // 2,
                        ]
                else:
                    raise OopaoError
            except:
                intensity = self.raw_data
                frame = obj.set_binning(intensity, self.resolution / obj.resolution)
            if self.binning != 1:
                try:
                    frame = obj.rebin(
                        frame,
                        (
                            obj.resolution // self.binning,
                            obj.resolution // self.binning,
                        ),
                    )
                except:
                    warning(
                        "The shape of the detector ("
                        + str(obj.frame.shape)
                        + ")"
                        + "is not valid with the binning value requested:"
                        + str(self.binning)
                        + "! -- Ignoring the binning."
                    )
            obj.integrate(frame)
        else:
            raise OopaoError("Error light propagated to the wrong type of object")

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop["pupil_diameter"] = f"{'Pupil diameter [px]':<25s}|{self.nSubap:^9d}"
        self.prop["pupil_separation"] = (
            f"{'Pupil separation [px]':<25s}|{self.n_pix_separation:^9.2f}"
        )
        self.prop["fov"] = f"{'Field of view [arcsec]':<25s}|{self.fov:^9.2f}"
        self.prop["modulation"] = (
            f"{'Modulation radius [l/D]':<25s}|{self.modulation:^9.1f}"
        )
        self.prop["psf_sampling"] = (
            f"{'PSF sampling [px/(l/D)]':<25s}|{self.zeroPaddingFactor:^9.2f}"
        )
        self.prop["psf_centering"] = (
            f"{'PSF centering':<25s}|{str(self.psfCentering):^9s}"
        )
        self.prop["n_valid_pixels"] = f"{'Valid pixels':<25s}|{self.nSignal:^9.0f}"
        self.prop["post_processing"] = (
            f"{'Post processing':<25s}|{self.postProcessing:^9s}"
        )
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + "\n"
        title = f'\n{" Bi-O Edge WFS ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table