# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:35:18 2020

@author: cheritie
"""
import multiprocessing
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
import scipy
from joblib import Parallel, delayed
from .tools.tools import warning, OopaoError

from .Detector import Detector
try:
    import cupy as xp
    fft2 = xp.fft.fft2
except:
    import numpy as xp
    fft2 = scipy.fft.fft2


class Pyramid:
    def __init__(self,
                 nSubap: float,
                 telescope,
                 modulation: float,
                 lightRatio: float,
                 postProcessing: str = 'slopesMaps',
                 psfCentering: bool = True,
                 n_pix_separation: float = 2.,
                 calibModulation: float = 50.,
                 n_pix_edge: float = None,
                 extraModulationFactor: int = 0,
                 binning: int = 1,
                 nTheta_user_defined: int = None,
                 userValidSignal: bool = None,
                 old_mask: bool = False,
                 rooftop: str = None,
                 delta_theta: float = 0.,
                 user_modulation_path: list = None,
                 pupilSeparationRatio: float = None,
                 edgePixel: int = None,
                 zeroPadding: int = None):
        """ PYRAMID
        A Pyramid object consists in defining a 2D phase mask located at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field.
        By default the Pyramid detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)

        Parameters
        ----------
        nSubap : float
            The number of subapertures (ie the diameter of the Pyramid Pupils in pixels).
        telescope : TYPE
            The telescope object to which the Pyramid is associated. This object carries the phase, flux and pupil information.
        modulation : float
            The Tip-Tilt modulation in [lambda/D] where lambda is the NGS wavelength and D the telescope diameter.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        postProcessing : str, optional
            Processing of the WFS signals ('fullFrame','slopesMaps','fullFrame_incidence_flux','slopesMaps_incidence_flux').
            The default is 'slopesMaps'.
        psfCentering : bool, optional
            If False, the Pyramid mask is centered on 1 pixel, if True, the Pyramid mask is centered on 4 pixels -- default value is True.
            The default is True.
        n_pix_separation : float, optional
            Number of pixels separating the Pyramid Pupils in number of pixels of the detector.
            The default is 2.
        calibModulation : float, optional
            Defines the modulation used to select the valid subapertures.
            The default is 50.
        n_pix_edge : float, optional
            number of pixel at the edge of the Pyramid Pupils in number of pixels of the detector.
            The default is None and corresponds to n_pix_separation/2.
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
        old_mask : bool, optional
            DEPRECATED -- Flag to use the old pyramid mask.
            The default is False.
        rooftop : str, optional
            If different to None, allows to compute a two-sided Pyramid ("V" corresponds to a vertical split, "H" corresponds to an horizontal split).
            The default is None.
        delta_theta : float, optional
            delta angle for the modulation points, default value is 0 (on the edge between two sides of the Pyramid).
            The default is 0.
        user_modulation_path : list, optional
            user-defined modulation path ( a list of [x,y] coordinates in lambda/D units is expected).
            The default is None.
        pupilSeparationRatio : float, optional
            DEPRECTATED -- Separation ratio of the PWFS pupils (Diameter/Distance Center to Center) -- DEPRECATED -> use n_pix_separation instead).
            The default is None.
        edgePixel : int, optional
            DEPRECATED -- number of pixel at the edge of the Pyramid Pupils.
            The default is None.
        zeroPadding : int, optional
            DEPRECATED -- User-defined zero-padding value in pixels that will be added to each side of the arrays.
            Consider using the n_pix_edge parameter that allows to do the same thing.
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
        try:
            import cupy as xp
            self.gpu_available = True
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
            self.nJobs = 1
            self.mempool = xp.get_default_memory_pool()
            from .tools.tools import get_gpu_memory
            self.mem_gpu = get_gpu_memory()

            print('GPU available!')
            for i in range(len(self.mem_gpu)):
                print('GPU device '+str(i)+' : ' +
                      str(self.mem_gpu[i]/1024) + 'GB memory')
        except:
            import numpy as xp

            def no_function(input_matrix):
                return input_matrix
            self.gpu_available = False
            self.convert_for_gpu = no_function
            self.convert_for_numpy = no_function

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
        # initialize the Pyramid Object
        # telescope attached to the wfs
        self.telescope = telescope
        if (self.telescope.resolution/nSubap) % 2 != 0 and self.telescope.resolution/nSubap != 1:
            raise OopaoError('The resolution should be an even number and be a multiple of 2**i where i>=2')
        if self.telescope.src is None:
            raise OopaoError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
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
        self.psfCentering = psfCentering
        # binning factor for the detector
        self.binning = binning
        self.old_mask = old_mask
        # user defined modulation path
        self.user_modulation_path = user_modulation_path
        # Separation ratio of the PWFS pupils (Diameter/Distance Center to Center) -- DEPRECATED -> use n_pix_separation instead)
        self.pupilSeparationRatio = pupilSeparationRatio
        self.weight_vector = None
        if edgePixel is not None:
            raise OopaoError('The use of the edgePixel property has been deprecated. Consider using n_pix_edge instead')
        if pupilSeparationRatio is not None:
            raise OopaoError('The use of the pupilSeparationRatio property has been deprecated. Consider using n_pix_separation instead')
        else:
            self.n_pix_separation = n_pix_separation
            self.sx = [0, 0, 0, 0]
            self.sy = [0, 0, 0, 0]
        if n_pix_edge is None:
            self.n_pix_edge = self.n_pix_separation//2
        else:
            self.n_pix_edge = n_pix_edge
            if n_pix_edge != self.n_pix_separation//2:
                warning('The recommanded value for n_pix_edge is ' +
                        str(self.n_pix_separation//2) + ' instead of ' + str(n_pix_edge))
        if self.gpu_available:
            self.joblib_setting = 'processes'
        else:
            self.joblib_setting = 'threads'
        self.rooftop = rooftop

        if zeroPadding is not None:
            raise OopaoError('The use of the zeroPadding property has been deprecated')
        # Case where the zero-padding is not specificed => taking the smallest value ensuring to get edgePixel space from the edge.
        self.nRes = int((self.nSubap*2+self.n_pix_separation +
                        self.n_pix_edge*2)*self.telescope.resolution/self.nSubap)
        # zero-Padding Factor
        self.zeroPaddingFactor = self.nRes/self.telescope.resolution
        # zero-Padding Factor
        self.zeroPadding = (self.nRes - self.telescope.resolution)//2
        # Tag of the object
        self.tag = 'pyramid'
        # WFS detector object (see Detector class)
        self.cam = Detector(round(nSubap*self.zeroPaddingFactor))
        # WFS focal plane detector object (see Detector class)
        self.focal_plane_camera = Detector(int(
            (modulation*4+12)*self.zeroPaddingFactor), psf_sampling=self.zeroPaddingFactor)
        self.focal_plane_camera.is_focal_plane_camera = True
        # Light ratio for the valid pixels selection
        self.lightRatio = lightRatio
        if calibModulation >= self.telescope.resolution/2:
            self.calibModulation = self.telescope.resolution/2 - 1
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
        self.center = self.nRes//2
        self.supportPadded = self.convert_for_gpu(np.pad(self.telescope.pupil.astype(self.precision_complex()), ((
            self.zeroPadding, self.zeroPadding), (self.zeroPadding, self.zeroPadding)), 'constant'))
        # case where a spatial filter is considered
        self.spatialFilter = None
        self.fov = 206265*self.nRes/self.zeroPaddingFactor * \
            (self.telescope.src.wavelength/self.telescope.D)  # fov in arcsec
        self.fov_l_d = self.nRes/self.zeroPaddingFactor  # fov in arcsec

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
            A = np.ones([self.nRes, self.nRes]) + 1j * \
                np.ones([self.nRes, self.nRes])
            self.n_max = int(0.75*(np.min(self.mem_gpu)/1024) /
                             (A.nbytes/1024/1024/1024))
            del A

        # Prepare the Tip Tilt for the modulation -- normalized to apply the modulation in terms of lambda/D
        [self.Tip, self.Tilt] = np.meshgrid(np.linspace(-np.pi, np.pi, self.telescope.resolution),
                                            np.linspace(-np.pi, np.pi, self.telescope.resolution))
        self.Tilt *= self.telescope.pupil
        self.Tip *= self.telescope.pupil

        # compute the phasor to center the PSF on 4 pixels
        [xx, yy] = np.meshgrid(np.linspace(0, self.nRes-1, self.nRes),
                               np.linspace(0, self.nRes-1, self.nRes))
        # phasor for the FFT centering
        self.phasor = self.convert_for_gpu(np.exp(-(1j*np.pi*(self.nRes+1)/self.nRes)*(xx+yy)))

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
        print('Selection of the valid pixels...')
        self.initialization(self.telescope)
        print('Acquisition of the reference slopes and units calibration...')
        # set the modulation radius and propagate light
        self.modulation = modulation
        self.wfs_calibration(self.telescope)
        self.telescope.resetOPD()
        self.wfs_measure(phase_in=self.telescope.src.phase)
        print(self)

    def mask_computation(self):
        print('Pyramid Mask initialization...')
        if self.old_mask is False:
            self.m = self.get_phase_mask(resolution=self.nRes, n_subap=self.nSubap, n_pix_separation=self.n_pix_separation,
                                         n_pix_edge=self.n_pix_edge, psf_centering=self.psfCentering, sx=self.sx, sy=self.sy)
            self.initial_m = self.m.copy()
            # compute the PWFS mask)
            self.mask = self.convert_for_gpu(np.complex64(np.exp(1j*self.m)))
            # Save a copy of the initial mask
            self.initial_mask = np.copy(self.mask)
        else:
            raise DeprecationWarning('The use of the old_mask parameter has been deprecated')

    def apply_shift_wfs(self, sx=None, sy=None, mis_reg=None, units='pixels'):
        if sx is None:
            sx = 0
        if sy is None:
            sy = 0
        if mis_reg is not None:
            sx = [mis_reg.dX_1, mis_reg.dX_2, mis_reg.dX_3, mis_reg.dX_4]
            sy = [mis_reg.dY_1, mis_reg.dY_2, mis_reg.dY_3, mis_reg.dY_4]
        # apply a TIP/TILT of the PWFS mask to shift the pupils
        if units == 'pixels':
            factor = 2
        if units == 'm':
            factor = 2/(self.telescope.D/self.nSubap)
        # sx and sy are the units of displacements in pixels
        if np.isscalar(sx) and np.isscalar(sy):
            shift_x = [factor*sx, factor*sx, factor*sx, factor*sx]
            shift_y = [factor*sy, factor*sy, factor*sy, factor*sy]
        else:
            if len(sx) == 4 and len(sy) == 4:
                shift_x = []
                shift_y = []
                [shift_x.append(i_x*factor) for i_x in sx]
                [shift_y.append(i_y*factor) for i_y in sy]
            else:
                raise OopaoError('Wrong size for sx and/or sy, a list of 4 values is expected.')
        if np.max(np.abs(shift_x)) > self.n_pix_edge or np.max(np.abs(shift_y)) > self.n_pix_edge:
            warning('The Pyramid pupils have been shifted outside of the detector!' +
                    'Wrapping of the signal is currently occuring!!')

        self.sx = np.asarray(shift_x)/factor
        self.sy = np.asarray(shift_y)/factor
        self.m = self.get_phase_mask(resolution=self.nRes, n_subap=self.nSubap, n_pix_separation=self.n_pix_separation,
                                     n_pix_edge=self.n_pix_edge, psf_centering=self.psfCentering, sx=shift_x, sy=shift_y)
        self.mask = self.convert_for_gpu(np.complex64(np.exp(1j*self.m)))
        self.slopesUnits = 1
        self.referenceSignal = 0
        self.referenceSignal_2D = 0
        self.wfs_calibration(self.telescope)

    def get_phase_mask(self, resolution, n_subap, n_pix_separation, n_pix_edge, psf_centering=False, sx=[0, 0, 0, 0], sy=[0, 0, 0, 0]):
        # size of the mask in pixel
        n_tot = int((n_subap*2+n_pix_separation+n_pix_edge*2) * self.telescope.resolution/self.nSubap)

        # normalization factor for the Tip/Tilt
        n_pix_per_subap = self.telescope.resolution/self.nSubap
        norma = n_pix_per_subap/4
        # support for the mask
        m = np.zeros([n_tot, n_tot])
        if psf_centering:
            # mask centered on 4 pixel
            lim = np.pi
            # create a Tip/Tilt combination for each quadrant
            [Tip, Tilt] = np.meshgrid(np.linspace(-lim, lim, n_tot//2, endpoint=False),
                                      np.linspace(-lim, lim, n_tot//2, endpoint=False))
            # make sur it is zero-mean
            Tip -= np.mean(Tip)
            Tilt -= np.mean(Tilt)

            m[:n_tot//2, : n_tot//2] = (Tip * (n_subap + n_pix_separation + sx[0]) + Tilt * (n_subap + n_pix_separation - sy[0])) * norma
            m[:n_tot//2, -n_tot//2:] = (-Tip * (n_subap + n_pix_separation - sx[1]) + Tilt * (n_subap + n_pix_separation - sy[1])) * norma
            m[-n_tot//2:, -n_tot//2:] = (-Tip * (n_subap + n_pix_separation - sx[2]) - Tilt * (n_subap + n_pix_separation + sy[2])) * norma
            m[-n_tot//2:, :n_tot//2] = (Tip * (n_subap + n_pix_separation + sx[3]) - Tilt * (n_subap + n_pix_separation + sy[3])) * norma
        else:
            # mask centered on 1 pixel => different normalization for each Tip/tilt
            d_pix = (np.pi) / (n_tot)     # size of a pixel in angle
            lim_p = np.pi
            lim_m = np.pi - 2*d_pix

            # create a Tip/Tilt combination for each quadrant
            [Tip_1, Tilt_1] = np.meshgrid(np.linspace(-lim_p, lim_p, n_tot//2 + 1, endpoint=True), np.linspace(-lim_p, lim_p, n_tot//2 + 1, endpoint=True))
            [Tip_2, Tilt_2] = np.meshgrid(np.linspace(-lim_p, lim_p, n_tot//2 + 1, endpoint=True), np.linspace(-lim_m, lim_m, n_tot//2 - 1, endpoint=False))
            [Tip_3, Tilt_3] = np.meshgrid(np.linspace(-lim_m, lim_m, n_tot//2 - 1, endpoint=False), np.linspace(-lim_m, lim_m, n_tot//2 - 1, endpoint=False))
            [Tip_4, Tilt_4] = np.meshgrid(np.linspace(-lim_m, lim_m, n_tot//2 - 1, endpoint=False), np.linspace(-lim_p, lim_p, n_tot//2 + 1, endpoint=True))

            # make sur it is zero-mean
            Tip_1 -= np.mean(Tip_1)
            Tilt_1 -= np.mean(Tilt_1)

            # make sur it is zero-mean
            Tip_2 -= np.mean(Tip_2)
            Tilt_2 -= np.mean(Tilt_2)

            # make sur it is zero-mean
            Tip_3 -= np.mean(Tip_3)
            Tilt_3 -= np.mean(Tilt_3)

            # make sur it is zero-mean
            Tip_4 -= np.mean(Tip_4)
            Tilt_4 -= np.mean(Tilt_4)

            m[:n_tot//2 + 1, :n_tot//2+1] = (Tip_1 * (n_subap + n_pix_separation + sx[0]) + Tilt_1 * (n_subap + n_pix_separation - sy[0]))*norma
            m[:n_tot//2 + 1, -n_tot//2+1:] = (-Tip_4 * (n_subap + n_pix_separation - sx[1]) + Tilt_4 * (n_subap + n_pix_separation - sy[1]))*norma
            m[-n_tot//2 + 1:, -n_tot//2 + 1:] = (-Tip_3 * (n_subap + n_pix_separation - sx[2]) - Tilt_3 * (n_subap + n_pix_separation + sy[2]))*norma
            m[-n_tot//2 + 1:, :n_tot//2 + 1] = (Tip_2 * (n_subap + n_pix_separation + sx[3]) - Tilt_2 * (n_subap + n_pix_separation + sy[3]))*norma

        return -m  # sign convention for backward compatibility

    def initialization(self, telescope):
        telescope.resetOPD()
        if self.userValidSignal is None:
            if self.lightRatio == 0:
                self.cam.frame = np.ones(
                    [self.cam.resolution, self.cam.resolution])
            else:
                print('The valid pixel are selected on flux considerations')
                # set the modulation to a large value
                self.modulation = self.calibModulation
                self.wfs_measure(phase_in=self.telescope.src.phase)
            # save initialization frame
            self.initFrame = self.cam.frame

            # save the number of signals depending on the case
            if self.postProcessing[:10] == 'slopesMaps':
                # select the valid pixels of the detector according to the flux (case slopes-maps)
                I1 = self.grabQuadrant(1)
                I2 = self.grabQuadrant(2)
                I3 = self.grabQuadrant(3)
                I4 = self.grabQuadrant(4)

                # sum of the 4 quadrants
                self.I4Q = I1+I2+I3+I4
                # valid pixels to consider for the slopes-maps computation
                self.validI4Q = (self.I4Q >= self.lightRatio*self.I4Q.max())
                self.validSignal = np.concatenate(
                    (self.validI4Q, self.validI4Q))
                self.nSignal = int(np.sum(self.validSignal))

            if self.postProcessing[:9] == 'fullFrame':
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validSignal = (
                    self.initFrame >= self.lightRatio*self.initFrame.max())
                self.nSignal = int(np.sum(self.validSignal))
        else:
            print('You are using a user-defined mask for the selection of the valid pixel')
            if self.postProcessing[:10] == 'slopesMaps':
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validI4Q = self.userValidSignal
                self.validSignal = np.concatenate(
                    (self.validI4Q, self.validI4Q))
                self.nSignal = int(np.sum(self.validSignal))

            if self.postProcessing[:9] == 'fullFrame':
                self.validSignal = self.userValidSignal
                self.nSignal = int(np.sum(self.validSignal))

        # Tag to indicate that the wfs is initialized
        self.isInitialized = True

    def wfs_calibration(self, telescope):
        # save current OPD to be re-applied after calibration:
        tmp_OPD = telescope.OPD.copy()
        # reference slopes acquisition
        telescope.OPD = telescope.pupil.astype(float)
        # compute the refrence signals
        self.wfs_measure(phase_in=self.telescope.src.phase)
        self.referenceSignal_2D, self.referenceSignal = self.signalProcessing()

        # 2D reference Frame before binning with detector
        self.referencePyramidFrame = np.copy(self.raw_data)
        if self.isCalibrated is False:
            print('WFS calibrated!')
        self.isCalibrated = True
        # re-applied the initial OPD
        telescope.OPD = tmp_OPD

    def pyramid_transform(self, phase_in):
        # copy of the support for the zero-padding
        support = self.supportPadded.copy()
        # em field corresponding to phase_in
        if np.ndim(self.telescope.OPD) == 2 or type(self.telescope.OPD) is list:
            if self.modulation == 0:
                em_field = self.maskAmplitude*np.exp(1j*(phase_in))
            else:
                em_field = self.maskAmplitude * \
                    np.exp(
                        1j*(self.convert_for_gpu(self.telescope.src.phase)+phase_in))
        else:
            em_field = self.maskAmplitude*np.exp(1j*phase_in)
        # zero-padding for the FFT computation
        support[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,
                self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field

        del em_field
        # case with mask centered on 4 pixels
        if self.psfCentering:
            em_field_ft = fft2(support*self.phasor).astype(self.precision_complex())
            em_field_pwfs = xp.fft.ifft2(em_field_ft*self.mask).astype(self.precision_complex())
            intensity = xp.abs(em_field_pwfs)**2
        # case with mask centered on 1 pixel
        else:
            if self.spatialFilter is not None:
                em_field_ft = xp.fft.fftshift(
                    fft2(support))*self.spatialFilter
            else:
                em_field_ft = xp.fft.fftshift(fft2(support)).astype(self.precision_complex())

            em_field_pwfs = xp.fft.ifft2(em_field_ft*self.mask).astype(self.precision_complex())
            intensity = xp.abs(em_field_pwfs)**2
        del support
        del em_field_pwfs
        self.modulation_camera_em.append(
            self.convert_for_numpy(em_field_ft)/em_field_ft.shape[0])

        del em_field_ft
        del phase_in
        return intensity

    def setPhaseBuffer(self, phaseIn):
        B = self.phaseBuffModulationLowres_CPU+phaseIn
        return B

    def pyramid_propagation(self, telescope):
        # backward compatibility with previous version
        self.wfs_measure(phase_in=telescope.src.phase)
        return

    def wfs_integrate(self):
        # propagate to the detector to apply the noise
        self*self.cam
        if self.isInitialized and self.isCalibrated:
            signal_2D, signal = self.signalProcessing()
            return signal_2D, signal
        else:
            return None, None

    def wfs_measure(self, phase_in=None, integrate=True):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        # mask amplitude for the light propagation
        self.maskAmplitude = self.convert_for_gpu(np.sqrt(
            self.telescope.src.fluxMap/self.nTheta)*self.telescope.pupilReflectivity)

        if self.spatialFilter is not None:
            if np.ndim(phase_in) == 2:
                support_spatial_filter = np.copy(self.supportPadded)
                em_field = self.maskAmplitude * \
                    np.exp(1j*(self.telescope.src.phase))
                support_spatial_filter[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution //
                                       2, self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
                self.em_field_spatial_filter = (
                    np.fft.fft2(support_spatial_filter*self.phasor))
                self.pupil_plane_spatial_filter = (np.fft.ifft2(
                    self.em_field_spatial_filter*self.spatialFilter))

        # initialize modulation camera em field buffer
        self.modulation_camera_em = []

        if self.modulation == 0:
            if np.ndim(phase_in) == 2:
                self.raw_data = self.convert_for_numpy(self.pyramid_transform(
                    self.convert_for_gpu(self.telescope.src.phase)))
                if integrate:
                    self.pyramidSignal_2D, self.pyramidSignal = self.wfs_integrate()
            else:
                nModes = phase_in.shape[2]
                # move axis to get the number of modes first
                self.phase_buffer = self.convert_for_gpu(
                    np.moveaxis(self.telescope.src.phase, -1, 0))

                # define the parallel jobs
                def job_loop_multiple_modes_non_modulated():
                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                        delayed(self.pyramid_transform)(i) for i in self.phase_buffer)
                    return Q
                # apply the pyramid transform in parallel
                maps = self.convert_for_numpy(xp.asarray(
                    job_loop_multiple_modes_non_modulated()))

                self.pyramidSignal_2D = np.zeros(
                    [self.validSignal.shape[0], self.validSignal.shape[1], nModes])
                self.pyramidSignal = np.zeros([self.nSignal, nModes])

                for i in range(nModes):
                    self.raw_data = maps[i, :, :]
                    if integrate:
                        self.pyramidSignal_2D[:, :, i], self.pyramidSignal[:, i] = self.wfs_integrate(
                        )
                del maps

        else:
            if np.ndim(phase_in) == 2:
                n_max_ = self.n_max
                if self.nTheta > n_max_:
                    # break problem in pieces:
                    nCycle = int(np.ceil(self.nTheta/n_max_))
                    maps = self.convert_for_numpy(
                        xp.zeros([self.nRes, self.nRes]))
                    for i in range(nCycle):
                        if self.gpu_available:
                            try:
                                self.mempool = xp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                warning('could not free the memory')
                        if i < nCycle-1:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(delayed(self.pyramid_transform)(
                                    i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:(i+1)*n_max_, :, :]))
                                return Q
                            maps += self.convert_for_numpy(
                                xp.sum(xp.asarray(job_loop_single_mode_modulated()), axis=0))
                        else:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(delayed(self.pyramid_transform)(
                                    i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:, :, :]))
                                return Q
                            maps += self.convert_for_numpy(
                                xp.sum(xp.asarray(job_loop_single_mode_modulated()), axis=0))
                    self.maps = maps.copy()
                    self.raw_data = maps
                    del maps
                else:
                    # define the parallel jobs
                    def job_loop_single_mode_modulated():
                        Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                            delayed(self.pyramid_transform)(i) for i in self.phaseBuffModulationLowres)
                        return Q
                    # apply the pyramid transform in parallel
                    self.maps = xp.asarray(job_loop_single_mode_modulated())
                    # compute the sum of the pyramid frames for each modulation points
                    if self.weight_vector is None:
                        self.raw_data = self.convert_for_numpy(
                            xp.sum((self.maps), axis=0))
                    else:
                        weighted_map = np.reshape(
                            self.maps, [self.nTheta, self.nRes**2])
                        self.weighted_map = np.diag(
                            self.weight_vector)@weighted_map
                        self.raw_data = np.reshape(self.convert_for_numpy(
                            xp.sum((self.weighted_map), axis=0))/self.nTheta, [self.nRes, self.nRes])
                if integrate:
                    self.pyramidSignal_2D, self.pyramidSignal = self.wfs_integrate()
            else:
                if np.ndim(phase_in) == 3:
                    nModes = phase_in.shape[2]
                    # move axis to get the number of modes first
                    self.phase_buffer = np.moveaxis(
                        self.telescope.src.phase, -1, 0)

                    def jobLoop_setPhaseBuffer():
                        Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(
                            delayed(self.setPhaseBuffer)(i) for i in self.phase_buffer)
                        return Q
                    self.phaseBuffer = (np.reshape(np.asarray(jobLoop_setPhaseBuffer()), [
                                        nModes*self.nTheta, self.telescope.resolution, self.telescope.resolution]))
                    n_measurements = nModes*self.nTheta
                    n_max = self.n_max
                    n_measurement_max = int(np.floor(n_max/self.nTheta))
                    maps = xp.zeros([n_measurements, self.nRes, self.nRes])

                    if n_measurements > n_max:
                        nCycle = int(np.ceil(nModes/n_measurement_max))
                        for i in range(nCycle):
                            if self.gpu_available:
                                try:
                                    self.mempool = xp.get_default_memory_pool()
                                    self.mempool.free_all_blocks()
                                except:
                                    warning('could not free the memory')
                            if i < nCycle-1:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(
                                        self.phaseBuffer[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta, :, :]))
                                    return Q
                                maps[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta, :, :] = xp.asarray(
                                    job_loop_multiple_mode_modulated())
                            else:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(delayed(self.pyramid_transform)(
                                        i) for i in self.convert_for_gpu(self.phaseBuffer[i*n_measurement_max*self.nTheta:, :, :]))
                                    return Q
                                maps[i*n_measurement_max*self.nTheta:, :,
                                     :] = xp.asarray(job_loop_multiple_mode_modulated())
                        self.bufferPyramidFrames = self.convert_for_numpy(maps)
                        del self.phaseBuffer
                        del maps
                        if self.gpu_available:
                            try:
                                self.mempool = xp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                warning('could not free the memory')
                    else:
                        def job_loop_multiple_mode_modulated():
                            Q = Parallel(n_jobs=self.nJobs, prefer=self.joblib_setting)(delayed(
                                self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer))
                            return Q

                        self.bufferPyramidFrames = self.convert_for_numpy(
                            xp.asarray(job_loop_multiple_mode_modulated()))

                    self.pyramidSignal_2D = np.zeros(
                        [self.validSignal.shape[0], self.validSignal.shape[1], nModes])
                    self.pyramidSignal = np.zeros([self.nSignal, nModes])

                    for i in range(nModes):
                        self.raw_data = xp.sum(
                            self.bufferPyramidFrames[i*(self.nTheta):(self.nTheta)+i*(self.nTheta)], axis=0)
                        if integrate:
                            self.pyramidSignal_2D[:, :, i], self.pyramidSignal[:, i] = self.wfs_integrate(
                            )
                    del self.bufferPyramidFrames
                else:
                    raise OopaoError('Wrong dimension for the input phase. Aborting')
                if self.gpu_available:
                    try:
                        self.mempool = xp.get_default_memory_pool()
                        self.mempool.free_all_blocks()
                    except:
                        warning('could not free the memory')

    def signalProcessing(self, cameraFrame=None):
        if cameraFrame is None:
            cameraFrame = self.cam.frame
        if self.postProcessing == 'slopesMaps':
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=None)*self.validI4Q
            I2 = self.grabQuadrant(2, cameraFrame=None)*self.validI4Q
            I3 = self.grabQuadrant(3, cameraFrame=None)*self.validI4Q
            I4 = self.grabQuadrant(4, cameraFrame=None)*self.validI4Q
            # global normalisation
            I4Q = I1+I2+I3+I4
            self.norma = np.mean(I4Q[self.validI4Q])
            # slopesMaps computation cropped to the valid pixels
            Sx = (I1-I2+I4-I3)
            Sy = (I1-I4+I2-I3)
            # 2D slopes maps
            slopesMaps = (np.concatenate((Sx, Sy)/self.norma) -
                          self.referenceSignal_2D) * self.slopesUnits
            # slopes vector
            slopes = slopesMaps[np.where(self.validSignal == 1)]
            return slopesMaps, slopes

        if self.postProcessing == 'slopesMaps_incidence_flux':
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=None)*self.validI4Q
            I2 = self.grabQuadrant(2, cameraFrame=None)*self.validI4Q
            I3 = self.grabQuadrant(3, cameraFrame=None)*self.validI4Q
            I4 = self.grabQuadrant(4, cameraFrame=None)*self.validI4Q
            # global normalisation
            subArea = (self.telescope.D / self.nSubap)**2
            self.norma = np.float64(
                self.telescope.src.nPhoton*self.telescope.samplingTime*subArea)
            # slopesMaps computation cropped to the valid pixels
            Sx = (I1-I2+I4-I3)
            Sy = (I1-I4+I2-I3)
            # 2D slopes maps
            slopesMaps = (np.concatenate((Sx, Sy)/self.norma) -
                          self.referenceSignal_2D) * self.slopesUnits
            # slopes vector
            slopes = slopesMaps[np.where(self.validSignal == 1)]
            return slopesMaps, slopes

        if self.postProcessing == 'slopesMaps_camera_flux':
            # slopes-maps computation
            I1 = self.grabQuadrant(1, cameraFrame=None)*self.validI4Q
            I2 = self.grabQuadrant(2, cameraFrame=None)*self.validI4Q
            I3 = self.grabQuadrant(3, cameraFrame=None)*self.validI4Q
            I4 = self.grabQuadrant(4, cameraFrame=None)*self.validI4Q
            # global normalisation
            self.norma = np.float64(self.cam.frame.mean())
            # slopesMaps computation cropped to the valid pixels
            Sx = (I1-I2+I4-I3)
            Sy = (I1-I4+I2-I3)
            # 2D slopes maps
            slopesMaps = (np.concatenate((Sx, Sy)/self.norma) -
                          self.referenceSignal_2D) * self.slopesUnits
            # slopes vector
            slopes = slopesMaps[np.where(self.validSignal == 1)]
            return slopesMaps, slopes

        if self.postProcessing == 'fullFrame_camera_flux':
            # global normalization
            self.norma = np.float64(self.cam.frame.mean())
            # 2D full-frame
            fullFrameMaps = (cameraFrame / self.norma) - \
                self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]
            return fullFrameMaps, fullFrame

        if self.postProcessing == 'fullFrame_incidence_flux':
            # global normalization
            subArea = (self.telescope.D / self.nSubap)**2
            self.norma = np.float64(
                self.telescope.src.nPhoton*self.telescope.samplingTime*subArea)/4
            # 2D full-frame
            fullFrameMaps = (cameraFrame / self.norma) - \
                self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]
            return fullFrameMaps, fullFrame

        if self.postProcessing == 'fullFrame_sum_flux':
            # global normalization
            self.norma = np.float64(self.cam.frame.sum())
            # 2D full-frame
            fullFrameMaps = (cameraFrame / self.norma) - \
                self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]
            return fullFrameMaps, fullFrame

        if self.postProcessing == 'fullFrame':
            # global normalization
            self.norma = np.sum(cameraFrame[self.validSignal])
            # 2D full-frame
            fullFrameMaps = (cameraFrame / self.norma) - \
                self.referenceSignal_2D
            # full-frame vector
            fullFrame = fullFrameMaps[np.where(self.validSignal == 1)]
            return fullFrameMaps, fullFrame

    def get_modulation_frame(self, radius=6, norma=True):
        if radius <= 0:
            warning('radius for the field of view must be a strictly positive number. Ignoring the input value.')
            radius = self.telescope.resolution//2
        self.modulation_camera_frame = self.focal_plane_camera.frame.astype(
            float)
        N_trunc = int(self.nRes/2 - radius*self.zeroPaddingFactor)
        if N_trunc <= 0:
            warning('radius Value is too high as the field of view is limited to ' +
                    str(int(self.fov_l_d/2)) + ' lambda/D -- ignoring')
            modulation_camera_frame_zoom = self.modulation_camera_frame.copy()
        else:
            modulation_camera_frame_zoom = self.modulation_camera_frame[N_trunc:-
                                                                        N_trunc, N_trunc:-N_trunc]

        if norma:
            modulation_camera_frame_zoom /= modulation_camera_frame_zoom.max()

        return modulation_camera_frame_zoom

    def grabQuadrant(self, n, cameraFrame=None):

        nExtraPix = int(np.round((self.n_pix_separation/self.nSubap) *
                        self.telescope.resolution/(self.telescope.resolution/self.nSubap)/2/self.binning))
        centerPixel = int(np.round((self.cam.resolution/self.binning)/2))
        n_pixels = int(np.ceil(self.nSubap/self.binning))
        if cameraFrame is None:
            cameraFrame = self.cam.frame.copy()

        if self.rooftop is None:
            if n == 3:
                quadrant = cameraFrame[nExtraPix+centerPixel:(
                    nExtraPix+centerPixel+n_pixels), nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels)]
            if n == 4:
                quadrant = cameraFrame[nExtraPix+centerPixel:(
                    nExtraPix+centerPixel+n_pixels), -nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel)]
            if n == 1:
                quadrant = cameraFrame[-nExtraPix+centerPixel -
                                       n_pixels:(-nExtraPix+centerPixel), -nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel)]
            if n == 2:
                quadrant = cameraFrame[-nExtraPix+centerPixel -
                                       n_pixels:(-nExtraPix+centerPixel), nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels)]
        else:
            if self.rooftop == 'V':
                if n == 1:
                    quadrant = cameraFrame[centerPixel-n_pixels//2:(
                        centerPixel)+n_pixels//2, (self.edgePixel//2):(self.edgePixel//2 + n_pixels)]
                if n == 2:
                    quadrant = cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2, (
                        self.edgePixel//2 + n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels)]
                if n == 4:
                    quadrant = cameraFrame[centerPixel-n_pixels//2:(
                        centerPixel)+n_pixels//2, (self.edgePixel//2):(self.edgePixel//2 + n_pixels)]
                if n == 3:
                    quadrant = cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2, (
                        self.edgePixel//2 + n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels)]
            else:
                if n == 1:
                    quadrant = cameraFrame[(self.edgePixel//2):(self.edgePixel//2 + n_pixels),
                                           centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n == 2:
                    quadrant = cameraFrame[(self.edgePixel//2 + n_pixels+nExtraPix*2):(self.edgePixel //
                                                                                       2+nExtraPix*2+2*n_pixels), centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n == 4:
                    quadrant = cameraFrame[(self.edgePixel//2):(self.edgePixel//2 + n_pixels),
                                           centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n == 3:
                    quadrant = cameraFrame[(self.edgePixel//2 + n_pixels+nExtraPix*2):(self.edgePixel //
                                                                                       2+nExtraPix*2+2*n_pixels), centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
        return quadrant

    def grabFullQuadrant(self, n, cameraFrame=None):

        n_tot = self.cam.resolution

        if cameraFrame is None:
            cameraFrame = self.cam.frame.copy()

        if n == 3:
            quadrant = cameraFrame[:n_tot//2, :n_tot//2]
        if n == 4:
            quadrant = cameraFrame[:n_tot//2, -n_tot//2:]
        if n == 1:
            quadrant = cameraFrame[-n_tot//2:, -n_tot//2:]
        if n == 2:
            quadrant = cameraFrame[-n_tot//2:, :n_tot//2]
        return quadrant

    # properties required for backward compatibility (20/10/2020)
    @property
    def pyramidSignal(self):
        return self._pyramidSignal

    @pyramidSignal.setter
    def pyramidSignal(self, val):
        self._pyramidSignal = val
        self.signal = val

    @property
    def pyramidSignal_2D(self):
        return self._pyramidSignal_2D

    @pyramidSignal_2D.setter
    def pyramidSignal_2D(self, val):
        self._pyramidSignal_2D = val
        self.signal_2D = val

    # #properties required for backward compatibility (20/09/2024)
    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, val):
        self._raw_data = val
        self.pyramidFrame = val

    @property
    def lightRatio(self):
        return self._lightRatio

    @lightRatio.setter
    def lightRatio(self, val):
        self._lightRatio = val
        if hasattr(self, 'isInitialized'):
            if self.isInitialized:
                print('Updating the map if valid pixels ...')
                self.validI4Q = (self.I4Q >= self._lightRatio*self.I4Q.max())
                self.validSignal = np.concatenate(
                    (self.validI4Q, self.validI4Q))
                self.validPix = (
                    self.initFrame >= self.lightRatio*self.initFrame.max())

                # save the number of signals depending on the case
                if self.postProcessing[:10] == 'slopesMaps':
                    self.nSignal = np.sum(self.validSignal)
                    # display
                    xPix, yPix = np.where(self.validI4Q == 1)
                    plt.figure()
                    plt.imshow(self.I4Q.T)
                    plt.plot(xPix, yPix, '+')
                if self.postProcessing[:9] == 'fullFrame':
                    self.nSignal = np.sum(self.validPix)
                print('Done!')

    @property
    def spatialFilter(self):
        return self._spatialFilter

    @spatialFilter.setter
    def spatialFilter(self, val):
        self._spatialFilter = val
        if self.isInitialized:
            if val is None:
                print('No spatial filter considered')
                self.mask = self.initial_mask
                if self.isCalibrated:
                    print(
                        'Updating the reference slopes and Wavelength Calibration for the new modulation...')
                    self.slopesUnits = 1
                    self.referenceSignal = 0
                    self.referenceSignal_2D = 0
                    self.wfs_calibration(self.telescope)
                    print('Done!')
            else:
                tmp = np.ones([self.nRes, self.nRes])
                tmp[:, 0] = 0
                Tip = (sp.morphology.distance_transform_edt(tmp))
                Tilt = (sp.morphology.distance_transform_edt(np.transpose(tmp)))

                # normalize the TT to apply the modulation in terms of lambda/D
                self.Tip_spatial_filter = (((Tip/Tip.max())-0.5)*2*np.pi)
                self.Tilt_spatial_filter = (((Tilt/Tilt.max())-0.5)*2*np.pi)
                if val.shape == self.mask.shape:
                    print('A spatial filter is now considered')
                    self.mask = self.initial_mask * val
                    plt.figure()
                    plt.imshow(np.real(self.mask))
                    plt.title('Spatial Filter considered')
                    if self.isCalibrated:
                        print(
                            'Updating the reference slopes and Wavelength Calibration for the new modulation...')
                        self.slopesUnits = 1
                        self.referenceSignal = 0
                        self.referenceSignal_2D = 0
                        self.wfs_calibration(self.telescope)
                        print('Done!')
                else:
                    warning('Wrong shape for the spatial filter. No spatial filter attached to the mask')
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
    def modulation(self):
        return self._modulation

    @modulation.setter
    def modulation(self, val):
        self._modulation = val
        if self._modulation >= (self.telescope.resolution//2):
            raise OopaoError('Error the modulation radius is too large for this resolution!' +
                             'Consider using a larger telescope resolution!')
        if val != 0:
            self.modulation_path = []
            if self.user_modulation_path is not None:
                self.modulation_path = self.user_modulation_path
                self.nTheta = len(self.user_modulation_path)
            else:
                # define the modulation points
                perimeter = np.pi*2*self._modulation
                if self.nTheta_user_defined is None:
                    self.nTheta = 4 * \
                        int((self.extraModulationFactor+np.ceil(perimeter/4)))
                else:
                    self.nTheta = self.nTheta_user_defined

                self.thetaModulation = np.linspace(
                    0+self.delta_theta, 2*np.pi+self.delta_theta, self.nTheta, endpoint=False)
                for i in range(self.nTheta):
                    dTheta = self.thetaModulation[i]
                    self.modulation_path.append(
                        [self.modulation*np.cos(dTheta)+self.delta_Tip, self.modulation*np.sin(dTheta)+self.delta_Tilt])

            self.phaseBuffModulation = np.zeros(
                [self.nTheta, self.nRes, self.nRes]).astype(xp.float32)
            self.phaseBuffModulationLowres = np.zeros(
                [self.nTheta, self.telescope.resolution, self.telescope.resolution]).astype(xp.float32)

            for i in range(self.nTheta):
                self.TT = (
                    self.modulation_path[i][0]*self.Tip+self.modulation_path[i][1]*self.Tilt)*self.telescope.pupil
                self.phaseBuffModulation[i, self.center-self.telescope.resolution//2:self.center+self.telescope.resolution //
                                         2, self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = self.TT
                self.phaseBuffModulationLowres[i, :, :] = self.TT
            self.phaseBuffModulationLowres_CPU = self.phaseBuffModulationLowres.copy()
            if self.gpu_available:
                if self.nTheta <= self.n_max:
                    self.phaseBuffModulationLowres = self.convert_for_gpu(
                        self.phaseBuffModulationLowres)
        else:
            self.nTheta = 1

        if hasattr(self, 'isCalibrated'):
            if self.isCalibrated:
                print('Updating the reference slopes and Wavelength Calibration for the new modulation')
                self.slopesUnits = 1
                self.referenceSignal = 0
                self.referenceSignal_2D = 0
                self.wfs_calibration(self.telescope)
                print('Done!')

    def __mul__(self, obj):
        if obj.tag == 'detector':
            obj._integrated_time += self.telescope.samplingTime
            try:
                if obj.is_focal_plane_camera:
                    intensity = np.sum(
                        np.abs(self.modulation_camera_em)**2, axis=0)
                    if obj.resolution > self.nRes:
                        frame = intensity
                        warning('Maximum resolution for focal plane camera is %i, cropping field to this dimension' % self.nRes)
                    else:
                        frame = intensity[intensity.shape[0]//2-obj.resolution//2:intensity.shape[0]//2+obj.resolution //
                                          2, intensity.shape[0]//2-obj.resolution//2:intensity.shape[0]//2+obj.resolution//2]
                else:
                    raise OopaoError
            except:
                intensity = self.raw_data
                frame = (obj.set_binning(intensity, self.nRes/obj.resolution))
            if self.binning != 1:
                try:
                    frame = (obj.rebin(frame, (obj.resolution //
                             self.binning, obj.resolution//self.binning)))
                except:
                    warning('The shape of the detector ('+str(obj.frame.shape)+')' +
                            'is not valid with the binning value requested:' + str(self.binning) + '! -- Ignoring the binning.')
            obj.integrate(frame)
        else:
            raise OopaoError('Error light propagated to the wrong type of object')

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['pupil_diameter'] = f"{'Pupil diameter [px]':<25s}|{self.nSubap:^9d}"
        self.prop['pupil_separation'] = f"{'Pupil separation [px]':<25s}|{self.n_pix_separation:^9.2f}"
        self.prop['fov'] = f"{'Field of view [arcsec]':<25s}|{self.fov:^9.2f}"
        self.prop['modulation'] = f"{'Modulation radius [l/D]':<25s}|{self.modulation:^9.1f}"
        self.prop['psf_sampling'] = f"{'PSF sampling [px/(l/D)]':<25s}|{self.zeroPaddingFactor:^9.2f}"
        self.prop['psf_centering'] = f"{'PSF centering':<25s}|{str(self.psfCentering):^9s}"
        self.prop['n_valid_pixels'] = f"{'Valid pixels':<25s}|{self.nSignal:^9.0f}"
        self.prop['post_processing'] = f"{'Post processing':<25s}|{self.postProcessing:^9s}"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Pyramid WFS ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table