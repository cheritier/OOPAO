# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie
"""

import numpy as np
import copy
import sys
from .runtime import array_backend, gpu_resident, precision_bits
xp, global_gpu_flag = array_backend()
from OOPAO.tools.tools import set_binning, warning, OopaoError, get_array_module


class Telescope:

    def __init__(self, resolution: int,
                 diameter: float,
                 samplingTime: float = 0.001,
                 centralObstruction: float = 0.,
                 fov: float = 0.,
                 pupil: bool = None,
                 pupilReflectivity: float = 1.,
                 display_optical_path: bool = False):
        """TELESCOPE
        A Telescope object consists in defining the 2D mask of the entrance pupil.
        The Telescope is a central object in OOPAO:
            A source object is associated to the Telescope that carries the flux and wavelength information.
            An Atmosphere object can be paired to the Telescope to propagate the light through turbulent phase screens.
            The Telescope is required to initialize many of the OOPAO classes as it carries the pupil definition and pixel size.

        Parameters
        ----------
        resolution : float
            The resolution of the pupil mask.
        diameter : float
            The physical diameter of the telescope in [m].
        samplingTime : float, optional
            Defines the frequency of the AO loop. It is used in the Atmosphere object
            to update the turbulence phase screens according to the wind speed.
            The default is 0.001.
        centralObstruction : float, optional
            Adds a central obstruction in percentage of diameter.
            The default is 0.
        fov : float, optional
            Defines the Field of View of the Telescope object.
            This is useful for off-axis targets but it hasn't been properly implemented yet.
            The default is 0.
        pupil : bool, optional
            A user-defined pupil mask can be input to the Telescope object. It should consist of a binary array.
            The default is None.
        pupilReflectivity : float, optional
            Defines the reflectivity of the Telescope object.
            If not set to 1, it can be input as a 2D map of uneven reflectivy correspondong to the pupil mask.
            The default is 1.
        display_optical_path : bool, optional
            If desired, the optical path can be printed at each time the light is propagated to a WFS object
            setting the display_optical_path property to True.
            The default is False.

        Returns
        -------
        None.

        ************************** ADDING SPIDERS *******************************
        It is possible to add spiders to the telescope pupil using the following property:

            tel.apply_spiders(angle,thickness_spider,offset_X = None, offset_Y=None)

        where :
            - angle is a list of angle in [degrees]. The length of angle defines the number of spider.
            - thickness is the width of the spider in [m]
            - offset_X is a list (same lenght as angle) of shift X to apply to the individual spider  in [m]
            - offset_Y is a list (same lenght as angle) of shift Y to apply to the individual spider  in [m]

        ************************** PRINTING THE OPTICAL PATH *******************************
        It is possible to print the current optical path to verify through which object the light went through using the print_optical_path method:

                tel.print_optical_path()

        If desired, the optical path can be printed at each time the light is propagated to a WFS object setting the display_optical_path property to True:
            tel.display_optical_path = True


        ************************** COUPLING A SOURCE OBJECT **************************

        Once generated, the telescope should be coupled with a Source object "src" that contains the wavelength and flux properties of a target.
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src

        ************************** COUPLING TO MULTIPLE SOURCES **************************

        The telescope can be coupled with an Asterism object "ast" that contains different sources objects.
        _ This is achieved using the * operator     : ast*tel
        _ It can be accessed using "tel.src" . See the Asterism object documentation.



        ************************** COUPLING WITH AN ATMOSPHERE OBJECT **************************

        The telescope can be coupled to an Atmosphere object. In that case, the OPD of the atmosphere is automatically added to the telescope object.
        _ Coupling an Atmosphere and telescope Object   : tel+atm
        _ Separating an Atmosphere and telescope Object : tel-atm

        ************************** COMPUTING THE PSF **************************

        1) PSF computation directly using the Telescope property
        tel.computePSF(zeroPaddingFactor)  : computes the square module of the Fourier transform of the tel.src.phase using the zeropadding factor for the FFT

        2) PSF computation using a Detector object
            - create a detector (see Detector class documentation to set the different parameters for the camera frame: integration time, binning of the image, sampling of the PSF, etc.)
                cam = Detector()
            - propagate the light from the source through the telescope to the detector
                src*tel*cam
            - the PSF is accessible in tel.PSF (no detector effect) and in cam.frame (that includes the detector effects such as noise, binning, etc)


        ************************** MAIN PROPERTIES **************************

        The main properties of a Telescope object are listed here:
        _ tel.OPD       : the optical path difference
        _ tel.src.phase : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ tel.PSF       : Point Spread Function corresponding to to the tel.src.phase. This is not automatically set, it requires to run tel.computePSF().


        ************************** EXEMPLE **************************

        1) Create an 8-m diameter circular telescope with a central obstruction of 15% and the pupil sampled with 100 pixels along the diameter.
        tel = Telescope(resolution = 100, diameter = 8, centralObstruction = 0.15)

        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8)
        src*tel

        3) Compute the PSF with a zero-padding factor of 2.
        tel.computePSF(zeroPaddingFactor = 2)

        """

        OOPAO_path = [s for s in sys.path if "OOPAO" in s]
        l = []
        for i in OOPAO_path:
            l.append(len(i))
        path = OOPAO_path[np.argmin(l)]
        precision = precision_bits()
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        # bridge used only around the PropagateField call in computePSF (the
        # FFT-heavy PSF hot path); everything else in this class stays numpy,
        # see the notes in set_pupil and computePSF
        self.gpu_available = global_gpu_flag
        self.gpu_resident = gpu_resident()
        if self.gpu_available:
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
        else:
            self.convert_for_gpu = lambda a: a
            self.convert_for_numpy = lambda a: a
        self.isInitialized = False                        # Resolution of the telescope
        self.resolution = resolution                   # Resolution of the telescope
        self.D = diameter                     # Diameter in m
        self.initial_D = diameter                     # Diameter in m
        self.initial_resolution = resolution                     # Diameter in m
        self.pixelSize = self.D/self.resolution       # size of the pixels in m
        self.centralObstruction = centralObstruction           # central obstruction
        self.fov = fov                          # Field of View in arcsec converted in radian
        self.rad2arcsec = (180/xp.pi)*3600
        # Field of View in arcsec converted in radian
        self.fov_rad = fov/self.rad2arcsec
        self.samplingTime = samplingTime                 # AO loop speed
        # Flag to remove the petalling effect with ane ELT system.
        self.isPetalFree = False
        # indexes of the pixels corresponfong to the M1 petals. They need to be set externally
        self.index_pixel_petals = None
        # indexes of the pixels corresponfong to the M1 petals. They need to be set externally
        self.optical_path = None
        # input user-defined pupil
        self.user_defined_pupil = pupil
        # Pupil Reflectivity <=> amplitude map
        self.pupilReflectivity = self.precision(pupilReflectivity)
        self.set_pupil()                            # set the pupil
        # temporary source object associated to the telescope object
        self.src = None
        self.tag = 'telescope'                  # tag of the object
        # indicate if telescope object is paired with an atmosphere object
        self.isPaired = False
        # property to take into account a spatial filter
        self.spatialFilter = None
        # flag to display the optical path at each iteration
        self.display_optical_path = display_optical_path
        # perfect coronograph diameter (circular)
        self.coronagraph_diameter = None
        print(self)
        self.isInitialized = True
        self.apply_off_axis_tip_tilt = True
        self.initial_pupil = self.pupil.copy()
        self.static_OPD = np.zeros([self.resolution, self.resolution], dtype=self.precision)

    def relay(self, src):
        self.src = src
        if src.tag == 'source':
            self.src_list = [src]
        elif src.tag == 'asterism':
            self.src_list = src.src
        for src in self.src_list:
            src.optical_path.append([self.tag, self])
            src.tel = self
            backend = xp if src.gpu_resident else np
            src.mask = backend.asarray(self.pupil).copy()
            if src.OPD is None:
                src.OPD_no_pupil = backend.zeros(self.pupil.shape)
            if np.ndim(src.OPD) == 2:
                src.OPD = (src.OPD_no_pupil)*src.mask
            else:
                src.OPD_no_pupil = backend.zeros(self.pupil.shape)
            if src.scintillation is None:
                src.scintillation_no_pupil = backend.ones(self.pupil.shape)
                src.scintillation = (src.scintillation_no_pupil)*src.mask
            elif np.ndim(src.scintillation) == 2:
                src.scintillation = (src.scintillation_no_pupil)*src.mask
            else:
                src.scintillation_no_pupil = backend.ones(self.pupil.shape)
            src.var = backend.var(src.phase[backend.where(src.mask == 1)])
            src.fluxMap = self.pupilReflectivity * src.nPhoton * self.samplingTime * (self.D / self.resolution) ** 2
        return

    def set_pupil(self):
        # kept on numpy deliberately (not xp): tel.pupil/pupilReflectivity/
        # pixelArea/pupilLogical are read all over the codebase (Source,
        # Atmosphere, DeformableMirror, the WFS classes, ...) by code that has
        # no cupy awareness -- only PropagateField's own local FFT hot path
        # actually runs on GPU, see the get_array_module dispatch there
        # Case where the pupil is not input: circular pupil with central obstruction
        if self.user_defined_pupil is None:
            D = self.resolution+1
            x = np.linspace(-self.resolution/2, self.resolution/2, self.resolution, dtype=self.precision())
            xx, yy = np.meshgrid(x, x)
            circle = xx**2+yy**2
            obs = circle >= (self.centralObstruction*D/2)**2
            self.pupil = circle < (D/2)**2
            self.pupil = self.pupil*obs
        else:
            warning('User-defined pupil, the central obstruction will not be taken into account...')
            self.pupil = self.user_defined_pupil.copy().astype(self.precision())

        # A non uniform reflectivity can be input by the user
        self.pupilReflectivity = (self.pupil*self.pupilReflectivity).astype(self.precision())
        # Total number of pixels in the pupil area
        self.pixelArea = np.sum(self.pupil)
        # index of valid pixels in the pupil
        self.pupilLogical = np.where(np.reshape(self.pupil, self.resolution*self.resolution) > 0)
        self.pupil = self.pupil

    def computeCoronoPSF(self, zeroPaddingFactor=2, display=False, coronagraphDiameter=4.5):
        raise OopaoError("The method computeCoronoPSF has been deprecated and is now integrated within the computePSF method setting the tel.coronograph_diameter property (default value is None and means no coronograph considered)")

    def computePSF(self, zeroPaddingFactor=2, detector=None, img_resolution=None):
        factor = 1*self.apply_off_axis_tip_tilt
        # case when a detector is provided to the telescope (tel*det)
        if detector is not None:
            zeroPaddingFactor = detector.psf_sampling
            if detector.resolution is not None:
                img_resolution = detector.resolution
            else:
                img_resolution = int(zeroPaddingFactor*self.resolution)
                detector.resolution = img_resolution
        # case where the image should be cropped to img_resolution (used in tel*det as well using det.resolution property)
        if img_resolution is None:
            img_resolution = int(zeroPaddingFactor*self.resolution)
        if self.src is None:
            # raise an error if no source is coupled to the telescope
            raise OopaoError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        elif self.src.tag == 'asterism':
            # case with multiple sources
            input_source = self.src.src
            # check where is located the source in the focal plane
            if self.src.n_source > 1:
                r = np.squeeze(np.asarray(self.src.coordinates))[:, 0]
                theta = np.squeeze(np.asarray(self.src.coordinates))[:, 1]
                x_max = max(np.abs(r * np.cos(np.deg2rad(theta))))
                y_max = max(np.abs(r * np.sin(np.deg2rad(theta))))
            else:
                r = np.squeeze(np.asarray(self.src.coordinates))[0]
                theta = np.squeeze(np.asarray(self.src.coordinates))[1]
                x_max = (np.abs(r * np.cos(np.deg2rad(theta))))
                y_max = (np.abs(r * np.sin(np.deg2rad(theta))))
        else:
            input_source = [self.src]
            r = np.squeeze(np.asarray(self.src.coordinates))[0]
            theta = np.squeeze(np.asarray(self.src.coordinates))[1]
            x_max = (np.abs(r * np.cos(np.deg2rad(theta))))
            y_max = (np.abs(r * np.sin(np.deg2rad(theta))))
        pixel_scale = self.rad2arcsec*(input_source[0].wavelength/self.D)/zeroPaddingFactor  # in arcsec
        maximum_fov = pixel_scale*img_resolution/2
        n_extra = np.abs(np.floor((maximum_fov - max(x_max, y_max))/pixel_scale) - img_resolution//2)
        n_pix = np.ceil(max(int(img_resolution/2 + n_extra)*2, img_resolution)).astype('int')
        detector_gpu = detector is not None and getattr(detector, 'gpu_available', False)
        psf_backend = xp if detector_gpu else np
        if self.apply_off_axis_tip_tilt:
            self.support_PSF = psf_backend.zeros([n_pix, n_pix])
        else:
            n_pix = img_resolution
            self.support_PSF = psf_backend.zeros([img_resolution, img_resolution])
        center = self.support_PSF.shape[0]//2

        input_wavelenght = input_source[0].wavelength
        output_PSF = []
        output_PSF_norma = []
        # iterate for each source
        for i_src in range(len(input_source)):
            if input_wavelenght == input_source[i_src].wavelength:
                input_wavelenght = input_source[i_src].wavelength
            else:
                raise OopaoError('The asterism contains sources with different wavelengths. Summing up PSFs with different wavelength is not implemented.')
            # check if the source interacted with a spatial filter
            # Source fields are device arrays only in explicit resident mode;
            # the pupil and reflectivity maps remain public NumPy arrays.
            if input_source[i_src].phase_filtered is None:
                field_backend = xp if input_source[i_src].gpu_resident else np
                amp_mask = field_backend.sqrt(input_source[i_src].intensity)
                phase = input_source[i_src].phase
            else:
                field_backend = xp if input_source[i_src].gpu_resident else np
                amp_mask = field_backend.asarray(input_source[i_src].amplitude_filtered)
                phase = field_backend.asarray(input_source[i_src].phase_filtered)
            # amp_mask = amp_mask * xp.sqrt(input_source[i_src].scintillation)
            # amplitude of the EM field:
            amp = amp_mask*field_backend.asarray(self.pupil)*field_backend.asarray(self.pupilReflectivity)
            # add a Tip/Tilt for off-axis sources
            [Tip, Tilt] = np.meshgrid(np.linspace(-np.pi, np.pi, self.resolution, endpoint=False, dtype=self.precision()),
                                      np.linspace(-np.pi, np.pi, self.resolution, endpoint=False, dtype=self.precision()))
            r = (input_source[i_src].coordinates[0])
            # X/Y shift inversion to match convention for atmosphere
            x_shift = r*np.sin(np.deg2rad(input_source[i_src].coordinates[1]))  # in arcsec
            y_shift = r*np.cos(np.deg2rad(input_source[i_src].coordinates[1]))  # in arcsec
            # shift in pixel of the PSF
            delta_x = int(factor*np.floor(np.abs(x_shift)/pixel_scale)*np.sign(x_shift))
            delta_y = int(factor*np.floor(np.abs(y_shift)/pixel_scale)*np.sign(y_shift))

            delta_Tilt = x_shift - delta_x*pixel_scale
            delta_Tip = y_shift - delta_y*pixel_scale

            self.delta_TT = field_backend.asarray((delta_Tip*Tip + delta_Tilt*Tilt)*self.pupil)*(self.D/input_source[i_src].wavelength)*(1/self.rad2arcsec)

            # axis in arcsec
            self.xPSF_arcsec = [-self.rad2arcsec*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                                self.rad2arcsec*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]
            self.yPSF_arcsec = [-self.rad2arcsec*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                                self.rad2arcsec*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]

            # axis in radians
            self.xPSF_rad = [-(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                             (input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]
            self.yPSF_rad = [-(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                             (input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]
            # propagate the EM Field -- amp/phase are numpy at this point (see
            # above); upload them here so the FFT in PropagateField (the
            # dominant cost) runs on GPU when available. PropagateField itself
            # is backend-agnostic (dispatches on whatever it's given, via
            # get_array_module), which is what lets LiFT.py call it directly
            # with its own choice of backend without going through here at all.
            self.PropagateField(amplitude=self.convert_for_gpu(amp),
                                phase=self.convert_for_gpu(phase+self.delta_TT*factor),
                                zeroPaddingFactor=zeroPaddingFactor,
                                img_resolution=img_resolution)
            if not detector_gpu:
                self.PSF = self.convert_for_numpy(self.PSF)
            # normalized PSF
            self.PSF_norma = self.PSF/self.PSF.max()
            output_PSF.append(self.PSF.copy())
            output_PSF_norma.append(self.PSF.copy())
            self.support_PSF[center+delta_x-img_resolution//2:center+delta_x+img_resolution//2,
                             center+delta_y-img_resolution//2:center+delta_y+img_resolution//2] += self.PSF.copy()
        if len(output_PSF) == 1:
            output_PSF = output_PSF[0]
            output_PSF_norma = output_PSF_norma[0]
        self.PSF = self.support_PSF
        self.PSF_norma = self.PSF/self.PSF.max()
        if detector_gpu:
            self.PSF_norma = self.convert_for_numpy(self.PSF_norma)
        if detector_gpu:
            if isinstance(output_PSF, list):
                output_PSF = [self.convert_for_numpy(frame) for frame in output_PSF]
            else:
                output_PSF = self.convert_for_numpy(output_PSF)
        self.PSF_list = output_PSF

    def PropagateField(self, amplitude, phase, zeroPaddingFactor, img_resolution=None):
        # dispatch on whatever backend `amplitude` actually is, not on the
        # module-level xp: this is what lets computePSF upload numpy amp/phase
        # to run this on GPU, while LiFT.py can call this method directly with
        # its own choice of backend (see its `xp = cp if self.gpu else np`) and
        # get a matching-backend result back, with no coupling between the two
        xp_ = get_array_module(amplitude)
        oversampling = 1
        resolution = self.pupil.shape[0]
        if oversampling is not None:
            oversampling = oversampling
        if img_resolution is not None:
            if img_resolution > zeroPaddingFactor * resolution:
                raise OopaoError('Error: image has too many pixels for this pupil sampling. Try using a pupil mask with more pixels')
        else:
            img_resolution = zeroPaddingFactor * resolution
        # sizes/paddings are plain scalars, not pixel data -- computed on numpy
        # regardless of xp_ so they stay ordinary python/numpy ints, safe to
        # use as pad widths and slice bounds on either backend
        if zeroPaddingFactor * oversampling < 2:
            oversampling = int(np.ceil(2.0 / zeroPaddingFactor))
        img_size = int(np.ceil(img_resolution * oversampling))
        N = int(np.fix(zeroPaddingFactor * oversampling * resolution))
        pad_width = int(np.ceil((N - resolution) / 2))
        supportPadded = xp_.pad(amplitude * xp_.exp(1j * phase), pad_width=((pad_width, pad_width), (pad_width, pad_width)), constant_values=0).astype(self.precision_complex())
        # make sure the number of pxels is correct after the padding
        N = supportPadded.shape[0]
        # case considering a coronograph
        if self.coronagraph_diameter is not None:
            # self.pupil lives on numpy (see set_pupil); bridge it here rather
            # than changing what backend it lives on everywhere else
            pupil_ = xp_.asarray(self.pupil) if xp_ is not np else self.pupil
            [xx, yy] = xp_.meshgrid(xp_.linspace(0, N-1, N, dtype=self.precision()), xp_.linspace(0, N-1, N, dtype=self.precision()))
            xxc = xx - (N-1)/2
            yyc = yy - (N-1)/2
            self.apodiser = xp_.sqrt(xxc**2 + yyc**2) < self.resolution/2
            self.pupilSpiderPadded = xp_.pad(pupil_, pad_width=((pad_width, pad_width), (pad_width, pad_width)), constant_values=0).astype(self.precision_complex())
            self.focalMask = xp_.sqrt(xxc**2 + yyc**2) > self.coronagraph_diameter/2 * zeroPaddingFactor
            self.lyotStop = ((xp_.sqrt((xxc-1.0)**2 + (yyc-1.0)**2) < N/2 * 0.9) * self.pupilSpiderPadded)
            # PSF computation
            [xx, yy] = xp_.meshgrid(xp_.linspace(0, N - 1, N, dtype=self.precision()), xp_.linspace(0, N - 1, N, dtype=self.precision()), copy=False)
            phasor = xp_.exp(-1j * xp_.pi / N * (xx + yy) * (1 - img_resolution % 2)).astype(self.precision_complex)
            #                                                        ^--- this is to account odd/even number of pixels
            # Propagate with Fourier shifting
            EMF = xp_.fft.fftshift(1 / N * xp_.fft.fft2(xp_.fft.ifftshift(supportPadded * phasor*self.apodiser)))
            self.B = EMF * self.focalMask * phasor
            self.C = xp_.fft.fftshift(1 * xp_.fft.ifft2(xp_.fft.ifftshift(self.B))).astype(self.precision_complex) * self.lyotStop * phasor
            EMF = (xp_.fft.fftshift(1 * xp_.fft.fft2(xp_.fft.ifftshift(self.C)))).astype(self.precision_complex)
        else:
            # PSF computation
            [xx, yy] = xp_.meshgrid(xp_.linspace(0, N - 1, N, dtype=self.precision()), xp_.linspace(0, N - 1, N, dtype=self.precision()), copy=False)
            self.phasor = xp_.exp(-1j * xp_.pi * (N + 1) / N * (xx + yy) * (1 - img_resolution % 2)).astype(self.precision_complex())
            #                                                        ^--- this is to account odd/even number of pixels
            # Propagate with Fourier shifting
            # (a leftover line here used to compute an fftshift'd EMF that was
            # never read before being immediately overwritten by this one --
            # a full extra FFT + fftshift/ifftshift round-trip discarded every
            # single call; removed, see git history for the introducing commit)
            EMF = (1 / N * xp_.fft.fft2((supportPadded * self.phasor))).astype(self.precision_complex())
        # Again, this is to properly crop a PSF with the odd/even number of pixels
        if N % 2 == img_size % 2:
            shift_pix = 0
        else:
            if N % 2 == 0:
                shift_pix = 1
            else:
                shift_pix = -1
        # Support only rectangular PSFs -- plain python ints (not xp_), same
        # reasoning as the sizes/paddings above: these are slice bounds
        id0 = int(np.ceil(N / 2) - img_size // 2 + (1 - N % 2) - 1)
        id1 = int(np.ceil(N / 2) + img_size // 2 + shift_pix)
        EMF = EMF[id0:id1, id0:id1]
        self.focal_EMF = EMF
        if oversampling != 1:
            self.PSF = set_binning(xp_.abs(EMF) ** 2, oversampling)
        else:
            self.PSF = xp_.abs(EMF) ** 2
        return oversampling

    def apply_spiders(self, angle, thickness_spider, offset_X=None, offset_Y=None):
        # kept on numpy throughout, same reasoning as set_pupil: this builds
        # tel.pupil, read everywhere with no cupy awareness
        self.isInitialized = False
        if thickness_spider > 0:
            self.set_pupil()
            pup = np.copy(self.pupil)
            max_offset = self.centralObstruction*self.D/2 - thickness_spider/2
            if offset_X is None:
                offset_X = np.zeros(len(angle))
            if offset_Y is None:
                offset_Y = np.zeros(len(angle))

            if np.max(np.abs(offset_X)) >= max_offset or np.max(np.abs(offset_Y)) > max_offset:
                warning('The spider offsets are too large! Weird things could happen!')
            for i in range(len(angle)):
                angle_val = (angle[i]+90) % 360
                x = np.linspace(-self.D/2, self.D/2, self.resolution, dtype=self.precision())
                [X, Y] = np.meshgrid(x, x)
                X += offset_X[i]
                Y += offset_Y[i]
                map_dist = np.abs(X*np.cos(np.deg2rad(angle_val)) + Y*np.sin(np.deg2rad(-angle_val)))
                if 0 <= angle_val < 90:
                    map_dist[:self.resolution//2, :] = thickness_spider
                if 90 <= angle_val < 180:
                    map_dist[:, :self.resolution//2] = thickness_spider
                if 180 <= angle_val < 270:
                    map_dist[self.resolution//2:, :] = thickness_spider
                if 270 <= angle_val < 360:
                    map_dist[:, self.resolution//2:] = thickness_spider
                pup *= map_dist > thickness_spider/2
            self.isInitialized = True
            self.pupil = pup.copy()
        else:
            warning('Thickness is <=0, returning default pupil')
            self.set_pupil()
        return

    def pad(self, padding_values=0, sky_offset=None):
        """
        This functions allows to pad the pupil of padding_values pixels on both sides.
        The Telescope properties associated to it are automatically updated.
        It accepts shift offsets to change the pupil position with respect to a centered position.
        Returns
        -------
        None.

        """
        pupil_padded = np.pad(self.initial_pupil, [padding_values, padding_values])*0
        n_extra_pix = padding_values
        if sky_offset is not None:
            if max(sky_offset) < n_extra_pix:
                pupil_padded[n_extra_pix-sky_offset[0]:-n_extra_pix-sky_offset[0],
                             n_extra_pix-sky_offset[1]:-n_extra_pix-sky_offset[1]] = self.initial_pupil
            else:
                raise OopaoError('The sky_offsets are too large for the considered pupil')
        else:
            pupil_padded[n_extra_pix:-n_extra_pix, n_extra_pix:-n_extra_pix] = self.initial_pupil
        self.resolution = pupil_padded.shape[0]
        self.D = self.resolution * self.pixelSize
        self.pupil = pupil_padded.copy()
        return

    @property
    def pupil(self):
        return self._pupil

    @pupil.setter
    def pupil(self, val):
        # numpy throughout, same reasoning as set_pupil
        self._pupil = val.astype(bool)
        self.pixelArea = np.sum(self._pupil)
        tmp = np.reshape(self._pupil, self.resolution**2)
        self.pupilLogical = np.where(tmp > 0)
        self.pupilReflectivity = self.pupil.astype(self.precision())
        if self.isInitialized:
            warning('A new pupil is now considered, its reflectivity is considered to be uniform. Assign the proper reflectivity map to tel.pupilReflectivity if required.')

    @property
    def OPD(self):
        if np.ndim(self.src.OPD) == 2:
            backend = get_array_module(self.src.OPD)
            pupil = backend.asarray(self.pupil)
            self.mean_removed_OPD = (self.src.OPD - backend.mean(self.src.OPD[backend.where(pupil == 1)]))*pupil
        return self.src.OPD

    @OPD.setter
    def OPD(self, val):
        self.src.OPD = val

    @property
    def OPD_no_pupil(self):
        return self.src.OPD_no_pupil

    @OPD_no_pupil.setter
    def OPD_no_pupil(self, val):
        self.src.OPD_no_pupil = val

    def resetOPD(self):
        warning('The use of resetOPD is deprecated. Consider using the Source reset option using the ** operator to reset the light propagation:\n src**tel')
        self.src**self

    # This function was replaced by relay functions in different objects
    # Remains here for now for backward compatibility
    def __mul__(self, obj):
        # case where multiple objects are considered
        if type(obj) is list:
            wfs_signal = []
            if type(self.OPD) is list:
                if len(self.OPD) == len(obj):
                    for i_obj in range(len(self.OPD)):
                        tel_tmp = copy.deepcopy(getattr(obj[i_obj], 'telescope'))
                        tel_tmp.OPD = self.OPD[i_obj]
                        tel_tmp.OPD_no_pupil = self.OPD_no_pupil[i_obj]
                        self.src.src[i_obj]*tel_tmp*obj[i_obj]
                        wfs_signal.append(obj[i_obj].signal)
                    obj[i_obj].signal = np.mean(wfs_signal, axis=0)
                else:
                    raise OopaoError('Error! There is a mis-match between the number of Sources ('+str(
                        len(self.OPD))+') and the number of WFS ('+str(len(obj))+')')
            else:
                for i_obj in range(len(obj)):
                    self*obj[i_obj]
        else:
            self.relay(self.src)
            obj.relay(self.src)
        return self
    # <JM @ SpaceODT> This is no longer needed as the atmosphere behaves as its own entity now.
    # Remains here for now for reference
    # Combining with an atmosphere object

    def __add__(self, obj):
        if obj.tag == 'atmosphere':
            self.isPaired = True
            self.src*obj*self
            self.atm = obj
            if self.isPetalFree:
                self.removePetalling()

        if obj.tag == 'spatialFilter':
            self.spatialFilter = obj
            self*obj
            print('Telescope and Spatial Filter combined!')

    # Separating from an atmosphere object
    def __sub__(self, obj):
        if obj.tag == 'atmosphere':
            self.isPaired = False
            self.src.reset()
            obj.asterism = None

        if obj.tag == 'spatialFilter':
            self.spatialFilter = None
            print('Telescope and Spatial Filter separated!')

    def print_optical_path(self):
        if self.optical_path is not None:
            tmp_path = ''
            for i in range(len(self.optical_path)):
                tmp_path += self.optical_path[i][0]
                if i < len(self.optical_path)-1:
                    tmp_path += ' ~~> '
            print(tmp_path)
        else:
            print('No light propagated through the telescope')
        return

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['diameter'] = f"{'Diameter [m]':<25s}|{self.D:^10.2f}"
        self.prop['resolution'] = f"{'Resolution [px]':<25s}|{self.resolution:^10.0f}"
        self.prop['pixel_size'] = f"{'Pixel size [m]':<25s}|{self.pixelSize:^10.2f}"
        self.prop['surface'] = f"{'Surface [m²]':<25s}|{self.pixelArea*self.pixelSize**2:^10.2f}"
        self.prop['obstruction'] = f"{'Central obstruction [%]':<25s}|{self.centralObstruction*100:^10.0f}"
        self.prop['n_pix_pupil'] = f"{'Pixels in pupil':<25s}|{self.pixelArea:^10.0f}"
        self.prop['fov'] = f"{'Field of view [arcsec]':<25s}|{self.fov:^10.2f}"
        if self.src:
            if self.src.type == 'asterism':
                for i_src in range(len(self.src.src)):
                    self.prop['source_%d'%i_src] = f"{'Source %s [m]'%self.src.src[i_src].type:<25s}|{self.src.src[i_src].wavelength:^10.2e}"
            else:
                self.prop['source_%d'] = f"{'Source %s [m]'%self.src.type:<25s}|{self.src.wavelength:^10.2e}"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Telescope ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
