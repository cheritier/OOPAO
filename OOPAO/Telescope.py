# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie
"""

import numpy as np
import copy
import sys
try:
    import cupy as xp
    global_gpu_flag = True
    xp = np #for now
except ImportError or ModuleNotFoundError:
    xp = np

from OOPAO.tools.tools import set_binning, warning, OopaoError


class Telescope:

    def __init__(self, resolution: float,
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
        precision = np.load(path+'/precision_oopao.npy')
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        self.isInitialized = False                        # Resolution of the telescope
        self.resolution = resolution                   # Resolution of the telescope
        self.D = diameter                     # Diameter in m
        self.pixelSize = self.D/self.resolution       # size of the pixels in m
        self.centralObstruction = centralObstruction           # central obstruction
        self.fov = fov                          # Field of View in arcsec converted in radian
        # Field of View in arcsec converted in radian
        self.fov_rad = fov/206265
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


    def relay(self, src):
        self.src = src
        self.src.mask = self.pupil.copy()

        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src

        for src in src_list:
            src.optical_path.append([self.tag, self])
            src.tel = self

            src.mask = self.pupil.copy()

            #TODO: Create a tel.OPD to add to the source OPD
            src.OPD_no_pupil = src.OPD_no_pupil.copy()
            src.OPD = src.OPD_no_pupil*src.mask



            src.var = np.var(src.phase[np.where(self.pupil == 1)])
            src.fluxMap = self.pupilReflectivity * src.nPhoton * \
                          self.samplingTime * (self.D / self.resolution) ** 2



    def set_pupil(self):
        # Case where the pupil is not input: circular pupil with central obstruction
        if self.user_defined_pupil is None:
            D = self.resolution+1
            x = xp.linspace(-self.resolution/2, self.resolution/2, self.resolution, dtype=self.precision())
            xx, yy = xp.meshgrid(x, x)
            circle = xx**2+yy**2
            obs = circle >= (self.centralObstruction*D/2)**2
            self.pupil = circle < (D/2)**2
            self.pupil = self.pupil*obs
        else:
            warning(
                'User-defined pupil, the central obstruction will not be taken into account...')
            self.pupil = self.user_defined_pupil.copy().astype(self.precision())

        # A non uniform reflectivity can be input by the user
        self.pupilReflectivity = (self.pupil*self.pupilReflectivity).astype(self.precision())
        # Total number of pixels in the pupil area
        self.pixelArea = xp.sum(self.pupil)
        # index of valid pixels in the pupil
        self.pupilLogical = xp.where(xp.reshape(self.pupil, self.resolution*self.resolution) > 0)
        self.pupil = self.pupil

    def computeCoronoPSF(self, zeroPaddingFactor=2, display=False, coronagraphDiameter=4.5):
        raise OopaoError("The method computeCoronoPSF has been deprecated and is now integrated within the computePSF method setting the tel.coronograph_diameter property (default value is None and means no coronograph considered)")

    def computePSF(self, zeroPaddingFactor=2, detector=None, img_resolution=None):
        conversion_constant = (180/xp.pi)*3600
        factor = 1
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
            img_resolution = zeroPaddingFactor*self.resolution
        if self.src is None:
            # raise an error if no source is coupled to the telescope
            raise OopaoError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        elif self.src.tag == 'asterism':
            # case with multiple sources
            input_source = self.src.src
            # check where is located the source in the focal plane
            if self.src.n_source > 1:
                r = xp.squeeze(xp.asarray(self.src.coordinates))[:, 0]
                theta = xp.squeeze(xp.asarray(self.src.coordinates))[:, 1]
                x_max = max(xp.abs(r * xp.cos(np.deg2rad(theta))))
                y_max = max(xp.abs(r * xp.sin(np.deg2rad(theta))))
            else:
                r = xp.squeeze(xp.asarray(self.src.coordinates))[0]
                theta = xp.squeeze(xp.asarray(self.src.coordinates))[1]
                x_max = (xp.abs(r * xp.cos(np.deg2rad(theta))))
                y_max = (xp.abs(r * xp.sin(np.deg2rad(theta))))
            

        else:
            input_source = [self.src]
            r = xp.squeeze(xp.asarray(self.src.coordinates))[0]
            theta = xp.squeeze(xp.asarray(self.src.coordinates))[1]
            x_max = (xp.abs(r * xp.cos(np.deg2rad(theta))))
            y_max = (xp.abs(r * xp.sin(np.deg2rad(theta))))

        pixel_scale = conversion_constant*(input_source[0].wavelength/self.D)/zeroPaddingFactor
        maximum_fov = pixel_scale*img_resolution/2
        n_extra = np.abs(np.floor((maximum_fov - max(x_max, y_max))/pixel_scale) - img_resolution//2)
        n_pix = max(int(img_resolution/2 + n_extra)*2, img_resolution)
        center = n_pix//2
        self.support_PSF = np.zeros([n_pix, n_pix])
        input_wavelenght = input_source[0].wavelength
        output_PSF = []
        output_PSF_norma = []
        # iterate for each source
        for i_src in range(len(input_source)):
            if input_wavelenght == input_source[i_src].wavelength:
                input_wavelenght = input_source[i_src].wavelength
            else:
                raise OopaoError('The asterism contains sources with different wavelengths. Summing up PSFs with different wavelength is not implemented.')
            if self.spatialFilter is None:
                amp_mask = 1
                phase = input_source[i_src].phase
            else:
                amp_mask = self.amplitude_filtered
                phase = self.phase_filtered
            # amplitude of the EM field:
            amp = amp_mask*self.pupil*self.pupilReflectivity * xp.sqrt(input_source[i_src].fluxMap)
            # add a Tip/Tilt for off-axis sources
            [Tip, Tilt] = xp.meshgrid(xp.linspace(-xp.pi, xp.pi, self.resolution, dtype=self.precision()),
                                      xp.linspace(-xp.pi, xp.pi, self.resolution, dtype=self.precision()))

            r = (input_source[i_src].coordinates[0])
            # X/Y shift inversion to match convention for atmosphere

            x_shift = r*xp.sin(np.deg2rad(input_source[i_src].coordinates[1]))
            y_shift = r*xp.cos(np.deg2rad(input_source[i_src].coordinates[1]))
            delta_x = int(np.floor(np.abs(x_shift)/pixel_scale)*np.sign(x_shift))
            delta_y = int(np.floor(np.abs(y_shift)/pixel_scale)*np.sign(y_shift))

            delta_Tip = (np.abs(x_shift) % pixel_scale)*np.sign(x_shift)
            delta_Tilt = (np.abs(y_shift) % pixel_scale)*np.sign(y_shift)
            self.delta_TT = (delta_Tip*Tip + delta_Tilt*Tilt)*self.pupil*(self.D/input_source[i_src].wavelength)*(1/conversion_constant)
            # axis in arcsec
            self.xPSF_arcsec = [-conversion_constant*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                                conversion_constant*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]
            self.yPSF_arcsec = [-conversion_constant*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                                conversion_constant*(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]

            # axis in radians
            self.xPSF_rad = [-(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                             (input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]
            self.yPSF_rad = [-(input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor),
                             (input_source[i_src].wavelength/self.D) * (n_pix/2/zeroPaddingFactor)]

            # propagate the EM Field
            self.PropagateField(amplitude=amp,
                                phase=phase+self.delta_TT*factor,
                                zeroPaddingFactor=zeroPaddingFactor,
                                img_resolution=img_resolution)

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
        self.PSF_list = output_PSF

    def PropagateField(self, amplitude, phase, zeroPaddingFactor, img_resolution=None):
        oversampling = 1
        resolution = self.pupil.shape[0]
        if oversampling is not None:
            oversampling = oversampling
        if img_resolution is not None:
            if img_resolution > zeroPaddingFactor * resolution:
                raise OopaoError('Error: image has too many pixels for this pupil sampling. Try using a pupil mask with more pixels')
        else:
            img_resolution = zeroPaddingFactor * resolution
        # If PSF is undersampled apply the integer oversampling
        if zeroPaddingFactor * oversampling < 2:
            oversampling = (xp.ceil(2.0 / zeroPaddingFactor)).astype('int')
        # This is to ensure that PSF will be binned properly if number of pixels is odd
        # if img_resolution is not None:
        #     if oversampling % 2 != img_resolution % 2:
        #         oversampling += 1

        img_size = xp.ceil(img_resolution * oversampling).astype('int')
        N = xp.fix(zeroPaddingFactor * oversampling * resolution).astype('int')
        pad_width = xp.ceil((N - resolution) / 2).astype('int')

        supportPadded = xp.pad(amplitude * xp.exp(1j * phase), pad_width=((pad_width, pad_width), (pad_width, pad_width)), constant_values=0).astype(self.precision_complex())
        # make sure the number of pxels is correct after the padding
        N = supportPadded.shape[0]

        # case considering a coronograph
        if self.coronagraph_diameter is not None:
            [xx, yy] = xp.meshgrid(xp.linspace(0, N-1, N, dtype=self.precision()), xp.linspace(0, N-1, N, dtype=self.precision()))
            xxc = xx - (N-1)/2
            yyc = yy - (N-1)/2
            self.apodiser = xp.sqrt(xxc**2 + yyc**2) < self.resolution/2
            self.pupilSpiderPadded = xp.pad(self.pupil, pad_width=((pad_width, pad_width), (pad_width, pad_width)), constant_values=0).astype(self.precision_complex())
            self.focalMask = xp.sqrt(xxc**2 + yyc**2) > self.coronagraph_diameter/2 * zeroPaddingFactor
            self.lyotStop = ((xp.sqrt((xxc-1.0)**2 + (yyc-1.0)**2) < N/2 * 0.9) * self.pupilSpiderPadded)

            # PSF computation
            [xx, yy] = xp.meshgrid(xp.linspace(0, N - 1, N, dtype=self.precision()), xp.linspace(0, N - 1, N, dtype=self.precision()), copy=False)

            phasor = xp.exp(-1j * xp.pi / N * (xx + yy) * (1 - img_resolution % 2)).astype(self.precision_complex)
            #                                                        ^--- this is to account odd/even number of pixels
            # Propagate with Fourier shifting
            EMF = xp.fft.fftshift(1 / N * xp.fft.fft2(xp.fft.ifftshift(supportPadded * phasor*self.apodiser)))
            self.B = EMF * self.focalMask * phasor
            self.C = xp.fft.fftshift(
                1 * xp.fft.ifft2(xp.fft.ifftshift(self.B))).astype(self.precision_complex) * self.lyotStop * phasor
            EMF = (xp.fft.fftshift(1 * xp.fft.fft2(xp.fft.ifftshift(self.C)))).astype(self.precision_complex)

        else:
            # PSF computation
            [xx, yy] = xp.meshgrid(xp.linspace(0, N - 1, N, dtype=self.precision()), xp.linspace(0, N - 1, N, dtype=self.precision()), copy=False)
            phasor = xp.exp(-1j * xp.pi / N * (xx + yy) * (1 - img_resolution % 2)).astype(self.precision_complex())
            #                                                        ^--- this is to account odd/even number of pixels
            # Propagate with Fourier shifting
            EMF = xp.fft.fftshift(
                1 / N * xp.fft.fft2(xp.fft.ifftshift(supportPadded * phasor))).astype(self.precision_complex())

        # Again, this is to properly crop a PSF with the odd/even number of pixels
        if N % 2 == img_size % 2:
            shift_pix = 0
        else:
            if N % 2 == 0:
                shift_pix = 1
            else:
                shift_pix = -1
        # Support only rectangular PSFs
        ids = xp.array(
            [xp.ceil(N / 2) - img_size // 2 + (1 - N % 2) - 1, xp.ceil(N / 2) + img_size // 2 + shift_pix]).astype(
            xp.int32)
        EMF = EMF[ids[0]:ids[1], ids[0]:ids[1]]

        self.focal_EMF = EMF

        if oversampling != 1:
            self.PSF = set_binning(xp.abs(EMF) ** 2, oversampling)

        else:
            self.PSF = xp.abs(EMF) ** 2

        return oversampling



    def apply_spiders(self, angle, thickness_spider, offset_X=None, offset_Y=None):
        self.isInitialized = False
        if thickness_spider > 0:
            self.set_pupil()
            pup = xp.copy(self.pupil)
            max_offset = self.centralObstruction*self.D/2 - thickness_spider/2
            if offset_X is None:
                offset_X = xp.zeros(len(angle))

            if offset_Y is None:
                offset_Y = xp.zeros(len(angle))

            if xp.max(xp.abs(offset_X)) >= max_offset or xp.max(xp.abs(offset_Y)) > max_offset:
                warning('The spider offsets are too large! Weird things could happen!')
            for i in range(len(angle)):
                angle_val = (angle[i]+90) % 360
                x = xp.linspace(-self.D/2, self.D/2, self.resolution, dtype=self.precision())
                [X, Y] = xp.meshgrid(x, x)
                X += offset_X[i]
                Y += offset_Y[i]
                map_dist = xp.abs(X*xp.cos(xp.deg2rad(angle_val)) + Y*xp.sin(xp.deg2rad(-angle_val)))
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

    def showPSF(self, zoom=1, GEO=False):
        raise DeprecationWarning('This method has been deprecated.')

    @property
    def pupil(self):
        return self._pupil

    @pupil.setter
    def pupil(self, val):
        self._pupil = val.astype(bool)
        self.pixelArea = xp.sum(self._pupil)
        tmp = xp.reshape(self._pupil, self.resolution**2)
        self.pupilLogical = xp.where(tmp > 0)
        self.pupilReflectivity = self.pupil.astype(self.precision())
        if self.isInitialized:
            warning('A new pupil is now considered, its reflectivity is considered to be uniform. Assign the proper reflectivity map to tel.pupilReflectivity if required.')

    # <JM @ SpaceODT> Updating tel OPD updates the src OPD for backwards compatibility
    @property
    def OPD(self):
        return self.src.OPD

    @OPD.setter
    def OPD(self, val):
        self.src.OPD = val

    @property
    def OPD_no_pupil(self):
        return self.src.OPD_no_pupil

    @OPD.setter
    def OPD_no_pupil(self, val):
        self.src.OPD_no_pupil = val

    def resetOPD(self):
        self.src.resetOPD()

    # <\JM @ SpaceODT>


    # <JM @ SpaceODT> 
    # This function was replaced by relay functions in different objects
    # Remains here for now for reference 
    def mul(self, obj):
        print(f"Multiplying Telescope with {obj.tag}")
        # case where multiple objects are considered
        if type(obj) is list:
            wfs_signal = []
            if type(self.OPD) is list:
                if len(self.OPD) == len(obj):
                    for i_obj in range(len(self.OPD)):
                        tel_tmp = copy.deepcopy(
                            getattr(obj[i_obj], 'telescope'))
                        tel_tmp.OPD = self.OPD[i_obj]
                        tel_tmp.OPD_no_pupil = self.OPD_no_pupil[i_obj]
                        self.src.src[i_obj]*tel_tmp*obj[i_obj]
                        wfs_signal.append(obj[i_obj].signal)
                    obj[i_obj].signal = xp.mean(wfs_signal, axis=0)
                else:
                    raise OopaoError('Error! There is a mis-match between the number of Sources ('+str(
                        len(self.OPD))+') and the number of WFS ('+str(len(obj))+')')
            else:
                for i_obj in range(len(obj)):
                    self*obj[i_obj]
        else:
            # interaction with WFS object: Propagation of the phase screen
            if obj.tag == 'pyramid' or obj.tag == 'double_wfs' or obj.tag == 'shackHartmann' or obj.tag == 'bioEdge':
                self.optical_path.append([obj.tag, id(obj)])
                if self.display_optical_path is True:
                    self.print_optical_path()
                self.optical_path = self.optical_path[:-1]

                if self.src.tag == 'asterism':
                    input_source = copy.deepcopy(self.src.src)
                    output_raw_data = xp.zeros(
                        [obj.raw_data.shape[0], obj.raw_data.shape[0]])
                    if obj.tag == 'pyramid':
                        output_raw_em = xp.zeros(
                            obj.focal_plane_camera.frame.shape)
                    obj.telescope = copy.deepcopy(self)

                    for i_src in range(len(input_source)):
                        obj.telescope.src = input_source[i_src]
                        [Tip, Tilt] = xp.meshgrid(
                            xp.linspace(-xp.pi, xp.pi, self.resolution, dtype=self.precision()), xp.linspace(-xp.pi, xp.pi, self.resolution, dtype=self.precision()))
                        delta_TT = input_source[i_src].coordinates[0]*(1/((180/xp.pi)*3600))*(self.D/input_source[i_src].wavelength)*(
                            xp.cos(input_source[i_src].coordinates[1])*Tip+xp.sin(input_source[i_src].coordinates[1])*Tilt)*self.pupil
                        obj.wfs_measure(
                            phase_in=input_source[i_src].phase + delta_TT, integrate=False)
                        output_raw_data += obj.raw_data.copy()
                        if obj.tag == 'pyramid':
                            obj*obj.focal_plane_camera
                            output_raw_em += obj.focal_plane_camera.frame
                    if obj.tag == 'pyramid':
                        obj.focal_plane_camera.frame = output_raw_em
                    obj.raw_data = output_raw_data.copy()
                    obj.signal_2D, obj.signal = obj.wfs_integrate()

                else:
                    obj.telescope = self
                    obj.wfs_measure(phase_in=self.src.phase)

            if obj.tag == 'detector':
                if self.optical_path[-1] != obj.tag:
                    self.optical_path.append([obj.tag, id(obj)])

                self.computePSF(detector=obj)
                obj.fov_arcsec = self.xPSF_arcsec[1] - self.xPSF_arcsec[0]
                obj.fov_rad = self.xPSF_rad[1] - self.xPSF_rad[0]
                if obj.integrationTime is not None:
                    if obj.integrationTime < self.samplingTime:
                        raise OopaoError('The Detector integration time is smaller than the AO loop sampling Time. ')
                obj._integrated_time += self.samplingTime
                if xp.ndim(self.PSF) == 3:
                    obj.integrate(xp.sum(self.PSF, axis=0))
                else:
                    obj.integrate(self.PSF)

                self.PSF = obj.frame

            if obj.tag == 'OPD_map':
                self.optical_path.append([obj.tag, id(obj)])

                self.OPD += obj.OPD
                self.OPD_no_pupil += obj.OPD

            if obj.tag == 'NCPA':
                self.optical_path.append([obj.tag, id(obj)])

                self.OPD += obj.OPD
                self.OPD_no_pupil += obj.OPD

            if obj.tag == 'spatialFilter':
                self.optical_path.append([obj.tag, id(obj)])

                self.spatialFilter = obj
                N = obj.resolution
                EF_in = xp.zeros([N, N], dtype=self.precision_complex())
                EF_in[obj.center-self.resolution//2:obj.center+self.resolution//2, obj.center-self.resolution//2:obj.center +
                      self.resolution//2] = self.pupilReflectivity*xp.exp(1j*(self.OPD_no_pupil*2*xp.pi/self.src.wavelength))
                FP_in = xp.fft.fft2(EF_in)
                FP_filtered = FP_in*xp.fft.fftshift(obj.mask)
                em_field = xp.fft.ifft2(FP_filtered)
                self.em_field_filtered = em_field[obj.center-self.resolution//2:obj.center +
                                                  self.resolution//2, obj.center-self.resolution//2:obj.center+self.resolution//2]
                # self.phase_filtered = xp.arctan2(xp.imag(self.em_field_filtered),xp.real(self.em_field_filtered))*self.pupil
                self.phase_filtered = (
                    (xp.angle(self.em_field_filtered)))*self.pupil

                self.amplitude_filtered = xp.abs(self.em_field_filtered)
                return self

            if obj.tag == 'deformableMirror':
                if self.optical_path[-1][1] != id(obj):
                    self.optical_path.append([obj.tag, id(obj)])

                pupil = xp.atleast_3d(self.pupil)

                if self.src.tag == 'source':
                    self.OPD_no_pupil = obj.dm_propagation(self)
                    if xp.ndim(self.OPD_no_pupil) == 2:
                        self.OPD = self.OPD_no_pupil*self.pupil
                    else:
                        self.OPD = self.OPD_no_pupil*pupil
                else:
                    if self.src.tag == 'asterism':
                        if len(self.OPD) == self.src.n_source:
                            for i in range(self.src.n_source):
                                if obj.altitude is not None:
                                    self.OPD_no_pupil[i] = obj.dm_propagation(
                                        self, OPD_in=self.OPD_no_pupil[i], i_source=i)
                                else:
                                    self.OPD_no_pupil[i] = obj.dm_propagation(
                                        self, OPD_in=self.OPD_no_pupil[i])
                                if xp.ndim(self.OPD_no_pupil[i]) == 2:
                                    self.OPD[i] = self.OPD_no_pupil[i] * \
                                        self.pupil
                                else:
                                    self.OPD[i] = self.OPD_no_pupil[i]*pupil
                            self.OPD = self.OPD
                            self.OPD_no_pupil = self.OPD_no_pupil
                        else:
                            raise OopaoError('The lenght of the OPD list ('+str(len(self._OPD_no_pupil))
                                             + ') does not match the number of sources ('+str(self.src.n_source)+')')
                    else:
                        raise OopaoError('The wrong object was attached to the telescope')
        return self


    # <JM @ SpaceODT> This is no longer needed as the atmosphere behaves as its own entity now.
    # Remains here for now for reference 
    # Combining with an atmosphere object
    def __add__(self, obj):
        if obj.tag == 'atmosphere':

            obj*self
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
            self.src.resetOPD()
            obj.asterism = None

        if obj.tag == 'spatialFilter':
            self.spatialFilter = None
            print('Telescope and Spatial Filter separated!')
    
    # <\JM @ SpaceODT>
    

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
        self.prop['surface'] = f"{'Surface [m²]':<25s}|{self.pixelSize:^10.2f}"
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
