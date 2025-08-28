# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:18:03 2024

@authors: astriffl & cheritie
"""

import numpy as np
import time
from OOPAO.tools.tools import set_binning, warning, OopaoError


class Detector:
    def __init__(self,
                 nRes: int = None,
                 integrationTime: float = None,
                 bits: int = None,
                 output_precision: int = None,
                 FWC: int = None,
                 gain: int = 1,
                 sensor: str = 'CCD',
                 QE: float = 1.,
                 binning: int = 1,
                 psf_sampling: float = 2.,
                 darkCurrent: float = 0.,
                 readoutNoise: float = 0,
                 photonNoise: bool = False,
                 backgroundNoise: bool = False,
                 backgroundFlux: float = None,
                 backgroundMap: float = None):
        '''
        The Detector allows to simulate the effects ot a real detector (noise, quantification...).


        Parameters
        ----------
        nRes : int
            Resolution in pixel of the detector. This value is ignored for the computation of PSFs using the Telescope class (see Telescope class for further documentation).
            In that case, the sampling of the detector is driven by the psf_sampling property
        integrationTime : float, optional
        Integration time of the detector object in [s].

            - If integrationTime is None, the value is set to the AO loop
            frequency defined by the samplingTime property of the Telescope.

            - If integrationTime >= samplingTime is requested, the Detector
            frames are concatenated into the buffer_frames property.
            When the integration is completed, the frames are summed together
            and readout by the Detector.

            - If integrationTime < samplingTime an error is raised.

            The default is None.
        bits : int, optional
            Quantification of the pixel in [bits] to simulate the finite
            precision of the Detector. If set to None the effect is ignored
            The default is None.
        FWC : int, optional
            Full Well Capacity of the pixels in [e-] to simulate the
            saturation of the pixels. If set to None the effect is ignored.
            The default is None.
        gain : int, optional
            Gain of the detector. The default is 1.
        sensor : str, optional
            Flag to specify if the sensor is a CCD/CMOS/EMCCD. This is used to
            simulate the associated noise effects when the gain property is set.
            The default is 'CCD'.
        QE : float, optional
            Quantum efficiency of the Detector. The default is 1.
        binning : int, optional
            Binning factor of the Detector. The default is 1.
        psf_sampling : float, optional
            ZeroPadding factor of the FFT to compute PSFs from a Telescope (see Telescope class for further documentation).
            The default is 2 (Shannon-sampled PSFs).
        darkCurrent : float, optional
            Dark current of the Detector in [e-/pixel/s]. The default is 0.
        readoutNoise : float, optional
            Readout noise of the detector in [e-/pixel]. The default is 0.
        photonNoise : bool, optional
            Flag to apply the photon noise to the detector frames.
            The default is False.
        backgroundNoise : bool, optional
            Flag to apply the background Noise to the detector frames.
            The default is False.
        backgroundFlux : float, optional
            Background 2D map to consider to apply the background noise.
            The default is None.
        backgroundFlux : float, optional
            Background 2D map to consider to be substracted to each frame.
            The default is None.

        -------
        None.

        '''
        self.resolution = nRes
        self.integrationTime = integrationTime
        self.bits = bits
        self.output_precision = output_precision
        self.FWC = FWC
        self.gain = gain
        self.sensor = sensor
        self.psf_sampling = psf_sampling
        if self.sensor not in ['EMCCD', 'CCD', 'CMOS']:
            raise OopaoError("Sensor must be 'EMCCD', 'CCD', or 'CMOS'")
        self.QE = QE
        self.binning = binning
        self.darkCurrent = darkCurrent
        self.readoutNoise = readoutNoise
        self.photonNoise = photonNoise
        self.backgroundNoise = backgroundNoise
        self.backgroundFlux = backgroundFlux
        self.backgroundMap = backgroundMap
        if self.resolution is not None:
            self.frame = np.zeros([self.resolution, self.resolution])
        else:
            self.frame = None

        self.saturation = 0
        self.tag = 'detector'
        self.buffer_frame = []
        self._integrated_time = 0
        self.fov_arcsec = None
        self.pixel_size_rad = None
        self.pixel_size_arcsec = None

        # noise initialisation
        self.quantification_noise = 0
        self.photon_noise = 0
        self.dark_shot_noise = 0

        # random state to create random values for the noise
        self.random_state_photon_noise = np.random.RandomState(
            seed=int(time.time()))      # random states to reproduce sequences of noise
        self.random_state_readout_noise = np.random.RandomState(
            seed=int(time.time()))      # random states to reproduce sequences of noise
        self.random_state_background_noise = np.random.RandomState(
            seed=int(time.time()))      # random states to reproduce sequences of noise
        self.random_state_dark_shot_noise = np.random.RandomState(
            seed=int(time.time()))      # random states to reproduce sequences of noise
        self.print_properties()

    def rebin(self, arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        out = (arr.reshape(shape).mean(-1).mean(1)) * \
            (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])
        return out

    def set_binning(self, array, binning_factor, mode='sum'):
        frame = set_binning(array, binning_factor, mode)
        return frame

    def set_sampling(self, array):
        sx, sy = array.shape
        pad_x = int(np.round((sx * (self.psf_sampling-1)) / 2))
        pad_y = int(np.round((sy * (self.psf_sampling-1)) / 2))
        array_padded = np.pad(array, (pad_x, pad_y))
        return array_padded

    def set_output_precision(self):
        if self.output_precision is None:
            value = self.bits

            if value == 8:
                self.output_precision = np.uint8
                self.clip_unsigned = 0
            elif value == 16:
                self.output_precision = np.uint16
                self.clip_unsigned = 0
            elif value == 32:
                self.output_precision = np.uint32
                self.clip_unsigned = 0
            elif value == 64:
                self.output_precision = np.uint64
                self.clip_unsigned = 0
            else:
                self.output_precision = float
                self.clip_unsigned = -np.inf

        return

    def conv_photon_electron(self, frame):
        frame = (frame * self.QE)
        return frame

    def set_saturation(self, frame):
        self.saturation = (100*frame.max()/self.FWC)
        if frame.max() > self.FWC:
            warning('The detector is saturating, %.1f %%' %
                    self.saturation)
        return np.clip(frame, a_min=0, a_max=self.FWC)

    def digitalization(self, frame):
        if self.FWC is None:
            return (frame / frame.max() * 2**self.bits)
            self.quantification_noise = 0
        else:
            self.quantification_noise = self.FWC * \
                2**(-self.bits) / np.sqrt(12)
            self.saturation = (100*frame.max()/self.FWC)
            if frame.max() > self.FWC:
                warning('The ADC is saturating (gain applyed %i), %.1f %%' % (
                    self.gain, self.saturation))
            frame = (frame / self.FWC * (2**self.bits-1)
                     )
            return np.clip(frame, a_min=max(frame.min(),self.clip_unsigned), a_max=2**self.bits-1)

    def set_photon_noise(self, frame):
        self.photon_noise = np.sqrt(self.signal)
        return self.random_state_photon_noise.poisson(frame)

    def set_background_noise(self, frame):
        if hasattr(self, 'backgroundFlux') is False or self.backgroundFlux is None:
            raise OopaoError('The background map backgroundFlux is not properly set. A map of shape '+str(frame.shape)+' is expected')
        else:
            self.backgroundNoiseAdded = self.random_state_background.poisson(
                self.backgroundFlux)
            frame += self.backgroundNoiseAdded
            return frame

    def set_readout_noise(self, frame):
        noise = (np.round(self.random_state_readout_noise.randn(
            frame.shape[0], frame.shape[1])*self.readoutNoise)).astype(int)  # before np.int64(...)
        frame += noise
        return frame

    def set_dark_shot_noise(self, frame):
        self.dark_shot_noise = np.sqrt(self.darkCurrent * self.integrationTime)
        dark_current_map = np.ones(frame.shape) * \
            (self.darkCurrent * self.integrationTime)
        dark_shot_noise_map = self.random_state_dark_shot_noise.poisson(
            dark_current_map)
        frame += dark_shot_noise_map
        return frame

    def remove_bakground(self, frame):
        try:
            frame -= self.backgroundMap
            return frame
        except:
            raise OopaoError('The shape of the backgroun map does not match the detector frame resolution')

    def readout(self):
        frame = np.sum(self.buffer_frame, axis=0)

        if self.darkCurrent != 0:
            frame = self.set_dark_shot_noise(frame)

        # Simulate the saturation of the detector (without blooming and smearing)
        if self.FWC is not None:
            frame = self.set_saturation(frame)

        # If the sensor is EMCCD the applyed gain is before the analog-to-digital conversion
        if self.sensor == 'EMCCD':
            frame *= self.gain

        # Simulate hardware binning of the detector
        if self.binning != 1:
            frame = set_binning(frame, self.binning)

        # set precision of output
        self.set_output_precision()

        # Apply readout noise
        if self.readoutNoise != 0:
            frame = self.set_readout_noise(frame)

        # Apply the CCD/CMOS gain
        if self.sensor == 'CCD' or self.sensor == 'CMOS':
            frame *= self.gain

        # Apply the digital quantification of the detector
        if self.bits is not None:
            frame = self.digitalization(frame)

        frame = frame.astype(self.output_precision)

        # Remove the dark fromthe detector
        if self.backgroundMap is not None:
            frame = self.remove_bakground(frame)
        # Save the integrated frame and buffer
        self.frame = frame.copy()
        self.buffer = self.buffer_frame.copy()
        if self.resolution is None:
            self.resolution = self.frame.shape[0]*self.binning
        if self.fov_arcsec is not None:
            self.pixel_size_rad = self.fov_rad/self.resolution*self.binning
            self.pixel_size_arcsec = self.fov_arcsec/self.resolution*self.binning

        # reset the buffer and _integrated_time property
        self.buffer_frame = []
        self._integrated_time = 0

    def integrate(self, frame):
        self.perfect_frame = frame.copy()
        self.flux_max_px = self.perfect_frame.max()
        self.signal = self.QE * self.flux_max_px
        # Apply photon noise
        if self.photonNoise != 0:
            frame = self.set_photon_noise(frame)

        # Apply background noise
        if self.backgroundNoise is True:
            frame = self.set_background_noise(frame)

        # Simulate the quantum efficiency of the detector (photons to electrons)
        frame = self.conv_photon_electron(frame)

        self.buffer_frame.append(frame)

        if self.integrationTime is None:
            self.readout()
        else:            
            if self.frame is None:
               self.frame =  self.perfect_frame.copy()*0
            if self._integrated_time >= self.integrationTime:
                self.readout()

    def computeSNR(self):
        if self.FWC is not None:
            self.SNR_max = self.FWC / np.sqrt(self.FWC)
        else:
            self.SNR_max = np.NaN

        self.SNR = self.signal / np.sqrt(self.quantification_noise**2 +
                                         self.photon_noise**2 + self.readoutNoise**2 + self.dark_shot_noise**2)
        print()
        print('Theoretical maximum SNR: %.2f' % self.SNR_max)
        print('Current SNR: %.2f' % self.SNR)

    def displayNoiseError(self):
        print()
        print('------------ Noise error ------------')
        if self.bits is not None:
            print('{:^25s}|{:^9.4f}'.format(
                'Quantization noise [e-]', self.quantification_noise))
        if self.photonNoise is True:
            print('{:^25s}|{:^9.4f}'.format(
                'Photon noise [e-]', self.photon_noise))
        if self.darkCurrent != 0:
            print('{:^25s}|{:^9.4f}'.format(
                'Dark shot noise [e-]', self.dark_shot_noise))
        if self.readoutNoise != 0:
            print('{:^25s}|{:^9.1f}'.format(
                'Readout noise [e-]', self.readoutNoise))
        print('-------------------------------------')
        pass

    def __mul__(self, obj) -> None:
        if obj.tag == 'GSC':
            if obj.calibration_ready is False:
                obj.calibration(self.frame)
                obj.detector_properties = self.properties()
            else:
                obj.compute_optical_gains(self.frame)
        else:
            raise OopaoError(f'Coupled object should be a "GSC" but is {obj.tag}')

    @property
    def backgroundNoise(self):
        return self._backgroundNoise

    @backgroundNoise.setter
    def backgroundNoise(self, val):
        self._backgroundNoise = val
        if val is True:
            if hasattr(self, 'backgroundFlux') is False or self.backgroundFlux is None:
                warning('The background noise is enabled but no property backgroundFlux is set.\nA map of shape ' +
                        str(self.frame.shape)+' is expected')
            else:
                print('Background Noise enabled! Using the following backgroundFlux:')
                print(self.backgroundFlux)

    @property
    def integrationTime(self):
        return self._integrationTime

    @integrationTime.setter
    def integrationTime(self, val):
        self._integrationTime = val
        self._integrated_time = 0
        self.buffer_frame = []

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['sensor'] = f"{'Sensor type':<25s}|{self.sensor:^9s}"
        if self.resolution is not None:
            self.prop['resolution'] = f"{'Resolution [px]':<25s}|{int(self.resolution//self.binning):^9d}"
        if self.integrationTime is not None:
            self.prop['exposure'] = f"{'Exposure time [ms]':<25s}|{self.integrationTime*1000:^9.2f}"
        if self.bits is not None:
            self.prop['quantization'] = f"{'Quantization [bits]':<25s}|{self.bits:^9d}"
        if self.FWC is not None:
            self.prop['FWC'] = f"{'Full well capacity [e-]':<25s}|{self.FWC:^9d}"
        self.prop['gain'] = f"{'Gain':<25s}|{self.gain:^9d}"
        self.prop['QE'] = f"{'Quantum efficiency [%]':<25s}|{int(self.QE*100):^9d}"
        self.prop['binning'] = f"{'Binning':<25s}|{str(self.binning)+'x'+str(self.binning):^9s}"
        self.prop['dark_current'] = f"{'Dark current [e-/px/s]':<25s}|{self.darkCurrent:^9.2f}"
        self.prop['photon_noise'] = f"{'Photon noise':<25s}|{str(self.photonNoise):^9s}"
        self.prop['bkg_noise'] = f"{'Bkg noise [e-]':<25s}|{str(self.backgroundNoise):^9s}"
        self.prop['readout_noise'] = f"{'Readout noise [e-/px]':<25s}|{self.readoutNoise:^9.1f}"
        return self.prop

    def __repr__(self) -> str:
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values()))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'

        title = f'{"Detector":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
