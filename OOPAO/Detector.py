# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:18:03 2024

@authors: astriffl & cheritie
"""

import numpy as np
import time

class Detector:
    def __init__(self,
                 nRes:int=10,
                 integrationTime:float=None,
                 bits:int=None,
                 FWC:int=None,
                 gain:int=1,
                 sensor:str='CCD',
                 QE:float=1,
                 binning:int=1,
                 psf_sampling:float=2,
                 darkCurrent:float=0,
                 readoutNoise:float=0,
                 photonNoise:bool=False,
                 backgroundNoise:bool=False,
                 backgroundNoiseMap:float=None):
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
            Full Well Capacity of the pixels in [counts] to simulate the 
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
        backgroundNoiseMap : float, optional
            Background 2D map to consider to apply the background noise.
            The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.resolution         = nRes
        self.integrationTime    = integrationTime
        self.bits               = bits
        self.FWC                = FWC
        self.gain               = gain
        self.sensor             = sensor
        self.psf_sampling       = psf_sampling
        if self.sensor not in ['EMCCD','CCD','CMOS']:
            raise ValueError("Sensor must be 'EMCCD', 'CCD', or 'CMOS'")
        self.QE                 = QE
        self.binning            = binning
        self.darkCurrent        = darkCurrent
        self.readoutNoise       = readoutNoise
        self.photonNoise        = photonNoise        
        self.backgroundNoise    = backgroundNoise   
        self.backgroundNoiseMap = backgroundNoiseMap
        self.frame              = np.zeros([nRes,nRes])
        self.saturation         = 0
        self.tag                = 'detector'   
        self.buffer_frame       = []
        self._integrated_time   = 0
        self.fov_arcsec         = None
        self.pixel_size_rad     = None
        self.pixel_size_arcsec  = None
        

        
        
        # random state to create random values for the noise
        self.random_state_photon_noise      = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise     = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background_noise  = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_dark_noise        = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.print_properties()

    def rebin(self,arr, new_shape):
            shape = (new_shape[0], arr.shape[0] // new_shape[0],
                     new_shape[1], arr.shape[1] // new_shape[1])        
            out = (arr.reshape(shape).mean(-1).mean(1)) * (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])        
            return out
    def set_binning(self, M, binning_factor,mode='sum'):
        if M.shape[0]%binning_factor == 0:
            if M.ndim == 2:
                new_shape = [int(M.shape[0]//binning_factor), int(M.shape[1]//binning_factor)]
                shape = (new_shape[0], M.shape[0] // new_shape[0], 
                         new_shape[1], M.shape[1] // new_shape[1])
                if mode == 'sum':
                    return M.reshape(shape).sum(-1).sum(1)
                else:
                    return M.reshape(shape).mean(-1).mean(1)
            else:
                new_shape = [int(M.shape[0]//binning_factor), int(M.shape[1]//binning_factor), M.shape[2]]
                shape = (new_shape[0], M.shape[0] // new_shape[0], 
                         new_shape[1], M.shape[1] // new_shape[1], new_shape[2])
                if mode == 'sum':
                    return M.reshape(shape).sum(-2).sum(1)
                else:
                    return M.reshape(shape).mean(-2).mean(1)
        else:
            raise ValueError('Binning factor %d not compatible with detector size'%(binning_factor))

    def set_sampling(self,array):
        sx, sy = array.shape
        pad_x = int(np.round((sx * (self.psf_sampling-1)) / 2))
        pad_y = int(np.round((sy * (self.psf_sampling-1)) / 2))
        array_padded = np.pad(array, (pad_x,pad_y))
        return array_padded
    
    def conv_photon_electron(self,frame):
        frame = (frame * self.QE)
        return frame
        
        
    def set_saturation(self,frame):
        self.saturation = (100*frame.max()/self.FWC)
        if frame.max() > self.FWC:
            print('Warning: the detector is saturating, %.1f %%'%self.saturation)
        return np.clip(frame, a_min = 0, a_max = self.FWC)
    
    
    def digitalization(self,frame):
        self.quantification_noise = self.FWC * 2**(-self.bits) / np.sqrt(12)
        if self.FWC is None:
            return (frame / frame.max() * 2**self.bits).astype(int)
        else:
            self.saturation = (100*frame.max()/self.FWC)
            if frame.max() > self.FWC:
                print('Warning: the detector is saturating (gain applyed %i), %.1f %%'%(self.gain,self.saturation))
            frame = (frame / self.FWC * (2**self.bits-1)).astype(int) 
            return np.clip(frame, a_min=frame.min(), a_max=2**self.bits-1)

    
    def set_photon_noise(self,frame):
        self.photon_noise = np.sqrt(self.signal)
        return self.random_state_photon_noise.poisson(frame)


    def set_background_noise(self,frame):
        if hasattr(self,'backgroundNoiseMap') is False or self.backgroundNoiseMap is None:
            raise ValueError('The background map backgroundNoiseMap is not properly set. A map of shape '+str(frame.shape)+' is expected')
        else:
            self.backgroundNoiseAdded = self.random_state_background.poisson(self.backgroundNoiseMap)
            frame += self.backgroundNoiseAdded
            return frame
        
        
    def set_readout_noise(self,frame):
        noise = (np.round(self.random_state_readout_noise.randn(frame.shape[0],frame.shape[1])*self.readoutNoise)).astype(int)  #before np.int64(...)
        frame += noise
        return frame
    
    
    def set_dark_shot_noise(self,frame):
        self.dark_shot_noise = np.sqrt(self.darkCurrent * self.integrationTime) 
        return frame 
    
    def readout(self):
            frame = np.sum(self.buffer_frame,axis=0)   
                        
            if self.darkCurrent!=0:
                frame = self.set_dark_shot_noise(frame)
            
            # Simulate the saturation of the detector (without blooming and smearing)
            if self.FWC is not None:
                frame = self.set_saturation(frame)    
            
            # If the sensor is EMCCD the applyed gain is before the analog-to-digital conversion
            if self.sensor == 'EMCCD': 
                frame = self.set_photon_noise(frame) * self.gain
    
            # Simulate hardware binning of the detector
            if self.binning != 1:
                frame = self.set_binning(frame,self.binning)
           
            # Apply readout noise
            if self.readoutNoise!=0:    
                frame = self.set_readout_noise(frame)    

            # Apply the CCD/CMOS gain
            if self.sensor == 'CCD' or self.sensor == 'CMOS':
                frame *= self.gain
                
            # Apply the digital quantification of the detector
            if self.bits is not None:
                frame = self.digitalization(frame)
            
            # Save the integrated frame and buffer
            self.frame  = frame.copy()
            self.buffer = self.buffer_frame.copy()
            self.resolution       = self.frame.shape[0]
            if self.fov_arcsec is not None:
                self.pixel_size_rad     = self.fov_rad/self.resolution 
                self.pixel_size_arcsec  = self.fov_arcsec/self.resolution
            
            # reset the buffer and _integrated_time property
            self.buffer_frame     = []
            self._integrated_time = 0
            return 
    
    def integrate(self,frame):
        self.perfect_frame = frame.copy()
        self.flux_max_px = self.perfect_frame.max() 
        self.signal = self.QE * self.flux_max_px
        
        # Apply photon noise 
        if self.photonNoise!=0:
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
            if self._integrated_time>=self.integrationTime: 
                self.readout()



    def computeSNR(self):
        if self.FWC is not None:
            self.SNR_max = self.FWC / np.sqrt(self.FWC)
        else:
            self.SNR_max = np.NaN
        
        self.SNR = self.signal / np.sqrt(self.quantification_noise**2 + self.photon_noise**2 + self.readoutNoise**2 + self.dark_shot_noise**2) 
        print()
        print('Theoretical maximum SNR: %.2f'%self.SNR_max)
        print('Current SNR: %.2f'%self.SNR)
        
        # self.SNR = I*self.QE*self.tInt / np.sqrt(I*self.QE*self.tInt + self.darkCurrent*self.tInt + self.readoutNoise**2)
    
    def displayNoiseError(self):
        print()
        print('------------ Noise error ------------')
        if self.bits is not None:
            print('{:^25s}|{:^9.4f}'.format('Quantization noise [e-]',self.quantification_noise))
        if self.photonNoise is True:
            print('{:^25s}|{:^9.4f}'.format('Photon noise [e-]',self.photon_noise))
        if self.darkCurrent!=0:    
            print('{:^25s}|{:^9.4f}'.format('Dark shot noise [e-]',self.dark_shot_noise))
        if self.readoutNoise!=0:
            print('{:^25s}|{:^9.1f}'.format('Readout noise [e-]',self.readoutNoise))
        print('----------------------------------')
        pass
    @property
    def backgroundNoise(self):
        return self._backgroundNoise
    
    @backgroundNoise.setter
    def backgroundNoise(self,val):
        self._backgroundNoise = val
        if val == True:
            if hasattr(self,'backgroundNoiseMap') is False or self.backgroundNoiseMap is None:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Warning: The background noise is enabled but no property backgroundNoiseMap is set.\nA map of shape '+str(self.frame.shape)+' is expected')
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            else:
                print('Background Noise enabled! Using the following backgroundNoiseMap:')
                print(self.backgroundNoiseMap)
    @property
    def integrationTime(self):
        return self._integrationTime
      
    @integrationTime.setter
    def integrationTime(self,val):          
        self._integrationTime = val
        self._integrated_time = 0
        self.buffer_frame = []
        
                
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def print_properties(self):
        print()
        print('------------ Detector ------------')
        print('{:^25s}|{:^9s}'.format('Sensor type',self.sensor))
        print('{:^25s}|{:^9d}'.format('Resolution [px]',self.resolution//self.binning))
        if self.integrationTime is not None:
            print('{:^25s}|{:^9.4f}'.format('Exposure time [s]',self.integrationTime))
        if self.bits is not None:
            print('{:^25s}|{:^9d}'.format('Quantization [bits]',self.bits))
        if self.FWC is not None:
            print('{:^25s}|{:^9d}'.format('Full well capacity [e-]',self.FWC))
        print('{:^25s}|{:^9d}'.format('Gain',self.gain))
        print('{:^25s}|{:^9d}'.format('Quantum efficiency [%]',int(self.QE*100)))
        print('{:^25s}|{:^9s}'.format('Binning',str(self.binning)+'x'+str(self.binning)))
        print('{:^25s}|{:^9d}'.format('Dark current [e-/pixel/s]',self.darkCurrent))
        print('{:^25s}|{:^9s}'.format('Photon noise',str(self.photonNoise)))
        print('{:^25s}|{:^9s}'.format('Bkg noise [e-]',str(self.backgroundNoise)))
        print('{:^25s}|{:^9.1f}'.format('Readout noise [e-/pixel]',self.readoutNoise))
        print('----------------------------------')
    def __repr__(self):
        self.print_properties()
        return ' '