# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:37:29 2020

@author: cheritie
"""

import numpy as np
import time

class Detector:
    def __init__(self,nRes:int,readoutNoise:float=0,photonNoise:bool=False,backgroundNoise:bool=False,backgroundNoiseMap:float = None,QE:float=1):
        '''

        Parameters
        ----------
        nRes : int
            Resolution of the detector in pixels.
        readoutNoise : float, optional
            Read-Out Noise from the detector given in electron variance. The default is 0.
        photonNoise : bool, optional
            Flag to enable or disable the photon noise. The default is False.
        QE : float, optional
            Quantum efficiency of the detector. The default is 1.

        Returns
        -------
        None.

        '''
        
        self.resolution         = nRes
        self.QE                 = QE
        self.readoutNoise       = readoutNoise
        self.photonNoise        = photonNoise        
        self.backgroundNoise    = backgroundNoise   
            
        self.frame        = np.zeros([nRes,nRes])
        self.tag          = 'detector'        
        # random state to create random values for the noise
        self.random_state_photon_noise  = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background    = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        
    def rebin(self,arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])        
        out = (arr.reshape(shape).mean(-1).mean(1)) * (arr.shape[0] // new_shape[0]) * (arr.shape[1] // new_shape[1])        
        return out
    
    def integrate(self,frame):
        
        self.frame = frame
        
        # apply photon noise 
        if self.photonNoise!=0:
            self.frame = self.random_state_photon_noise.poisson(self.frame)

        # apply background noise
        if self.backgroundNoise is True:    
            if hasattr(self,'backgroundNoiseMap') is False or self.backgroundNoiseMap is None:
                raise ValueError('The background map backgroundNoiseMap is not properly set. A map of shape '+str(self.frame.shape)+' is expected')
            else:
                self.backgroundNoiseAdded = self.random_state_background.poisson(self.backgroundNoiseMap)
                self.frame +=self.backgroundNoiseAdded

        # apply quantum efficiency            
        self.frame *= self.QE
        
        # apply readout Noise
        if self.readoutNoise!=0:
            self.frame += np.int64(np.round(self.random_state_readout_noise.randn(self.resolution,self.resolution)*self.readoutNoise))                
        


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

            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def print_properties(self):
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DETECTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^18s}'.format('Resolution')           + '{: ^18s}'.format(str(self.resolution))                   +'{: ^18s}'.format('[pixels]'   ))
        print('{: ^18s}'.format('Readout Noise')        + '{: ^18s}'.format(str(self.readoutNoise))                 +'{: ^18s}'.format('[e-]'   ))
        print('{: ^18s}'.format('Photon Noise')         + '{: ^18s}'.format(str(self.photonNoise))                       +'{: ^18s}'.format(''   ))
        print('{: ^18s}'.format('Background Noise')     + '{: ^18s}'.format(str(self.backgroundNoise))                   +'{: ^18s}'.format('[lamda/D]'  ))
        print('{: ^18s}'.format('Quantum Efficiency')   + '{: ^18s}'.format(str(self.QE))                           +'{: ^18s}'.format('[lamda/D]'  ))

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def __repr__(self):
        self.print_properties()
        return ' '