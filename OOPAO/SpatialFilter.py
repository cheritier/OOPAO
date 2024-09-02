# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:18 2021

@author: cheritie
"""

#%%
import numpy as np
#
class SpatialFilter():
    def __init__(self,telescope, shape, diameter, zeroPaddingFactor =2):
        self.tag='spatialFilter'
        self.telescope_resolution = telescope.resolution
        self.diameter             = diameter
        self.shape                = shape
        # set up an initial filter
        self.set_spatial_filter(zeroPaddingFactor = zeroPaddingFactor)

        
    def set_spatial_filter(self,zeroPaddingFactor):
        self.diameter_padded    = self.diameter*zeroPaddingFactor
        self.resolution         = int(self.telescope_resolution*zeroPaddingFactor) 
        self.center             = self.resolution//2
        valid_input             = False
        
        if self.shape == 'circular':
            D               = self.resolution
            x               = np.linspace(-D//2,D//2,D)
            xx,yy           = np.meshgrid(x,x)
            R               = xx**2+yy**2
            SF              = R<=(self.diameter_padded)**2 
            self.mask       = (SF +1j*SF)/np.sqrt(2)
            valid_input     = True
                        
        if self.shape == 'square':
            SF              = np.zeros([self.resolution,self.resolution],dtype=float)
            SF[self.center-self.diameter_padded//2:self.center+self.diameter_padded//2,self.center-self.diameter_padded//2:self.center+self.diameter_padded//2] = 1
            self.mask       = (SF +1j*SF)/np.sqrt(2)
            valid_input     = True


        if self.shape == 'foucault':
            SF       = np.zeros([self.resolution,self.resolution],dtype=float)
            SF[:self.center] = 1
            self.mask  = (SF +1j*SF)/np.sqrt(2)            
            valid_input     = True
        
        if valid_input is False:
            raise ValueError("The input shape: '"+str(self.shape)+"' is not valid. The valid inputs are: 'circular', 'square', 'foucault'.")
            
        self.mask[:self.resolution-1,:self.resolution-1]=self.mask[1:,1:]
        
        return