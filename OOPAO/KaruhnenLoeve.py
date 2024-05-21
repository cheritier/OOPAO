# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:33:57 2023

@author: cheritier
"""

import inspect
from .calibration.compute_KL_modal_basis import compute_KL_basis
import numpy as np

# =============================================================================
#                               CLASS DEFINITION
# =============================================================================

class KarhunenLoeve:
    def __init__(self, atmosphere, telescope, deformable_mirror = None,n_modes= None):
        self.resolution         = telescope.resolution
        self.nModes             = None
        
        if deformable_mirror is None:
            from.DeformableMirror import DeformableMirror
            # create a virtual dm
            virtual_modes = np.eye(telescope.resolution**2)
            deformable_mirror = DeformableMirror(telescope, nSubap= 2,modes = virtual_modes)
            
            
        # use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
        self.M2C = compute_KL_basis(telescope, atmosphere, deformable_mirror)
        
        
        
    
 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag + ':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))