# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:31:33 2020

@author: cheritie
"""

import inspect

import aotools as ao
import numpy as np


# =============================================================================
#                               CLASS DEFINITION
# =============================================================================

class Zernike:
    def __init__(self, telObject, J=1):
        self.resolution         = telObject.resolution
        self.D                  = telObject.D
        self.centralObstruction = telObject.centralObstruction    
        self.nModes             = J
    
    def zernike_tel(self, tel, j):
        """
         Creates the Zernike polynomial with radial index, n, and azimuthal index, m.
    
         Args:
            n (int): The radial order of the zernike mode
            m (int): The azimuthal order of the zernike mode
            N (int): The diameter of the zernike more in pixels
         Returns:
            ndarray: The Zernike mode
         """
        X, Y = np.where(tel.pupil > 0)
                            
        X = ( X-(tel.resolution + tel.resolution%2-1)/2 ) / tel.resolution * tel.D
        Y = ( Y-(tel.resolution + tel.resolution%2-1)/2 ) / tel.resolution * tel.D
        #                                          ^- to properly allign coordinates relative to the (0,0) for even/odd telescope resolutions
        R = np.sqrt(X**2 + Y**2)
        R = R/R.max()
        theta = np.arctan2(Y, X)
        out = np.zeros([tel.pixelArea,j])
        outFullRes = np.zeros([tel.resolution**2, j])

        for i in range(1, j+1):
            n, m = ao.zernike.zernIndex(i+1)
            if m == 0:
                Z = np.sqrt(n+1) * ao.zernike.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = np.sqrt(2*(n+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * np.cos(m * theta)
                else:   #i is odd
                    m = abs(m)
                    Z = np.sqrt(2*(n+1)) * ao.zernike.zernikeRadialFunc(n, m, R) * np.sin(m * theta)
            
            Z -= Z.mean()
            Z *= (1/np.std(Z))

            # clip
            out[:, i-1] = Z
            outFullRes[tel.pupilLogical, i-1] = Z
            
        outFullRes = np.reshape( outFullRes, [tel.resolution, tel.resolution, j] )
        return out, outFullRes
    
    def computeZernike(self, telObject2):
        self.modes, self.modesFullRes = self.zernike_tel(telObject2, self.nModes)      
        # normalize modes  

    def modeName(self, index):
        modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X-shaped)', 'Astigmatism (+-shaped)',
            'Coma vertical', 'Coma horizontal', 'Trefoil vertical', 'Trefoil horizontal',
            'Sphere', 'Secondary astigmatism (X-shaped)', 'Secondary astigmatism (+-shaped)',
            'Quadrofoil vertical', 'Quadrofoil horizontal',
            'Secondary coma horizontal', 'Secondary coma vertical',
            'Secondary trefoil horizontal', 'Secondary trefoil vertical',
            'Pentafoil horizontal', 'Pentafoil vertical'
        ]
        
        if index < 0:
            return('Incorrent index!')
        elif index >= len(modes_names):
            return('Z', index+2)
        else:
            return(modes_names[index])


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
                            
                            