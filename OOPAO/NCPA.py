# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:22:10 2022

@author: cheritier
"""
import numpy as np
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class NCPA:
    
    def __init__(self,telescope, zernike_coefficients = None):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        An NCPA object consists in defining the 2D map that acts as a static OPD offset. It requires the following parameters
        _ telescope              : the telescope object that contains all the informations (diameter, pupil, etc)
                
        ************************** OPTIONAL PARAMETERS **************************
        
        _ zernike_coefficients  : a list of coefficients of Zernike Polynomials (see Zernike class). The coefficients are normalized to 1 m. 

        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ NCPA.OPD       : the optical path difference

        ************************** EXEMPLE **************************
        
        1) Create a blank NCPA object corresponding to a Telescope object
        ncpa = NCPA(telescope = tel)
        
        2) Update the OPD of the NCPA object using a given OPD_map
        ncpa.OPD = OPD_map
        
        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8) 
        src*tel
        
        3) Create an NCPA object corresponding based on a linear combinaison of Zernike coefficients 
        list_coefficients = [0,0,10e-9,20e-9]
        ncpa = NCPA(telescope = tel, zernike_coefficients = list_coefficients)
        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        self.isInitialized               = False                # Resolution of the telescope
        if zernike_coefficients is None:
            self.OPD = self.pupil.astype(float)
        else:
            if type(zernike_coefficients) is list:
                n_max = len(zernike_coefficients)
                from OOPAO.Zernike import Zernike
                self.Z = Zernike(telescope,J=n_max)
                self.Z.computeZernike(telescope)
                
            else:
                raise TypeError('The zernike coefficients should be input as a list.')
                
                
        self.OPD = np.matmul(self.Z.modesFullRes,np.asarray(zernike_coefficients))
        self.tag = 'NCPA'