# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:56:25 2023

@author: cheritier
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class OPD_map:
    
    def __init__(self,telescope, OPD = None):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        An OPD_map object consists in defining the 2D map that acts as a static OPD offset. It requires the following parameters
        _ telescope              : the telescope object that contains all the informations (diameter, pupil, etc)
                
        ************************** OPTIONAL PARAMETERS **************************
        
        _ zernike_coefficients  : a list of coefficients of Zernike Polynomials (see Zernike class). The coefficients are normalized to 1 m. 

        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ static_phase_screen.OPD       : the optical path difference

        ************************** EXEMPLE **************************
        
        1) Create a blank OPD_map object corresponding to a Telescope object
        opd = OPD_map(telescope = tel)
        
        2) Update the OPD of the OPD_map object using a given OPD_map
        opd.OPD = OPD_map
        
        3) propagate through the telescope
        src*tel*opd
        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        self.isInitialized               = False                
        if OPD is None:
            self.OPD = telescope.pupil.astype(float)
        else:
            self.OPD = OPD
            
            
        self.tag = 'OPD_map'